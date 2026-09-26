"""Kit colour, appearance embeddings and jersey-number OCR."""
from __future__ import annotations
import logging
import re

import cv2
import numpy as np

log = logging.getLogger("appearance")


def _clip_box(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    return x1, y1, x2, y2


def torso_crop(frame, box):
    """Central chest region of a player box (mostly shirt, little background)."""
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    cx1, cy1, cx2, cy2 = _clip_box(frame, x1 + 0.22 * w, y1 + 0.18 * h, x2 - 0.22 * w, y1 + 0.55 * h)
    if cx2 - cx1 < 6 or cy2 - cy1 < 8:
        return None
    return frame[cy1:cy2, cx1:cx2]


def kit_colour(crop) -> np.ndarray | None:
    """Median CIE-Lab colour of the torso crop. The median is robust to the small share of background pixels."""
    if crop is None or crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)


def colour_hist(crop) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten().astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else None


# ----------------------------------------------------------------------------------------------------------------- encoders
class ColorHistEncoder:
    """Cheap appearance descriptor. NOT discriminative between players of the same team, so identity never relies on it."""
    name = "color_hist"
    is_discriminative = False

    def encode(self, frame, box) -> np.ndarray | None:
        return colour_hist(torso_crop(frame, box))


class OsnetEncoder:
    """Deep re-identification embedding (OSNet via torchreid). Optional; install `torchreid` to use REID_ENCODER=osnet."""
    name = "osnet_x1_0"
    is_discriminative = True

    def __init__(self, device: str):
        from torchreid.utils import FeatureExtractor  # type: ignore
        self.ex = FeatureExtractor(model_name="osnet_x1_0", model_path="", device=device or "cpu")

    def encode(self, frame, box) -> np.ndarray | None:
        x1, y1, x2, y2 = _clip_box(frame, *box)
        if x2 - x1 < 12 or y2 - y1 < 24:
            return None
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        f = self.ex([crop])[0].cpu().numpy().astype(np.float32)
        n = np.linalg.norm(f)
        return f / n if n > 0 else None


def make_encoder(cfg):
    if cfg.reid_encoder == "osnet":
        return OsnetEncoder(cfg.device)
    return ColorHistEncoder()


# ------------------------------------------------------------------------------------------------------------------- OCR
_DIGITS = re.compile(r"^\d{1,2}$")


class EasyOcrReader:
    """Reads jersey numbers from the upper body. Numbers are only visible on some frames (back/chest facing camera),
    so results are aggregated by voting per track, never trusted from a single frame."""

    def __init__(self, device: str):
        import easyocr  # type: ignore
        self.reader = easyocr.Reader(["en"], gpu=(device not in ("", "cpu")) or _cuda(), verbose=False)

    def read(self, frame, box) -> list[tuple[int, float]]:
        x1, y1, x2, y2 = box
        h = y2 - y1
        cx1, cy1, cx2, cy2 = _clip_box(frame, x1, y1 + 0.10 * h, x2, y1 + 0.62 * h)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return []
        scale = max(1.0, 160.0 / crop.shape[0])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        out = []
        for _bbox, text, conf in self.reader.readtext(gray, allowlist="0123456789", detail=1, paragraph=False):
            text = text.strip()
            if _DIGITS.match(text) and 1 <= int(text) <= 99 and conf >= 0.35:
                out.append((int(text), float(conf)))
        return out


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def make_ocr(cfg):
    if cfg.jersey_ocr == "easyocr":
        return EasyOcrReader(cfg.device)
    return None
