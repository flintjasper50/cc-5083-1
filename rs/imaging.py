"""Image IO and validity visualization."""
import base64
import io

import numpy as np
from PIL import Image

from .corrector import (REASON_BEHIND, REASON_BOUNDS, REASON_FOLD, REASON_MOVING,
                        REASON_POSE)

REASON_COLORS = [
    (REASON_MOVING, (240, 220, 60)),   # yellow: independent motion (unsupported)
    (REASON_FOLD, (60, 200, 220)),     # cyan: local folding
    (REASON_BEHIND, (180, 80, 220)),   # purple: behind camera / parallel to plane
    (REASON_POSE, (255, 160, 40)),     # orange: outside pose coverage
    (REASON_BOUNDS, (230, 60, 60)),    # red: out of input bounds
]
VALID_COLOR = (38, 38, 38)


def to_uint8(img):
    return np.clip(np.asarray(img) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def save_png(path, img):
    Image.fromarray(to_uint8(img)).save(path)


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def validity_image(reasons):
    out = np.tile(np.array(VALID_COLOR, dtype=np.uint8), reasons.shape + (1,))
    for bit, color in REASON_COLORS:
        out[(reasons & bit) != 0] = color
    return out


def png_data_uri(img):
    arr = img if img.dtype == np.uint8 else to_uint8(img)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
