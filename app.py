import base64
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
MAX_SIDE = 2048


# ── Helpers ────────────────────────────────────────────────────────────────────

def decode_b64(data_url: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    _, enc = data_url.split(",", 1)
    arr = np.frombuffer(base64.b64decode(enc), np.uint8)
    return cv2.imdecode(arr, flags)


def encode_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def limit_size(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ── Watermark detection ────────────────────────────────────────────────────────

def build_mask(img: np.ndarray, cx: int, cy: int,
               base_radius: int = 55, tolerance: int = 32) -> np.ndarray:
    """
    Given a click point (cx, cy), detect the watermark shape using:
      1. Flood-fill from the click pixel to capture connected similar pixels
      2. A guaranteed minimum circle so a mis-click still erases something
    Returns a binary mask (same HxW as img).
    """
    h, w = img.shape[:2]
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    # ROI around the click
    sr = base_radius + 40
    x1, y1 = max(0, cx - sr), max(0, cy - sr)
    x2, y2 = min(w, cx + sr), min(h, cy + sr)
    roi = img[y1:y2, x1:x2].copy()
    rh, rw = roi.shape[:2]

    sx = int(np.clip(cx - x1, 0, rw - 1))
    sy = int(np.clip(cy - y1, 0, rh - 1))

    # --- Flood fill from click ---
    flood_mask = np.zeros((rh + 2, rw + 2), np.uint8)
    diff = (tolerance,) * 3
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(roi, flood_mask, (sx, sy), 0, diff, diff, flags)
    region_ff = flood_mask[1:-1, 1:-1]

    # --- Minimum guaranteed circle ---
    region_circ = np.zeros((rh, rw), np.uint8)
    cv2.circle(region_circ, (sx, sy), base_radius, 255, -1)

    # Union: accept whichever is larger
    combined = cv2.bitwise_or(region_ff, region_circ)

    # Place into full-image mask
    full = np.zeros((h, w), np.uint8)
    full[y1:y2, x1:x2] = combined

    # Dilate so inpainting covers the semi-transparent fringe
    k = np.ones((7, 7), np.uint8)
    full = cv2.dilate(full, k, iterations=3)

    return full


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/remove", methods=["POST"])
def remove():
    data = request.get_json(force=True)

    img = decode_b64(data["image"])
    img = limit_size(img)
    h, w = img.shape[:2]

    # Coordinates are sent as 0-1 ratios relative to display canvas
    cx = int(data["x"] * w)
    cy = int(data["y"] * h)

    mask = build_mask(img, cx, cy)

    # TELEA inpainting with a generous radius for smooth reconstruction
    result = cv2.inpaint(img, mask, inpaintRadius=18, flags=cv2.INPAINT_TELEA)

    # Optional: second lighter pass with NS to smooth residual artifacts
    result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)

    return jsonify({"result": encode_b64(result)})


app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB max upload

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
