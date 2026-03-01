import base64
import cv2
import numpy as np
import io
from flask import Flask, jsonify, render_template, request, send_file

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
               base_radius: int = 38, tolerance: int = 32) -> np.ndarray:
    """
    Build a tight mask around the clicked watermark using flood-fill plus
    a guaranteed minimum circle. Smaller radius and dilation mean less
    background gets erased, giving the inpainting algorithm less to reconstruct.
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

    # Lighter dilation — just enough to cover semi-transparent fringe
    k = np.ones((5, 5), np.uint8)
    full = cv2.dilate(full, k, iterations=2)

    return full


def smooth_edges(result: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a gentle Gaussian blur in a narrow ring just inside and outside
    the mask boundary. This hides the hard seam between inpainted and
    original pixels without affecting the interior or the rest of the image.
    """
    k = np.ones((7, 7), np.uint8)
    inner = cv2.erode(mask, k, iterations=2)
    outer = cv2.dilate(mask, k, iterations=2)
    boundary = (outer > 0) & (inner == 0)          # ring around the mask edge

    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=2.5)

    out = result.copy()
    out[boundary] = blurred[boundary]
    return out


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


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

    # TELEA inpainting — larger radius searches further for matching texture
    result = cv2.inpaint(img, mask, inpaintRadius=22, flags=cv2.INPAINT_TELEA)

    # Smooth the boundary ring so the patch blends into the surrounding image
    result = smooth_edges(result, mask)

    return jsonify({"result": encode_b64(result)})


@app.route("/api/download", methods=["POST"])
def download():
    data = request.get_json(force=True)
    _, enc = data["image"].split(",", 1)
    img_bytes = base64.b64decode(enc)
    return send_file(
        io.BytesIO(img_bytes),
        mimetype="image/png",
        as_attachment=True,
        download_name="watermark-removed.png",
    )


app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB max upload

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
