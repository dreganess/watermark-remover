import base64
import cv2
import numpy as np
import io
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)
MAX_SIDE = 2048

# ── LaMa model (loaded once at startup) ───────────────────────────────────────
# simple-lama-inpainting wraps the LaMa open-source inpainting model.
# Falls back to OpenCV TELEA if the package isn't available.
_lama = None
try:
    from PIL import Image as PILImage
    from simple_lama_inpainting import SimpleLama
    _lama = SimpleLama()
    print("LaMa model loaded successfully")
except Exception as e:
    print(f"LaMa unavailable, falling back to TELEA: {e}")


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


# ── Watermark mask ─────────────────────────────────────────────────────────────

def build_mask(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """
    Build a tight mask around the clicked watermark.
    For bright watermarks (sparkles, logos): uses luminance thresholding so
    only the actual watermark pixels are masked.
    For other watermarks: falls back to colour-similarity flood fill.
    """
    h, w = img.shape[:2]
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    click_lum = int(gray[cy, cx])

    sr = 80
    x1, y1 = max(0, cx - sr), max(0, cy - sr)
    x2, y2 = min(w, cx + sr), min(h, cy + sr)
    roi   = img[y1:y2, x1:x2].copy()
    roi_g = gray[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    sx, sy = cx - x1, cy - y1

    # Method A: colour flood fill
    flood = np.zeros((rh + 2, rw + 2), np.uint8)
    cv2.floodFill(roi, flood, (sx, sy), 0,
                  (30,) * 3, (30,) * 3,
                  4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    region_ff = flood[1:-1, 1:-1]

    # Method B: luminance threshold for bright sparkle/logo watermarks
    region_lum = np.zeros((rh, rw), np.uint8)
    if click_lum > 160:
        threshold = max(click_lum - 60, 160)
        bright = (roi_g >= threshold).astype(np.uint8) * 255
        conn = np.zeros((rh + 2, rw + 2), np.uint8)
        cv2.floodFill(bright.copy(), conn, (sx, sy), 0,
                      (40,), (40,),
                      4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        region_lum = conn[1:-1, 1:-1]

    combined = cv2.bitwise_or(region_ff, region_lum)
    circle = np.zeros((rh, rw), np.uint8)
    cv2.circle(circle, (sx, sy), 70, 255, -1)
    combined = cv2.bitwise_and(combined, circle)

    if combined.sum() == 0:
        cv2.circle(combined, (sx, sy), 30, 255, -1)

    full = np.zeros((h, w), np.uint8)
    full[y1:y2, x1:x2] = combined
    full = cv2.dilate(full, np.ones((3, 3), np.uint8), iterations=1)
    return full


# ── Inpainting ─────────────────────────────────────────────────────────────────

def inpaint(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Use LaMa AI model if available, otherwise fall back to OpenCV TELEA.
    LaMa produces seamless, context-aware results even for complex backgrounds.
    """
    if _lama is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = PILImage.fromarray(img_rgb)
        pil_mask = PILImage.fromarray(mask)
        result_pil = _lama(pil_img, pil_mask)
        return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    # Fallback
    return cv2.inpaint(img_bgr, mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)


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

    cx = int(data["x"] * w)
    cy = int(data["y"] * h)

    mask = build_mask(img, cx, cy)
    result = inpaint(img, mask)

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
