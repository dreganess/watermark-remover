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
    h, w = img.shape[:2]
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

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

    combined = cv2.bitwise_or(region_ff, region_circ)

    full = np.zeros((h, w), np.uint8)
    full[y1:y2, x1:x2] = combined

    k = np.ones((5, 5), np.uint8)
    full = cv2.dilate(full, k, iterations=2)

    return full


def poisson_blend(original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Use Poisson blending (seamlessClone) to integrate the inpainted patch into
    the original image. The solver automatically matches colour gradients at the
    boundary — no visible seam or blur.

    Pads the image before blending so corners and edges are handled correctly,
    then crops back to the original size.
    """
    h, w = original.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return inpainted

    cx_c = int((int(xs.min()) + int(xs.max())) / 2)
    cy_c = int((int(ys.min()) + int(ys.max())) / 2)

    # Radius of the mask bounding box
    r = max(int(xs.max()) - int(xs.min()), int(ys.max()) - int(ys.min())) // 2 + 15

    # How far the centre is from the nearest edge
    margin = min(cx_c, cy_c, w - cx_c - 1, h - cy_c - 1)

    # Pad just enough so seamlessClone has room — critical for corner watermarks
    pad = max(0, r - margin + 5)
    if pad > 0:
        orig_p = cv2.copyMakeBorder(original,  pad, pad, pad, pad, cv2.BORDER_REFLECT)
        inp_p  = cv2.copyMakeBorder(inpainted, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        mask_p = cv2.copyMakeBorder(mask,      pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        center = (cx_c + pad, cy_c + pad)
        blended = cv2.seamlessClone(inp_p, orig_p, mask_p, center, cv2.NORMAL_CLONE)
        return blended[pad:pad + h, pad:pad + w]

    return cv2.seamlessClone(inpainted, original, mask, (cx_c, cy_c), cv2.NORMAL_CLONE)


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

    # Step 1: reconstruct the background texture with TELEA inpainting
    inpainted = cv2.inpaint(img, mask, inpaintRadius=22, flags=cv2.INPAINT_TELEA)

    # Step 2: Poisson-blend the patch back in — auto colour-matches the boundary
    try:
        result = poisson_blend(img, inpainted, mask)
    except Exception:
        result = inpainted  # fallback to plain inpainting if blend fails

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
