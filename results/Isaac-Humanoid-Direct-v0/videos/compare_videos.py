#!/usr/bin/env python3
"""
make_panel.py  —  Build a 2×2 comparison figure from four Isaac-Humanoid videos.

Usage examples
--------------
python make_panel.py                       # default snapshot at 5 s
python make_panel.py --t 3.2 --crop 30     # different time + thicker crop
python make_panel.py --w 512 --h 288       # smaller panels
"""

import cv2, argparse, pathlib
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------
# 1.  Map algorithm → video file
# ---------------------------------------------------------------------
VIDEOS = {
    "DDPG": "DDPG.mp4",
    "PPO":  "PPO.mp4",
    "SAC":  "SAC.mp4",
    "TD3":  "TD3.mp4",   
    #"TRAINING_ENV": "TRAIN.mp4"   # rename or edit if yours isn’t .mp4
}

# ---------------------------------------------------------------------
def grab_frame(video_path: str, time_s: float, out_png: pathlib.Path):
    """Extract a single frame at <time_s> seconds and save to <out_png>."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Couldn’t read frame at {time_s}s in {video_path}")
    # Crop the frame by 20 pixels from each side
    h, w = frame.shape[:2]
    crop = 100
    frame = frame[crop:h-crop, crop:w-crop]
    cv2.imwrite(str(out_png), frame)
    cap.release()

# Pillow-version-agnostic helper --------------------------------------
def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Return (width, height) of rendered *text* with *font*.
    Uses textbbox() when available (Pillow ≥ 8.0) else falls back.
    """
    if hasattr(draw, "textbbox"):                          # Pillow ≥ 8.0
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    else:                                                  # very old Pillow
        return draw.textsize(text, font=font)

# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Create a 2×2 comparison panel.")
    p.add_argument("--t",     type=float, default=5.0,  help="snapshot time (s)")
    p.add_argument("--w",     type=int,   default=640,  help="panel width (px)")
    p.add_argument("--h",     type=int,   default=360,  help="panel height (px)")
    p.add_argument("--crop",  type=int,   default=20,   help="final crop (px)")
    p.add_argument("--out",   default="humanoid_alg_comparison.png",
                   help="output PNG filename")
    args = p.parse_args()

    # 2.  Extract & resize frames -------------------------------------
    tmp_dir = pathlib.Path("_frames");  tmp_dir.mkdir(exist_ok=True)
    thumbs = []
    for algo, vid in VIDEOS.items():
        frame_png = tmp_dir / f"{algo}.png"
        grab_frame(vid, args.t, frame_png)
        thumbs.append((algo, Image.open(frame_png).resize((args.w, args.h))))

    # 3.  Assemble 2×2 grid -------------------------------------------
    margin = 10
    grid_w = 2*args.w + 3*margin
    grid_h = 2*args.h + 3*margin
    canvas = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 255))

    positions = [
        (margin,               margin),
        (2*margin + args.w,    margin),
        (margin,          2*margin + args.h),
        (2*margin + args.w,2*margin + args.h),
    ]

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    for (algo, img), (x, y) in zip(thumbs, positions):
        canvas.paste(img.convert("RGBA"), (x, y))

        # ----- corner label ------------------------------------------
        draw = ImageDraw.Draw(canvas, "RGBA")
        label_xy = (x + 8, y + 8)
        text_w, text_h = text_size(draw, algo, font)
        bg = (
            label_xy[0] - 4,           label_xy[1] - 2,
            label_xy[0] + text_w + 4,  label_xy[1] + text_h + 2
        )
        draw.text(label_xy, algo, font=font, fill="white")

    # 5.  Save at 300 dpi ---------------------------------------------
    canvas.convert("RGB").save(args.out, dpi=(300, 300))
    print(f"Saved comparison figure →  {args.out}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
