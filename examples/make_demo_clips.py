"""Creates 3 tiny stick-figure clips on pure black so you can test the pipeline without your own footage.
Usage: python examples/make_demo_clips.py   (writes into ./characters)"""
import math, os, subprocess
import cv2, numpy as np

os.makedirs("characters", exist_ok=True)
W, H, FPS, SEC = 540, 960, 30, 3
COLORS = [((40, 60, 230), (20, 20, 120)), ((230, 120, 30), (90, 40, 20)), ((60, 190, 90), (20, 70, 30))]

for k, (shirt, shorts) in enumerate(COLORS):
    out = f"characters/demo_{k + 1}.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", out], stdin=subprocess.PIPE)
    for f in range(FPS * SEC):
        t = f / FPS
        img = np.zeros((H, W, 3), np.uint8)
        sway = int(14 * math.sin(t * 3 + k))
        cx = W // 2 + sway
        cv2.circle(img, (cx, 250), 62, (150, 190, 235), -1)                       # head
        cv2.circle(img, (cx, 232), 64, (15, 15, 20), 8)                           # near-black hair
        cv2.rectangle(img, (cx - 95, 320), (cx + 95, 600), shirt, -1)             # shirt
        arm = int(60 * math.sin(t * 4 + k))
        cv2.line(img, (cx + 95, 340), (cx + 170, 470 - arm), (150, 190, 235), 36)
        cv2.line(img, (cx - 95, 340), (cx - 170, 470 + arm), (150, 190, 235), 36)
        cv2.rectangle(img, (cx - 90, 600), (cx + 90, 720), shorts, -1)            # dark shorts
        cv2.rectangle(img, (cx - 85, 720), (cx - 15, 880), (150, 190, 235), -1)
        cv2.rectangle(img, (cx + 15, 720), (cx + 85, 880), (150, 190, 235), -1)
        cv2.rectangle(img, (cx - 95, 880), (cx - 5, 920), (245, 245, 245), -1)
        cv2.rectangle(img, (cx + 5, 880), (cx + 95, 920), (245, 245, 245), -1)
        p.stdin.write(cv2.GaussianBlur(img, (3, 3), 0).tobytes())
    p.stdin.close(); p.wait()
    print("wrote", out)
