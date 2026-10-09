import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

from .background import Scene, load_landmark_image
from .keying import checker, key_black

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")


def find_clips(folder):
    files = [f for f in sorted(glob.glob(os.path.join(folder, "*"))) if f.lower().endswith(VIDEO_EXT)]
    if not files:
        sys.exit(f"No videos found in '{folder}'. Put your black-background character clips there.")
    return files


def _probe_duration(path, trim=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return (n / (fps or 30.0))


def auto_config(files, a):
    n = len(files)
    screens = a.screens or max(3, min(n + 1, 8))
    zooms = [1.0, 1.14, 1.0, 1.28]
    depths = [0.0, 0.55, 0.15, 0.7]
    intro, pan = a.intro, a.pan
    cfg = {
        "output": {"width": 720, "height": 1280, "fps": 30, "handle": a.handle},
        "background": {"screens": screens, "seed": a.seed, "landmarks_dir": a.landmarks, "chalk_lines": True},
        "handheld": a.handheld,
        "outro": a.outro,
        "grade": {"saturation": 1.08, "contrast": 1.04, "warmth": 0.3},
        "characters": [], "camera": [],
    }
    if a.music:
        cfg["music"] = a.music
    t = intro
    xs = [0.5] if n == 1 else [i / (n - 1) for i in range(n)]
    H = 1280
    keys = []
    x0 = xs[0]
    keys.append({"t": 0.0, "x": min(1.0, x0 + 0.22), "zoom": 1.0, "y": 0.5})
    keys.append({"t": max(intro, 0.01), "x": x0, "zoom": 1.0, "y": 0.5})
    for i, f in enumerate(files):
        dur = _probe_duration(f)
        height = a.height
        d = depths[i % len(depths)] if a.depth else 0.0
        if d > 0:
            sc_ = 1 - 0.72 * d
            z = min(3.0, max(1.0, 0.95 / sc_))
            ys = 0.9 + (0.61 - 0.9) * d
            cyf = max(0.5 / z, min(1 - 0.5 / z, ys - height * sc_ / 2))
        else:
            z = zooms[i % len(zooms)]
            cyf = max(0.5, min(1 - 0.5 / z, (0.9 * H - (H / 2 - 70) / z) / H)) if z > 1 else 0.5
        offset = [-0.1, 0.1, 0.0, -0.12, 0.12][i % 5] if a.stagger else 0.0
        cfg["characters"].append({"file": f, "start": round(t, 3), "x": round(xs[i], 4), "offset": offset, "height": height,
                                  "flip": bool(a.flip_alternate and i % 2), "depth": d})
        if i > 0:
            keys.append({"t": round(t, 3), "x": xs[i - 1], "zoom": zooms[(i - 1) % len(zooms)], "y": 0.5})
        keys.append({"t": round(t + (pan if i > 0 else 0.01), 3), "x": xs[i], "zoom": z, "y": round(cyf, 3)})
        keys.append({"t": round(t + dur, 3), "x": xs[i], "zoom": z, "y": round(cyf, 3)})
        t += dur
    cfg["camera"] = keys
    return cfg


def cmd_render(a):
    if a.config:
        cfg = json.load(open(a.config))
        if a.handle:
            cfg.setdefault("output", {})["handle"] = a.handle
    else:
        files = find_clips(a.characters)
        if a.shuffle:
            random.Random(a.seed).shuffle(files)
        cfg = auto_config(files, a)
    cfg.setdefault("key", {})
    if a.key_lo is not None:
        cfg["key"]["lo"] = a.key_lo
    if a.key_hi is not None:
        cfg["key"]["hi"] = a.key_hi
    if a.music:
        cfg["music"] = a.music
    if a.save_config:
        json.dump(cfg, open(a.save_config, "w"), indent=2)
        print("saved layout ->", a.save_config)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    from .render import render
    render(cfg, a.output)
    print("done ->", a.output)


def cmd_keytest(a):
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    outs = []
    for idx in np.linspace(0, max(n - 1, 0), 3).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok:
            continue
        pm, al = key_black(fr, a.lo, a.hi)
        bg = checker(*fr.shape[:2])
        comp = pm.astype(np.float32) + bg.astype(np.float32) * (1 - al[..., None] / 255.0)
        outs.append(cv2.resize(np.clip(comp, 0, 255).astype(np.uint8), (360, int(360 * fr.shape[0] / fr.shape[1]))))
    cv2.imwrite(a.output, np.hstack(outs))
    print("wrote", a.output, "- the subject should sit cleanly on the checkerboard")


def cmd_bgpreview(a):
    sc = Scene({"screens": a.screens, "seed": a.seed, "landmarks_dir": a.landmarks})
    frames = [sc.render(x, 1.0, 0.5) for x in np.linspace(0, 1, 4)]
    cv2.imwrite(a.output, np.hstack([cv2.resize(f, (360, 640)) for f in frames]))
    print("wrote", a.output)


def cmd_sketchify(a):
    bgr, al = load_landmark_image(a.input, True)
    out = np.dstack([bgr, al if al is not None else np.full(bgr.shape[:2], 255, np.uint8)])
    cv2.imwrite(a.output, out)
    print("wrote", a.output, "(PNG with transparent sky) - put it in assets/landmarks/ as-is is NOT needed; the tool sketchifies on load")


def cmd_init(a):
    cfg = {
        "output": {"width": 720, "height": 1280, "fps": 30, "handle": "yourhandle"},
        "background": {
            "screens": 4, "seed": 7, "chalk_lines": True,
            "landmarks": [
                {"type": "image", "file": "assets/landmarks/sagrada.png", "x": 0.55, "height": 0.55, "layer": "far"},
                {"type": "aqueduct", "x": 0.2, "height": 0.24, "layer": "far"},
                {"type": "tower", "x": 0.8, "height": 0.40, "layer": "near"}
            ],
            "props": [
                {"type": "podium", "x": 0.5, "base_y": 0.82, "height": 0.14},
                {"type": "flag", "x": 0.6, "base_y": 0.95, "height": 0.30, "front": True},
                {"type": "flag", "x": 0.38, "base_y": 0.82, "height": 0.30, "colors": [[196, 38, 36], [250, 190, 40], [196, 38, 36]]}
            ]
        },
        "camera": [
            {"t": 0, "x": 0.1, "zoom": 1.0}, {"t": 2, "x": 0.1, "zoom": 1.0},
            {"t": 8, "x": 0.5, "zoom": 1.15, "y": 0.55}
        ],
        "characters": [
            {"file": "characters/player1.mp4", "start": 2.0, "x": 0.1, "offset": 0.0, "height": 0.62, "flip": False},
            {"file": "characters/player2.mp4", "start": 8.0, "x": 0.5, "offset": 0.1, "height": 0.7, "depth": 0.6, "trim": [0, 5]}
        ],
        "grade": {"saturation": 1.08, "contrast": 1.04, "warmth": 0.3},
        "outro": 2.0
    }
    json.dump(cfg, open(a.output, "w"), indent=2)
    print("wrote", a.output)


def main():
    p = argparse.ArgumentParser(prog="cutmerge", description="Merge black-background character videos into a sketch-style scene.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="make the video")
    r.add_argument("-c", "--characters", default="characters", help="folder of black-background clips (auto layout)")
    r.add_argument("--config", help="use a hand-written JSON scene instead of auto layout")
    r.add_argument("-o", "--output", default="output/result.mp4")
    r.add_argument("--handle", help="social handle shown as watermark/outro")
    r.add_argument("--landmarks", default="assets/landmarks", help="folder of landmark images (optional)")
    r.add_argument("--music")
    r.add_argument("--intro", type=float, default=1.5, help="seconds of empty scene panning at the start")
    r.add_argument("--outro", type=float, default=2.0, help="seconds of dark end card")
    r.add_argument("--pan", type=float, default=0.7, help="camera pan time between characters")
    r.add_argument("--height", type=float, default=0.62, help="character height as fraction of frame")
    r.add_argument("--screens", type=int, help="scene width in screens (auto by default)")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--handheld", type=float, default=1.0, help="0 = locked-off camera")
    r.add_argument("--shuffle", action="store_true")
    r.add_argument("--no-depth", dest="depth", action="store_false", help="keep every character on the front line (no far-back players)")
    r.add_argument("--stagger", action="store_true", default=True, help="alternate characters left/right of centre")
    r.add_argument("--flip-alternate", action="store_true")
    r.add_argument("--key-lo", type=int)
    r.add_argument("--key-hi", type=int)
    r.add_argument("--save-config", help="write the auto layout to JSON so you can hand-edit it")
    r.set_defaults(fn=cmd_render)

    k = sub.add_parser("keytest", help="check how well a clip keys out")
    k.add_argument("video")
    k.add_argument("-o", "--output", default="output/keytest.png")
    k.add_argument("--lo", type=int, default=18)
    k.add_argument("--hi", type=int, default=60)
    k.set_defaults(fn=cmd_keytest)

    b = sub.add_parser("bg-preview", help="render a strip showing the background")
    b.add_argument("-o", "--output", default="output/background.png")
    b.add_argument("--screens", type=int, default=4)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--landmarks", default="assets/landmarks")
    b.set_defaults(fn=cmd_bgpreview)

    s = sub.add_parser("sketchify", help="preview a landmark photo converted to pencil sketch")
    s.add_argument("input")
    s.add_argument("output")
    s.set_defaults(fn=cmd_sketchify)

    i = sub.add_parser("init", help="write an example scene.json")
    i.add_argument("-o", "--output", default="scene.json")
    i.set_defaults(fn=cmd_init)

    args = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(getattr(args, "output", "."))) or ".", exist_ok=True)
    args.fn(args)
