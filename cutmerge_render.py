"""Composites keyed character clips onto the scene and encodes the video."""
import json
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from .background import Scene
from .camera import Camera
from .keying import key_black


# ------------------------------------------------------------------ character clip
class CharClip:
    def __init__(self, spec, key_cfg):
        self.spec = spec
        self.path = spec["file"]
        self.key = dict(lo=18, hi=60, fill_dark=True)
        self.key.update(key_cfg or {})
        self.key.update(spec.get("key", {}))
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        trim = spec.get("trim") or [0, self.n / self.fps]
        self.i0 = max(0, int(trim[0] * self.fps))
        self.i1 = min(self.n, int(trim[1] * self.fps)) if trim[1] else self.n
        self.duration = max(0.05, (self.i1 - self.i0) / self.fps)
        self.start = float(spec.get("start", 0))
        self._probe_bbox()
        self._pos = -1
        self._last_idx = None
        self._last = None

    def _probe_bbox(self):
        xs0 = ys0 = 10 ** 9
        xs1 = ys1 = -1
        for idx in np.linspace(self.i0, self.i1 - 1, 14).astype(int):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = self.cap.read()
            if not ok:
                continue
            _, a = key_black(fr, **self.key)
            ys, xs = np.nonzero(a > 100)
            if len(xs) < 50:
                continue
            xs0, xs1 = min(xs0, xs.min()), max(xs1, xs.max())
            ys0, ys1 = min(ys0, ys.min()), max(ys1, ys.max())
            self.fh, self.fw = fr.shape[:2]
        if xs1 < 0:
            raise RuntimeError(f"No subject found in {self.path}. Is the background really black? Try --key-lo lower.")
        pad = 10
        self.box = (max(0, xs0 - pad), max(0, ys0 - pad), min(self.fw, xs1 + pad), min(self.fh, ys1 + pad))
        x0, y0, x1, y1 = self.box
        self.cw, self.ch = x1 - x0, y1 - y0
        # anchor: bottom-centre of the subject's feet
        anchor = self.spec.get("anchor", "feet")
        if anchor == "center":
            self.anchor = (self.cw / 2, self.ch / 2)
        else:
            self.anchor = (self.cw / 2, self.ch - pad)
        self._pos = -1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.i0)
        self._pos = self.i0

    def get(self, local_t):
        idx = self.i0 + min(int(local_t * self.fps), self.i1 - self.i0 - 1)
        if idx == self._last_idx:
            return self._last
        if idx != self._pos:
            if idx < self._pos or idx - self._pos > 12:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._pos = idx
            else:
                while self._pos < idx:
                    self.cap.read()
                    self._pos += 1
        ok, fr = self.cap.read()
        self._pos += 1
        if not ok:
            return self._last
        x0, y0, x1, y1 = self.box
        pm, a = key_black(fr[y0:y1, x0:x1], **self.key)
        self._last_idx, self._last = idx, (pm, a)
        return self._last

    def audio_ok(self):
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", self.path],
                                 capture_output=True, text=True).stdout.strip()
            return bool(out)
        except Exception:
            return False


# ------------------------------------------------------------------ warp helpers
def _warp_roi(src, M, W, H, interp=cv2.INTER_LINEAR):
    h, w = src.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float32) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    return cv2.warpAffine(src, M2, (x1 - x0, y1 - y0), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0), x0, y0


def draw_character(canvas, clip, local_t, geom, cam, scene, shadow_cfg, opacity=1.0):
    W, H = scene.W, scene.H
    cxf, z, cyf = cam
    cx = W / 2 + cxf * (scene.Ws - W)
    cy = cyf * H
    got = clip.get(local_t)
    if got is None:
        return
    pm, a = got
    f = -1.0 if geom["flip"] else 1.0
    sc = geom["height"] * H / clip.ch                    # scene units per source pixel
    au, av = clip.anchor
    if geom["flip"]:
        au = clip.cw - au
        pm, a = pm[:, ::-1], a[:, ::-1]
    xs, ys = geom["xs"], geom["ys"]

    # ---- ground shadow (sheared, flattened silhouette that falls to the left like in the reference)
    if shadow_cfg.get("enabled", True):
        sx, sy = shadow_cfg.get("dx", 0.55), shadow_cfg.get("dy", 0.16)
        sx = sx * (-1 if geom["flip"] else 1)
        Ms = np.array([[sc * f, sx * sc, xs - sc * f * au - sx * sc * av],
                       [0, -sy * sc, ys + sy * sc * av]], np.float32)
        Ms = np.array([[z, 0, -z * cx + W / 2], [0, z, -z * cy + H / 2], [0, 0, 1]], np.float32)[:2] @ np.vstack([Ms, [0, 0, 1]])
        r = _warp_roi(a, Ms, W, H)
        if r is not None:
            sh, x0, y0 = r
            sh = cv2.GaussianBlur(sh, (0, 0), 6 * z).astype(np.float32) / 255.0
            k = shadow_cfg.get("opacity", 0.38) * opacity
            reg = canvas[y0:y0 + sh.shape[0], x0:x0 + sh.shape[1]].astype(np.float32)
            tint = np.array([40, 70, 30], np.float32)
            reg = reg * (1 - k * sh[..., None]) + tint * (k * 0.25 * sh[..., None])
            canvas[y0:y0 + sh.shape[0], x0:x0 + sh.shape[1]] = np.clip(reg, 0, 255).astype(np.uint8)

    # ---- character
    s_eff = sc * z
    src4 = np.dstack([pm, a])
    pre = 1.0
    if s_eff < 0.6:
        pre = max(s_eff * 1.3, 0.05)
        src4 = cv2.resize(src4, (max(1, int(src4.shape[1] * pre)), max(1, int(src4.shape[0] * pre))), interpolation=cv2.INTER_AREA)
    M = np.array([[z * sc * f / pre, 0, z * (xs - au * sc * f - cx) + W / 2],
                  [0, z * sc / pre, z * (ys - av * sc - cy) + H / 2]], np.float32)
    r = _warp_roi(src4, M, W, H)
    if r is None:
        return
    patch, x0, y0 = r
    al = patch[..., 3:4].astype(np.float32) / 255.0 * opacity
    pmc = patch[..., :3].astype(np.float32) * opacity
    reg = canvas[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]].astype(np.float32)
    canvas[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = np.clip(pmc + reg * (1 - al), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ finishing
def _vignette(W, H, strength):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.41
    return (1 - strength * d ** 2.2)[..., None]


def finish(frame, grade, vig, rng, grain):
    f = frame.astype(np.float32)
    sat = grade.get("saturation", 1.08)
    if sat != 1.0:
        g = f.mean(axis=2, keepdims=True)
        f = g + (f - g) * sat
    c = grade.get("contrast", 1.04)
    f = (f - 128) * c + 128
    warm = grade.get("warmth", 0.0)
    if warm:
        f[..., 2] += 8 * warm
        f[..., 0] -= 8 * warm
    if vig is not None:
        f = f * vig
    if grain:
        f += rng.normal(0, grain, f.shape[:2])[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def _text_center(frame, text, scale, alpha):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    org = ((frame.shape[1] - tw) // 2, frame.shape[0] // 2 + th // 2)
    ov = frame.copy()
    cv2.putText(ov, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0)


# ------------------------------------------------------------------ main render
def render(cfg, out_path, preview=False, progress=True):
    o = cfg.get("output", {})
    W, H, fps = int(o.get("width", 720)), int(o.get("height", 1280)), int(o.get("fps", 30))
    scene = Scene(cfg.get("background", {}), W, H)
    cam = Camera(cfg.get("camera", []), cfg.get("handheld", 0.0))
    key_cfg = cfg.get("key", {})
    chars = []
    for spec in cfg["characters"]:
        clip = CharClip(spec, key_cfg)
        h = float(spec.get("height", 0.62))
        offset = float(spec.get("offset", 0.0))
        geom = dict(height=h, flip=bool(spec.get("flip", False)),
                    xs=W / 2 + float(spec.get("x", 0.0)) * (scene.Ws - W) + offset * W,
                    ys=float(spec.get("foot_y", 0.9)) * H)
        chars.append((clip, geom, spec))

    outro = float(cfg.get("outro", 0.0))
    end = max(c.start + c.duration for c, _, _ in chars)
    total = float(o.get("duration", end + outro))
    nframes = int(total * fps)
    handle = o.get("handle") or cfg.get("handle")
    shadow_cfg = cfg.get("shadow", {})
    grade = cfg.get("grade", {})
    vig = _vignette(W, H, cfg.get("vignette", 0.18)) if cfg.get("vignette", 0.18) else None
    rng = np.random.default_rng(0)
    grain = cfg.get("grain", 1.2)
    fade = float(cfg.get("fade_in", 0.12))

    tmp_video = out_path + ".noaudio.tmp.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(o.get("crf", 18)), "-preset", "medium", tmp_video]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for fi in range(nframes):
            t = fi / fps
            c = cam.at(t)
            frame = scene.render(*c)
            for clip, geom, spec in chars:
                lt = t - clip.start
                if 0 <= lt < clip.duration:
                    op = min(1.0, lt / fade) if fade > 0 else 1.0
                    draw_character(frame, clip, lt, geom, c, scene, spec.get("shadow", shadow_cfg) if isinstance(spec.get("shadow", None), dict) else shadow_cfg, op)
            frame = finish(frame, grade, vig, rng, grain)
            if handle:
                cv2.putText(frame, "@" + handle.lstrip("@"), (W - 250, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if outro > 0 and t > end:
                u = min(1.0, (t - end) / min(outro, 0.8))
                frame = (frame.astype(np.float32) * (1 - 0.6 * u)).astype(np.uint8)
                if handle:
                    frame = _text_center(frame, "@" + handle.lstrip("@"), 1.0, u)
            proc.stdin.write(frame.tobytes())
            if progress and fi % 15 == 0:
                sys.stdout.write(f"\r  rendering {fi + 1}/{nframes} frames")
                sys.stdout.flush()
        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        raise RuntimeError("ffmpeg failed; is it installed and on PATH?")
    if progress:
        print()

    mux_audio(tmp_video, out_path, chars, cfg, total)
    if os.path.exists(tmp_video):
        os.remove(tmp_video)
    return out_path


def mux_audio(video, out, chars, cfg, total):
    inputs, filters, labels = [], [], []
    i = 1
    if cfg.get("use_clip_audio", True):
        for clip, _, spec in chars:
            if clip.audio_ok():
                t0 = (spec.get("trim") or [0, 0])[0]
                inputs += ["-i", clip.path]
                d = int(clip.start * 1000)
                filters.append(f"[{i}:a]atrim={t0}:{t0 + clip.duration},asetpts=PTS-STARTPTS,volume={spec.get('volume', 1.0)},adelay={d}|{d}[a{i}]")
                labels.append(f"[a{i}]")
                i += 1
    music = cfg.get("music")
    if music:
        inputs += ["-stream_loop", "-1", "-i", music]
        filters.append(f"[{i}:a]atrim=0:{total},volume={cfg.get('music_volume', 0.6)}[a{i}]")
        labels.append(f"[a{i}]")
        i += 1
    if not labels:
        os.replace(video, out)
        return
    mix = "".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=longest,atrim=0:{total}[aout]"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video] + inputs + ["-filter_complex", ";".join(filters + [mix]),
           "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("  audio mix failed; writing video without audio")
        os.replace(video, out)
