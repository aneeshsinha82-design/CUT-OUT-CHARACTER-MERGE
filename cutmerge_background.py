"""Builds the hand-drawn "sketch + watercolor" scene as parallax layers.

Look being reproduced:
  * light-blue watercolor sky
  * pencil-sketched landmarks (arches, towers, spires) with pale grey wash fill
  * bright green pitch with rough white chalk lines
  * flat orange/gold watercolor props (podium, flags, trophy art)

Every landmark can be a procedural sketch OR your own image (see docs/BACKGROUND_STYLE.md).
"""
import glob
import math
import os

import cv2
import numpy as np

VW, VH = 720, 1280          # default output size (9:16)
HORIZON = 0.56              # where grass starts, as fraction of scene height


# ----------------------------------------------------------------- helpers
def _fbm(h, w, rng, octaves=5, base=6):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        g = max(2, base * (2 ** o))
        gw_, gh_ = (g, max(2, int(g * h / w))) if w >= h else (max(2, int(g * w / h)), g)
        small = rng.random((gh_, gw_)).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.min()) / max(out.max() - out.min(), 1e-6)


def _rgb(r, g, b):
    return np.array([b, g, r], np.float32)


def _paste(dst_bgr, dst_a, src_bgr, src_a, x, y):
    """Alpha-over paste of a (non-premultiplied) src onto a premultiplied dst, clipped."""
    sh, sw = src_a.shape
    H, W = dst_a.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, W), min(y + sh, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = src_bgr[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = (src_a[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255.0)[..., None]
    d = dst_bgr[y0:y1, x0:x1].astype(np.float32)
    da = dst_a[y0:y1, x0:x1].astype(np.float32)[..., None] / 255.0
    na = a + da * (1 - a)
    dst_bgr[y0:y1, x0:x1] = np.clip((s * a + d * da * (1 - a)) / np.maximum(na, 1e-4), 0, 255).astype(np.uint8)
    dst_a[y0:y1, x0:x1] = (np.clip(na, 0, 1)[..., 0] * 255).astype(np.uint8)


# ----------------------------------------------------------------- pencil drawing
class Pencil:
    """Tiny wobbly-pencil canvas that yields a BGR image + alpha."""

    def __init__(self, w, h, seed=0):
        self.w, self.h = w, h
        self.lines = np.full((h, w), 255, np.uint8)
        self.mask = np.zeros((h, w), np.uint8)
        self.shade = np.zeros((h, w), np.uint8)
        self.r = np.random.default_rng(seed)

    def line(self, p0, p1, th=2, gray=70, jit=1.6):
        p0, p1 = np.array(p0, np.float32), np.array(p1, np.float32)
        n = max(2, int(np.linalg.norm(p1 - p0) / 16))
        pts = np.linspace(p0, p1, n)
        pts[1:-1] += self.r.normal(0, jit, pts[1:-1].shape)
        cv2.polylines(self.lines, [pts.astype(np.int32)], False, gray, th, cv2.LINE_AA)

    def poly(self, pts, th=2, gray=70, closed=False):
        pts = np.array(pts, np.float32)
        for i in range(len(pts) - (0 if closed else 1)):
            self.line(pts[i], pts[(i + 1) % len(pts)], th, gray)

    def fill(self, pts):
        cv2.fillPoly(self.mask, [np.array(pts, np.int32)], 255)

    def hole(self, pts):
        cv2.fillPoly(self.mask, [np.array(pts, np.int32)], 0)

    def arch_pts(self, x0, x1, y_bot, y_spring, steps=14):
        cx, rr = (x0 + x1) / 2, (x1 - x0) / 2
        arc = [(cx + rr * math.cos(math.pi - math.pi * i / steps),
                y_spring - rr * math.sin(math.pi * i / steps)) for i in range(steps + 1)]
        return [(x0, y_bot)] + arc + [(x1, y_bot)]

    def arch(self, x0, x1, y_bot, y_spring, th=2, carve=True):
        pts = self.arch_pts(x0, x1, y_bot, y_spring)
        if carve:
            self.hole(pts)
        self.poly(pts, th)

    def hatch(self, region, spacing=7, length=22, gray=150, angle=-60):
        ys, xs = np.nonzero(region > 0)
        if len(xs) == 0:
            return
        n = int(len(xs) / (spacing * spacing) * 1.2)
        idx = self.r.integers(0, len(xs), n)
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        for i in idx:
            x, y = int(xs[i]), int(ys[i])
            l = length * self.r.uniform(0.4, 1)
            cv2.line(self.lines, (x, y), (int(x + ca * l), int(y + sa * l)), int(gray + self.r.integers(-25, 25)), 1, cv2.LINE_AA)

    def render(self, wash=(236, 234, 228)):
        h, w = self.h, self.w
        noise = _fbm(h, w, self.r, 4, 5)
        grad = np.linspace(1.04, 0.90, h, dtype=np.float32)[:, None]
        base = np.array(wash[::-1], np.float32)[None, None, :] * (0.93 + 0.12 * noise[..., None]) * grad[..., None]
        ln = (self.lines.astype(np.float32) / 255.0)[..., None]
        out = np.clip(base * (0.25 + 0.75 * ln) * 1.0, 0, 255)
        out = np.where(ln < 0.55, out * 0.9, out)
        a = cv2.GaussianBlur(self.mask, (3, 3), 0)
        # pencil outlines keep full opacity even where wash alpha is 0 (e.g. thin edges)
        return out.astype(np.uint8), a


# ----------------------------------------------------------------- procedural landmarks
def _aqueduct(w, h, seed):
    p = Pencil(w, h, seed)
    tiers = 2
    th_ = h / tiers
    p.fill([(0, 0.35 * h), (w, 0.35 * h), (w, h), (0, h)])
    p.poly([(0, 0.35 * h), (w, 0.35 * h)], 3)
    p.line((0, h - 2), (w, h - 2), 3)
    for t in range(tiers):
        yb = h - t * (0.65 * h / tiers)
        yt = yb - 0.65 * h / tiers
        n = 10 if t == 0 else 14
        aw = w / n
        for i in range(n):
            x0, x1 = i * aw + aw * 0.14, (i + 1) * aw - aw * 0.14
            p.arch(x0, x1, yb - 2, yt + (x1 - x0) / 2 + 10)
        p.line((0, yt), (w, yt), 2)
    p.hatch(p.mask, 16, 14, 190, -55)
    return p.render()


def _gate(w, h, seed):
    p = Pencil(w, h, seed)
    p.fill([(0, 0.2 * h), (w, 0.2 * h), (w, h), (0, h)])
    p.poly([(0, 0.2 * h), (w, 0.2 * h), (w, h), (0, h)], 3, closed=True)
    p.poly([(0.25 * w, 0.2 * h), (0.25 * w, 0.0), (0.75 * w, 0.0), (0.75 * w, 0.2 * h)], 3)
    p.fill([(0.25 * w, 0), (0.75 * w, 0), (0.75 * w, 0.2 * h), (0.25 * w, 0.2 * h)])
    for fx in (0.12, 0.42, 0.72):
        p.arch(fx * w, (fx + 0.16) * w, h - 3, 0.58 * h)
    p.line((0, 0.32 * h), (w, 0.32 * h), 2)
    for i in range(7):
        x = (0.3 + i * 0.065) * w
        p.poly([(x, 0.0), (x + 6, -0.0 + 14), (x - 6, 14)], 2)
    p.hatch(p.mask, 14, 12, 190)
    return p.render()


def _tower(w, h, seed):
    p = Pencil(w, h, seed)
    cr = 0.08 * h
    tiers = [(0.0, 0.5, 1.0), (0.5, 0.78, 0.82), (0.78, 1.0, 0.66)]
    # tiers listed top->bottom in fractions of height; widths relative
    body = [(0.12, 1.0, 0.74), (0.04, 0.52, 0.88), (0.0, 0.0, 1.0)]
    y0, yb = 0.18 * h, h
    segs = [(0.18 * h, 0.52 * h, 0.62), (0.52 * h, 0.78 * h, 0.8), (0.78 * h, h, 1.0)]
    for ys, ye, rw in segs:
        bw = w * rw
        x0 = (w - bw) / 2
        p.fill([(x0, ys), (x0 + bw, ys), (x0 + bw, ye), (x0, ye)])
        p.poly([(x0, ys), (x0, ye)], 3)
        p.poly([(x0 + bw, ys), (x0 + bw, ye)], 3)
        p.line((x0 - 4, ye), (x0 + bw + 4, ye), 3)
    tw = w * 0.62
    tx = (w - tw) / 2
    n = 7
    for i in range(n):
        cx0 = tx + i * tw / n
        cw = tw / n * 0.55
        p.fill([(cx0, 0.18 * h - cr), (cx0 + cw, 0.18 * h - cr), (cx0 + cw, 0.18 * h), (cx0, 0.18 * h)])
        p.poly([(cx0, 0.18 * h), (cx0, 0.18 * h - cr), (cx0 + cw, 0.18 * h - cr), (cx0 + cw, 0.18 * h)], 2)
    for ys, ye, rw in segs:
        for fx in (0.3, 0.7):
            xw = w * fx
            p.arch(xw - 9, xw + 9, ys + (ye - ys) * 0.7, ys + (ye - ys) * 0.4 + 9, 2, carve=False)
    p.hatch(p.mask, 14, 16, 185, -60)
    return p.render()


def _cathedral(w, h, seed):
    p = Pencil(w, h, seed)
    rng = np.random.default_rng(seed)
    n = 4
    for i in range(n):
        tw = w / (n + 0.5)
        x0 = i * (w - tw) / (n - 1)
        top = h * (0.0 + 0.1 * rng.random() + (0.0 if i in (1, 2) else 0.18))
        pts = [(x0, h), (x0, top + (h - top) * 0.5), (x0 + tw * 0.38, top + 10),
               (x0 + tw * 0.5, top), (x0 + tw * 0.62, top + 10), (x0 + tw, top + (h - top) * 0.5), (x0 + tw, h)]
        p.fill(pts)
        p.poly(pts, 2, 60)
        for k in range(1, 8):
            yy = top + (h - top) * k / 8.5
            for j in range(3):
                xx = x0 + tw * (0.22 + 0.28 * j)
                p.poly([(xx, yy), (xx, yy + (h - top) / 14)], 2, 90)
        p.line((x0, top + (h - top) * 0.5), (x0 + tw, top + (h - top) * 0.5), 2, 90)
    p.fill([(0, 0.55 * h), (w, 0.55 * h), (w, h), (0, h)])
    p.poly([(0, 0.55 * h), (w, 0.55 * h)], 2)
    p.hatch(p.mask, 14, 20, 175, -80)
    return p.render()


_PROCEDURAL = {"aqueduct": (_aqueduct, 3.2), "gate": (_gate, 1.8), "tower": (_tower, 0.78), "cathedral": (_cathedral, 0.9)}


# ----------------------------------------------------------------- user image -> pencil sketch
def sketchify(img_bgr, alpha=None, line_strength=1.0):
    """Convert a photo/illustration of a landmark into the pencil + grey-wash look.

    Returns (bgr, alpha). If `alpha` is None the sky is auto-removed (works best on skies that are
    smooth blue/white; for perfect results supply a transparent PNG cut-out).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (0, 0), 3.2)
    dodge = cv2.divide(gray, 255 - blur + 1, scale=256).astype(np.float32)
    lines = np.clip(255 - (255 - dodge) * 1.6 * line_strength, 0, 255)
    if alpha is None:
        alpha = _auto_mask(img_bgr)
    rng = np.random.default_rng(1)
    h, w = gray.shape
    noise = _fbm(h, w, rng, 4, 5)
    wash = np.array([228, 232, 236], np.float32)[None, None, :] * (0.94 + 0.10 * noise[..., None])
    out = np.clip(wash * (0.30 + 0.70 * (lines / 255.0)[..., None]), 0, 255).astype(np.uint8)
    return out, alpha


def _auto_mask(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = cv2.GaussianBlur(cv2.Laplacian(gray, cv2.CV_32F), (0, 0), 2)
    smooth = (np.abs(grad) < 2.2).astype(np.uint8) * 255
    b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
    skyish = ((b >= r - 6) & (gray > 110)).astype(np.uint8) * 255
    cand = cv2.bitwise_and(smooth, skyish)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(cand)
    sky = np.zeros_like(cand)
    for i in range(1, n):
        comp = lab == i
        if comp[0, :].any() or comp[:, 0].any() or comp[:, -1].any():
            if comp.sum() > 0.01 * h * w:
                sky[comp] = 255
    sky = cv2.dilate(sky, np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_not(sky)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(mask, (5, 5), 0)


def load_landmark_image(path, sketch=True):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3]
        img = img[..., :3]
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if sketch:
        return sketchify(img, alpha)
    return img, alpha if alpha is not None else np.full(img.shape[:2], 255, np.uint8)


# ----------------------------------------------------------------- watercolor props
def _watercolor_rect(w, h, rgb, seed):
    rng = np.random.default_rng(seed)
    n = _fbm(h, w, rng, 4, 4)
    base = _rgb(*rgb)[None, None, :] * (0.88 + 0.22 * n[..., None])
    edge = np.zeros((h, w), np.uint8)
    cv2.rectangle(edge, (1, 1), (w - 2, h - 2), 255, 3)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.2).astype(np.float32) / 255
    out = base * (1 - 0.35 * edge[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _star(size, rgb=(250, 200, 70)):
    img = np.zeros((size, size, 3), np.uint8)
    a = np.zeros((size, size), np.uint8)
    c = size / 2
    pts = []
    for i in range(10):
        r = size * (0.48 if i % 2 == 0 else 0.20)
        ang = -math.pi / 2 + i * math.pi / 5
        pts.append((c + r * math.cos(ang), c + r * math.sin(ang)))
    cv2.fillPoly(a, [np.array(pts, np.int32)], 255)
    img[:] = _rgb(*rgb)
    cv2.polylines(img, [np.array(pts, np.int32)], True, (40, 50, 80), 3, cv2.LINE_AA)
    return img, a


def make_podium(w, h, rgb=(236, 158, 40), seed=3):
    bgr = _watercolor_rect(w, h, rgb, seed)
    a = np.full((h, w), 255, np.uint8)
    base_h = max(8, h // 6)
    bgr[h - base_h:] = _watercolor_rect(w, base_h, (226, 70, 30), seed + 1)
    s = int(min(w, h) * 0.45)
    sb, sa = _star(s)
    _paste(bgr, a, sb, sa, int(w * 0.12), int(h * 0.30))
    return bgr, a


def make_flag(w, h, colors=((196, 38, 36), (250, 190, 40), (196, 38, 36)), ratios=(1, 2, 1), seed=5, pole=True):
    pw = max(4, w // 22)
    fw, fh = w - pw, int(h * 0.38)
    img = np.zeros((h, w, 3), np.uint8)
    a = np.zeros((h, w), np.uint8)
    y = 0
    tot = sum(ratios)
    for col, r in zip(colors, ratios):
        hh = int(fh * r / tot)
        img[y:y + hh, pw:] = _rgb(*col)
        y += hh
    rng = np.random.default_rng(seed)
    n = _fbm(h, w, rng, 3, 4)
    img = np.clip(img.astype(np.float32) * (0.9 + 0.2 * n[..., None]), 0, 255).astype(np.uint8)
    # wave
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    ys = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1) + 7 * np.sin(xs / w * 6.2)
    img = cv2.remap(img, xs, ys, cv2.INTER_LINEAR)
    a[:fh + 10, pw:] = 255
    a = cv2.remap(a, xs, ys, cv2.INTER_LINEAR)
    if pole:
        img[:, :pw] = (150, 150, 150)
        a[:, :pw] = 255
    return img, a


# ----------------------------------------------------------------- ground / sky
def make_sky(w, h, seed=11, top=(138, 182, 228), bottom=(214, 230, 242)):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    grad = _rgb(*top)[None, None, :] * (1 - t) + _rgb(*bottom)[None, None, :] * t
    n = _fbm(h, w, rng, 5, 4)
    sky = grad * (0.92 + 0.14 * n[..., None])
    wisp = _fbm(h, w, rng, 4, 3)
    cloud = np.clip((wisp - 0.62) * 3.0, 0, 0.5)[..., None]
    sky = sky * (1 - cloud) + 255 * cloud
    sky += rng.normal(0, 2.5, sky.shape)
    return np.clip(sky, 0, 255).astype(np.uint8)


def make_ground(w, h, horizon, seed=21, chalk=True):
    """Returns (bgr, alpha) for the grass. y < horizon is transparent."""
    rng = np.random.default_rng(seed)
    gh = h - horizon
    t = np.linspace(0, 1, gh, dtype=np.float32)[:, None, None]
    near, far = _rgb(96, 168, 52), _rgb(140, 196, 90)
    g = far[None, None, :] * (1 - t) + near[None, None, :] * t
    n = _fbm(gh, w, rng, 6, 10)
    g = g * (0.86 + 0.28 * n[..., None])
    stripe = (np.sin(np.arange(w, dtype=np.float32) / 70.0) > 0).astype(np.float32)[None, :, None]
    g = g * (0.96 + 0.07 * stripe)
    g += rng.normal(0, 3, g.shape)
    img = np.zeros((h, w, 3), np.uint8)
    img[horizon:] = np.clip(g, 0, 255).astype(np.uint8)
    a = np.zeros((h, w), np.uint8)
    a[horizon:] = 255
    a = cv2.GaussianBlur(a, (1, 15), 0)
    if chalk:
        ch = np.zeros((h, w), np.uint8)
        r = np.random.default_rng(seed + 1)

        def stroke(pts, th):
            pts = np.array(pts, np.float32)
            pts += r.normal(0, 1.5, pts.shape)
            cv2.polylines(ch, [pts.astype(np.int32)], False, 255, th, cv2.LINE_AA)

        for cx in range(int(0.25 * w), w, int(w / 3.2)):
            ell = [(cx + 330 * math.cos(a_), horizon + 0.20 * h + 120 * math.sin(a_)) for a_ in np.linspace(0, 2 * math.pi, 70)]
            stroke(ell, 12)
            stroke([(cx, horizon + 0.20 * h - 120), (cx, h)], 12)
        stroke([(0, horizon + 0.07 * h), (w, horizon + 0.075 * h)], 9)
        rough = _fbm(h, w, r, 5, 40)
        ch = (ch.astype(np.float32) * np.clip(0.55 + rough, 0, 1)).astype(np.uint8)
        ch = cv2.GaussianBlur(ch, (0, 0), 1.6).astype(np.float32)[..., None] / 255.0
        img = np.clip(img * (1 - 0.88 * ch) + 250 * 0.88 * ch, 0, 255).astype(np.uint8)
    return img, a


# ----------------------------------------------------------------- Scene
class Layer:
    def __init__(self, bgr, alpha, parallax):
        self.parallax = parallax
        if alpha is None:
            self.pm = bgr
            self.opaque = True
        else:
            pm = (bgr.astype(np.float32) * (alpha.astype(np.float32) / 255.0)[..., None]).astype(np.uint8)
            self.pm = np.dstack([pm, alpha])
            self.opaque = False


class Scene:
    """Parallax stack: sky (slow) -> far landmarks -> near landmarks -> ground+props (1:1 with characters)."""

    def __init__(self, cfg=None, out_w=VW, out_h=VH):
        cfg = cfg or {}
        self.W, self.H = out_w, out_h
        self.screens = float(cfg.get("screens", 4))
        self.ss = float(cfg.get("supersample", 1.5))
        self.seed = int(cfg.get("seed", 7))
        self.Ws = int(self.W * self.screens)
        self.horizon = float(cfg.get("horizon", HORIZON))
        self.layers = self._build(cfg)

    # layer width so that edges line up at both ends of the camera range
    def layer_width(self, p):
        return self.W + p * (self.Ws - self.W)

    def _new_layer(self, p):
        w = int(self.layer_width(p) * self.ss)
        h = int(self.H * self.ss)
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.uint8)

    def _place(self, dst, rel_x, rel_base_y, height, bgr, alpha, layer_w_scene):
        s = self.ss
        ph = int(height * self.H * s)
        pw = int(ph * bgr.shape[1] / bgr.shape[0])
        b = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        a = cv2.resize(alpha, (pw, ph), interpolation=cv2.INTER_AREA)
        x = int(rel_x * layer_w_scene * s - pw / 2)
        y = int(rel_base_y * self.H * s - ph)
        _paste(dst[0], dst[1], b, a, x, y)

    def _build(self, cfg):
        s, W, H, Ws = self.ss, self.W, self.H, self.Ws
        rng = np.random.default_rng(self.seed)
        layers = []
        t = cfg.get("theme", {}) if isinstance(cfg.get("theme"), dict) else {}

        # sky: parallax 0.2
        p_sky = 0.2
        sw, sh = int(self.layer_width(p_sky) * s * 1.1), int(H * s)
        layers.append(Layer(make_sky(sw, sh, self.seed, tuple(t.get("sky_top", (138, 182, 228))), tuple(t.get("sky_bottom", (214, 230, 242)))), None, p_sky))

        far_p, near_p = 0.45, 0.7
        far = self._new_layer(far_p)
        near = self._new_layer(near_p)
        ground_p = 1.0
        gw, gh = int(self.Ws * s), int(H * s)
        gimg, ga = make_ground(gw, gh, int(self.horizon * gh), self.seed + 2, cfg.get("chalk_lines", True))
        ground = [gimg, ga]

        specs = cfg.get("landmarks")
        landmarks_dir = cfg.get("landmarks_dir", "assets/landmarks")
        if not specs:
            specs = self._auto_landmarks(landmarks_dir, rng)
        layer_map = {"far": (far, far_p), "near": (near, near_p)}
        for sp in specs:
            lay, p = layer_map.get(sp.get("layer", "far"), (far, far_p))
            lw = self.layer_width(p)
            typ = sp.get("type", "image")
            hh = float(sp.get("height", 0.35))
            base = float(sp.get("base_y", self.horizon + 0.02))
            if typ == "image":
                bgr, a = load_landmark_image(sp["file"], sp.get("sketch", True))
            else:
                fn, aspect = _PROCEDURAL[typ]
                ph = 520
                bgr, a = fn(int(ph * aspect), ph, int(sp.get("seed", rng.integers(0, 9999))))
            self._place(lay, sp["x"], base, hh, bgr, a, lw)
        layers.append(Layer(far[0], far[1], far_p))
        layers.append(Layer(near[0], near[1], near_p))

        for pr in cfg.get("props", self._auto_props()):
            typ = pr.get("type", "image")
            hh = float(pr.get("height", 0.2))
            base = float(pr.get("base_y", 0.9))
            ph = int(hh * H * s)
            if typ == "podium":
                aspect = float(pr.get("aspect", 2.4))
                bgr, a = make_podium(int(ph * aspect), ph, tuple(pr.get("color", (236, 158, 40))))
            elif typ == "flag":
                cols = pr.get("colors", [(196, 38, 36), (250, 190, 40), (196, 38, 36)])
                bgr, a = make_flag(int(ph * 0.75), ph, tuple(tuple(c) for c in cols), tuple(pr.get("ratios", (1, 2, 1))))
            else:
                bgr, a = load_landmark_image(pr["file"], False)
                if bgr.shape[2] == 3 and a is None:
                    a = np.full(bgr.shape[:2], 255, np.uint8)
            self._place(ground, pr["x"], base, hh, bgr, a, Ws)
        layers.append(Layer(ground[0], ground[1], ground_p))
        return layers

    def _auto_landmarks(self, landmarks_dir, rng):
        specs = []
        files = sorted(glob.glob(os.path.join(landmarks_dir, "*.png")) + glob.glob(os.path.join(landmarks_dir, "*.jpg")) + glob.glob(os.path.join(landmarks_dir, "*.jpeg")) + glob.glob(os.path.join(landmarks_dir, "*.webp")))
        kinds = list(_PROCEDURAL.keys())
        n_far = max(3, int(self.screens * 1.4))
        for i in range(n_far):
            x = (i + 0.5) / n_far
            if files and i % 2 == 0:
                specs.append({"type": "image", "file": files[(i // 2) % len(files)], "x": x, "height": 0.42, "layer": "far"})
            else:
                k = kinds[i % len(kinds)]
                specs.append({"type": k, "x": x, "height": {"aqueduct": 0.22, "gate": 0.26, "tower": 0.40, "cathedral": 0.52}[k], "layer": "far", "seed": int(rng.integers(0, 9999))})
        n_near = max(2, int(self.screens * 0.8))
        for i in range(n_near):
            x = (i + 0.3 + 0.4 * (i % 2)) / n_near
            k = ["tower", "cathedral", "gate", "aqueduct"][(i + 1) % 4]
            specs.append({"type": k, "x": min(x, 0.97), "height": {"aqueduct": 0.17, "gate": 0.2, "tower": 0.32, "cathedral": 0.4}[k], "layer": "near", "seed": int(rng.integers(0, 9999))})
        return specs

    def _auto_props(self):
        props = [{"type": "podium", "x": 0.22, "base_y": 0.80, "height": 0.13},
                 {"type": "flag", "x": 0.31, "base_y": 0.80, "height": 0.30},
                 {"type": "podium", "x": 0.70, "base_y": 0.80, "height": 0.13},
                 {"type": "flag", "x": 0.81, "base_y": 0.80, "height": 0.30}]
        return props

    # ---- rendering
    def render(self, cx_frac, zoom, cy_frac=0.5):
        """cx_frac: 0..1 horizontal position along the scene. cy_frac: vertical centre fraction."""
        W, H, z, s = self.W, self.H, zoom, self.ss
        cy = cy_frac * H
        canvas = None
        for lay in self.layers:
            lw = self.layer_width(lay.parallax)
            cxl = W / 2 + cx_frac * (lw - W)
            M = np.array([[z / s, 0, W / 2 - z * cxl], [0, z / s, H / 2 - z * cy]], np.float32)
            if lay.opaque:
                # sky is wider than needed; map by its own centre
                lw_sky = lay.pm.shape[1] / s
                cxl = lw_sky / 2 + (cx_frac - 0.5) * (self.layer_width(lay.parallax) - W)
                M = np.array([[z / s, 0, W / 2 - z * cxl], [0, z / s, H / 2 - z * cy]], np.float32)
                canvas = cv2.warpAffine(lay.pm, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                w = cv2.warpAffine(lay.pm, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                a = w[..., 3:4].astype(np.float32) / 255.0
                canvas = np.clip(w[..., :3].astype(np.float32) + canvas.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)
        return canvas
