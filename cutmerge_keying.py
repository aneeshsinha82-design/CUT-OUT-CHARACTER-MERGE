"""Black-background keying: turns a character shot on pure black into RGBA."""
import cv2
import numpy as np

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def key_black(frame_bgr, lo=18, hi=60, fill_dark=True):
    """Return (premultiplied_bgr uint8, alpha uint8).

    lo/hi  : brightness range that fades from transparent to opaque.
    fill_dark : keeps genuinely dark parts of the subject (black hair, navy shorts)
                opaque by filling holes inside the silhouette.
    """
    mx = frame_bgr.max(axis=2).astype(np.float32)
    a = np.clip((mx - lo) / max(hi - lo, 1), 0, 1)
    a = a * a * (3 - 2 * a)

    if fill_dark:
        m = (mx > lo + 6).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K7)
        # drop compression specks: keep blobs that are a decent fraction of the biggest one
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            biggest = st[1:, cv2.CC_STAT_AREA].max()
            keep = np.zeros(n, np.uint8)
            keep[1:] = (st[1:, cv2.CC_STAT_AREA] >= 0.03 * biggest).astype(np.uint8) * 255
            m = keep[lab]
        h, w = m.shape
        pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        ffmask = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(pad, ffmask, (0, 0), 255)
        enclosed = (pad[1:-1, 1:-1] == 0)
        solid = np.where(enclosed | (m > 0), 255, 0).astype(np.uint8)
        solid = cv2.erode(solid, _K3)
        solid = cv2.GaussianBlur(solid, (5, 5), 0).astype(np.float32) / 255.0
        # specks outside the kept blobs must not leak through the luma alpha
        a = np.maximum(a * (m > 0), solid)

    # un-multiply soft edges so dark fringes don't show against a bright background
    af = np.clip(a, 0, 1)
    soft = (af > 0.02) & (af < 0.98)
    col = frame_bgr.astype(np.float32)
    col[soft] = np.clip(col[soft] / af[soft, None], 0, 255)
    pm = col * af[..., None]
    return pm.astype(np.uint8), (af * 255).astype(np.uint8)


def checker(h, w, s=24):
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // s) + (xx // s)) % 2) * 40 + 110
    return np.dstack([c, c, c]).astype(np.uint8)
