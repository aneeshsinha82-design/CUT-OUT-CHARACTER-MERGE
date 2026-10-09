"""Keyframed camera: x (0..1 along the scene), zoom, y (0..1 vertical centre)."""
import math


def _ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


class Camera:
    def __init__(self, keys, handheld=0.0):
        self.keys = sorted([{"t": 0.0, "x": 0.0, "zoom": 1.0, "y": 0.5, **k} for k in keys], key=lambda k: k["t"]) or [
            {"t": 0.0, "x": 0.0, "zoom": 1.0, "y": 0.5}]
        self.handheld = handheld

    def at(self, t):
        ks = self.keys
        if t <= ks[0]["t"]:
            k = ks[0]
            x, z, y = k["x"], k["zoom"], k["y"]
        elif t >= ks[-1]["t"]:
            k = ks[-1]
            x, z, y = k["x"], k["zoom"], k["y"]
        else:
            for a, b in zip(ks, ks[1:]):
                if a["t"] <= t <= b["t"]:
                    u = _ease((t - a["t"]) / max(b["t"] - a["t"], 1e-6))
                    x = a["x"] + (b["x"] - a["x"]) * u
                    z = math.exp(math.log(a["zoom"]) + (math.log(b["zoom"]) - math.log(a["zoom"])) * u)
                    y = a["y"] + (b["y"] - a["y"]) * u
                    break
        if self.handheld:
            h = self.handheld
            z *= 1 + 0.006 * h * math.sin(t * 1.3)
            x += 0.004 * h * math.sin(t * 0.9 + 1)
            y += 0.003 * h * math.sin(t * 1.7)
        return min(max(x, 0.0), 1.0), z, y
