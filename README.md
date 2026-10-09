# Cut-Out Character Merge

Drop in many **black-background character videos** and get a vertical (9:16) video where the characters
stand in a **hand-drawn sketch landmarks + watercolor sky + grass pitch** scene, with a panning/zooming camera,
ground shadows, parallax, optional music and a handle outro. See `docs/BACKGROUND_STYLE.md` for the exact look.

## Install
```bash
pip install -r requirements.txt     # numpy + opencv
# ffmpeg must be installed and on PATH
```

## Quick start (auto layout)
```bash
# 1. put your cut-out clips (black background) in ./characters   (any number, sorted by filename)
# 2. (optional) put landmark photos in ./assets/landmarks
python -m cutmerge render --handle yourname -o output/result.mp4
```
No clips yet? `python examples/make_demo_clips.py` creates 3 test clips.

What auto layout does: intro pan over the empty scene -> camera glides to each character in turn (alternating zoom and
left/right placement) -> dark end card with your handle. Clip audio is kept; add `--music song.mp3`.

## Other commands
| Command | Use |
|---|---|
| `python -m cutmerge keytest clip.mp4` | Check black-removal on a checkerboard |
| `python -m cutmerge bg-preview` | Preview the background strip |
| `python -m cutmerge sketchify photo.png out.png` | Preview a landmark in pencil style |
| `python -m cutmerge init` | Write an example `scene.json` |
| `python -m cutmerge render --config scene.json` | Full manual control |
| `render --save-config layout.json` | Save the auto layout, then hand-edit it |

Useful `render` flags: `--intro 1.5 --outro 2 --pan 0.7 --height 0.62 --handheld 0 --shuffle --flip-alternate --screens 6 --seed 3 --key-lo 18 --key-hi 60`.

## scene.json essentials
* `characters[]`: `file`, `start` (s), `x` (0-1 position along the scene), `offset` (screens, shifts left/right of that spot), `height` (fraction of frame height at zoom 1), `flip`, `trim:[s,e]`, `foot_y`, `volume`. Several characters can overlap in time (a group shot) - give them the same `start` and different `offset`.
* `camera[]`: keyframes `{t, x, zoom, y}`, eased between keys.
* `background`: `screens`, `seed`, `landmarks[]` (`type`: image | aqueduct | gate | tower | cathedral; `x`, `height`, `layer`: far | near), `props[]` (podium | flag | image), `theme.sky_top/sky_bottom`, `chalk_lines`.
* `grade`, `vignette`, `grain`, `shadow` (`dx`, `dy`, `opacity`, `enabled`), `music`, `outro`.

## Depth (2.5D) - players in the back
* Each character has `depth` from `0` (front of the pitch) to `1` (far back near the horizon). Back characters are drawn smaller and higher on the grass, and always behind nearer ones.
* Auto layout alternates front and back players and zooms the camera in on the back ones. Turn off with `--no-depth`.
* Background layers zoom at different rates (far sky/buildings slowly, grass fastest), which gives the 3D parallax feel when the camera pushes in.
* Props with `"front": true` are drawn in front of characters (e.g. a flag the player stands behind).
* Group shot: give two characters the same `start`, different `depth` and `offset`; add a camera key with a `zoom` of 1.5-3 to push in on the back one.

## Layout
```
cutmerge/    keying.py (black -> alpha)  background.py (scene)  camera.py  render.py  cli.py
characters/  your clips      assets/landmarks/  your landmark images      docs/BACKGROUND_STYLE.md
```
