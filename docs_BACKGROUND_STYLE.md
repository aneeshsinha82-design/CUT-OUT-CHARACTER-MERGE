# The background style in your reference video

Your video is a **hand-drawn sketch city on a watercolor sky, with a real-looking grass pitch in front**.
Characters are live-action cut-outs standing on that grass, with soft shadows falling to the left.

## The recipe (what to use)

| Layer | What it looks like | How to get it |
|---|---|---|
| **Sky** | Soft light blue (#8AB6E4 at top → #D6E6F2 at horizon), watercolor mottling, faint cloud wisps, paper grain | Built in (`make_sky`). Colors editable via `background.theme.sky_top / sky_bottom`. |
| **Far landmarks** | Famous buildings of the country/theme drawn in **grey pencil outline + pale grey wash**, almost no color. Arches show blue sky through them. Lots of fine hatching. | Built-in procedural sketches (aqueduct, gate, tower, cathedral) **or your own** images in `assets/landmarks/` (auto-converted to pencil sketch). |
| **Near landmarks** | Same pencil style, bigger, slightly lower, move faster when the camera pans (parallax) | Same as above, `"layer": "near"` |
| **Ground** | Bright saturated green (#8CC45A far → #60A834 near), mow stripes, rough white chalk lines (circle, touchline) | Built in (`make_ground`). |
| **Props** | Flat orange/gold watercolor (podium with a star, giant trophy, flags). Dark thin outline. This is the only strongly saturated color besides the grass. | `podium`, `flag`, or any PNG via `{"type":"image","file":...}` |

**Key rule: the background is calm and low-contrast (grey + pastel blue); the characters are the saturated, sharp thing.** Keep all background colors except grass and props desaturated.

## Using your own landmark images

1. Pick 3-6 famous landmarks for your theme (photos or drawings). Side-on, whole building visible, landscape or portrait.
2. Best: **transparent PNG** (sky removed). Free: remove.bg, Photoshop "Remove background", Canva BG remover. Otherwise the tool tries to auto-remove a smooth sky.
3. Drop them in `assets/landmarks/`. The tool converts them to pencil sketch automatically. Preview:
   `python -m cutmerge sketchify landmark.png preview.png`
4. For props like the trophy: make a PNG with transparency, orange/gold watercolor look, and add it as an `image` prop in the config.

## AI image prompts (to make landmark/prop art)

Landmark sheet (then cut each building out with a background remover):

> Pencil sketch illustration of [Sagrada Familia / Roman aqueduct of Segovia / Torre del Oro / Puerta de Alcalá], grey graphite outline with fine cross-hatching and a pale grey wash, white paper background, no color, no people, front elevation, whole building in frame, hand-drawn travel-sketchbook style

Trophy / props:

> Watercolor illustration of a giant golden World Cup trophy, flat orange and gold washes with thin dark outline, paper texture, white background, no shadow, front view

Full painted backplate alternative (skip the procedural scene entirely):

> Vertical 9:16 illustration, hand-drawn pencil sketch of famous [country] landmarks in grey graphite on a light-blue watercolor sky, bright green football pitch with rough white chalk lines in the lower 45%, empty scene, no people, soft paper grain

## Shooting / exporting the character clips

* Pure **black** background (#000). Cut-out apps that export "black background" or "alpha over black" both work.
* Vertical 9:16 (720x1280 or higher), whole body visible if you want feet on the grass.
* Avoid **bright** reflective/black-ish halos around the edges; run `python -m cutmerge keytest clip.mp4` and check on the checkerboard.
* Dark clothing/hair is fine (holes are filled automatically). If edges look eaten: `--key-lo 10`. If you see a dark fringe/noise: `--key-lo 25 --key-hi 80`.
