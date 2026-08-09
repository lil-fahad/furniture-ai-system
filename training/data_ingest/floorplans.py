"""Floor-plan dataset generation (synthetic) and optional real-data fetch.

Layout contract — must match ``training/train_floorplan_segmenter.py``:
- ``<dest_dir>/images/<stem>.png`` — RGB floor-plan renders.
- ``<dest_dir>/masks/<stem>.png``  — single-channel (mode "L") class-index PNGs.
- 5 classes with mask pixel values in ``[0, 4]`` (trainer default
  ``--classes 5``, ``--mask-remap none``):

  =======  =============
  value    meaning
  =======  =============
  0        background (outside the building footprint)
  1        room floor (interior of a room)
  2        wall
  3        door opening in a wall
  4        window segment in an outer wall
  =======  =============

Generation is fully deterministic for a given ``seed`` (stdlib ``random``
only), so CI and cloud runs reproduce identical pairs.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

#: Number of segmentation classes expected by train_floorplan_segmenter.py.
NUM_CLASSES = 5
CLASS_BACKGROUND = 0
CLASS_ROOM = 1
CLASS_WALL = 2
CLASS_DOOR = 3
CLASS_WINDOW = 4

HOUSEEXPO_README_URL = (
    "https://raw.githubusercontent.com/Thomas0810/HouseExpo/master/README.md"
)


def _draw_plan(rng: random.Random, size: int) -> tuple[Image.Image, Image.Image]:
    """Draw one randomized floor plan; returns (image, mask) PIL images."""
    image = Image.new("RGB", (size, size), (235, 235, 230))
    mask = Image.new("L", (size, size), CLASS_BACKGROUND)
    draw_image = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)

    margin = rng.randint(size // 12, size // 8)
    x0, y0 = margin, margin
    x1, y1 = size - margin, size - margin
    wall = max(2, rng.randint(2, 4))

    # Partition the footprint into rooms with 1-2 vertical + 1-2 horizontal cuts.
    vertical_cuts = sorted(rng.sample(range(x0 + 20, x1 - 20), k=rng.randint(1, 2)))
    horizontal_cuts = sorted(rng.sample(range(y0 + 20, y1 - 20), k=rng.randint(1, 2)))
    columns = [x0, *vertical_cuts, x1]
    rows = [y0, *horizontal_cuts, y1]
    rooms: list[tuple[int, int, int, int]] = [
        (columns[c], rows[r], columns[c + 1], rows[r + 1])
        for c in range(len(columns) - 1)
        for r in range(len(rows) - 1)
    ]

    # Room floors.
    for index, (rx0, ry0, rx1, ry1) in enumerate(rooms):
        shade = 200 + (index * 17 + rng.randint(0, 20)) % 45
        tint = (shade, shade - rng.randint(0, 15), shade - rng.randint(0, 25))
        draw_image.rectangle((rx0, ry0, rx1, ry1), fill=tint)
        draw_mask.rectangle((rx0, ry0, rx1, ry1), fill=CLASS_ROOM)

    # Walls: full building outline plus the interior cut lines.
    wall_segments: list[tuple[int, int, int, int]] = [
        (x0, y0, x1, y0),  # top
        (x0, y1, x1, y1),  # bottom
        (x0, y0, x0, y1),  # left
        (x1, y0, x1, y1),  # right
    ]
    wall_segments += [(cx, y0, cx, y1) for cx in vertical_cuts]
    wall_segments += [(x0, cy, x1, cy) for cy in horizontal_cuts]
    for sx0, sy0, sx1, sy1 in wall_segments:
        box = (sx0 - wall // 2, sy0 - wall // 2, sx1 + wall // 2, sy1 + wall // 2)
        draw_image.rectangle(box, fill=(40, 40, 45))
        draw_mask.rectangle(box, fill=CLASS_WALL)

    outer_walls = set(range(4))

    # Doors: one opening per room on a random wall segment.
    for rx0, ry0, rx1, ry1 in rooms:
        door_len = rng.randint(size // 16, size // 10)
        if rng.random() < 0.5:  # horizontal wall
            wy = rng.choice([ry0, ry1])
            wx = rng.randint(rx0 + 2, max(rx0 + 3, rx1 - door_len - 2))
            box = (wx, wy - wall // 2 - 1, wx + door_len, wy + wall // 2 + 1)
        else:  # vertical wall
            wx = rng.choice([rx0, rx1])
            wy = rng.randint(ry0 + 2, max(ry0 + 3, ry1 - door_len - 2))
            box = (wx - wall // 2 - 1, wy, wx + wall // 2 + 1, wy + door_len)
        draw_image.rectangle(box, fill=(150, 95, 40))
        draw_mask.rectangle(box, fill=CLASS_DOOR)

    # Windows: short segments on outer walls only.
    for segment_index in outer_walls:
        if rng.random() < 0.3:
            continue
        win_len = rng.randint(size // 12, size // 8)
        sx0, sy0, sx1, sy1 = wall_segments[segment_index]
        if sy0 == sy1:  # top / bottom wall
            wx = rng.randint(min(sx0, sx1) + 2, max(sx0, sx1) - win_len - 2)
            box = (wx, sy0 - wall // 2, wx + win_len, sy0 + wall // 2)
        else:  # left / right wall
            wy = rng.randint(min(sy0, sy1) + 2, max(sy0, sy1) - win_len - 2)
            box = (sx0 - wall // 2, wy, sx0 + wall // 2, wy + win_len)
        draw_image.rectangle(box, fill=(120, 190, 235))
        draw_mask.rectangle(box, fill=CLASS_WINDOW)

    return image, mask


def generate_synthetic(
    dest_dir: Path, n: int = 200, seed: int = 7, size: int = 256
) -> dict[str, int]:
    """Generate ``n`` deterministic image/mask floor-plan pairs under ``dest_dir``.

    Returns a small summary dict, e.g. ``{"pairs": n, "images": n, "masks": n}``.
    """
    dest_dir = Path(dest_dir)
    images_dir = dest_dir / "images"
    masks_dir = dest_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    for index in range(n):
        image, mask = _draw_plan(rng, size)
        stem = f"plan_{index:05d}"
        image.save(images_dir / f"{stem}.png", format="PNG")
        mask.save(masks_dir / f"{stem}.png", format="PNG")
    return {"pairs": n, "images": n, "masks": n}


def fetch_houseexpo(dest_dir: Path, timeout: int = 30) -> Path | None:
    """Optionally fetch HouseExpo metadata; never crashes the pipeline.

    Downloads the project README as provenance/guidance. The full HouseExpo
    JSON dataset is large and not mirrored on a stable URL; when unreachable
    (or for the bulk data) we print manual instructions and return ``None``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    guidance = (
        "HouseExpo bulk data is not available via a stable direct URL.\n"
        "Manual steps: git clone https://github.com/Thomas0810/HouseExpo and\n"
        "convert its JSON floor plans into images/ + masks/ PNG pairs with\n"
        "class ids 0..4 as documented in training/data_ingest/floorplans.py."
    )
    try:
        import urllib.request

        request = urllib.request.Request(
            HOUSEEXPO_README_URL, headers={"User-Agent": "furniture-ai-data-ingest/1.3"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
    except Exception as exc:  # noqa: BLE001 - must never crash the pipeline
        print(f"warning: could not reach HouseExpo metadata ({exc})", file=sys.stderr)
        print(guidance, file=sys.stderr)
        return None
    readme_path = dest_dir / "houseexpo_README.md"
    readme_path.write_bytes(content)
    print(f"fetched HouseExpo README -> {readme_path}")
    print(guidance)
    return readme_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.data_ingest.floorplans",
        description="Generate synthetic floor-plan image/mask pairs (segmenter dataset).",
    )
    parser.add_argument("--dest", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--synthetic", type=int, default=200, help="Number of synthetic pairs")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic RNG seed")
    parser.add_argument("--size", type=int, default=256, help="Square image size in pixels")
    parser.add_argument(
        "--houseexpo", action="store_true", help="Also try to fetch HouseExpo metadata"
    )
    args = parser.parse_args(argv)

    summary = generate_synthetic(args.dest, n=args.synthetic, seed=args.seed, size=args.size)
    print(f"==> wrote {summary['pairs']} synthetic pairs under {args.dest} "
          f"(images/ + masks/, classes 0..{NUM_CLASSES - 1})")
    if args.houseexpo:
        fetch_houseexpo(args.dest / "houseexpo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
