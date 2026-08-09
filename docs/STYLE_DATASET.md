# Building a style-classification dataset

`scripts/fetch_openverse_styles.py` creates a small, reviewable starter
dataset for the seven interior styles used by Furniture AI:

- `minimalist`
- `scandinavian`
- `industrial`
- `bohemian`
- `luxury`
- `mid_century_modern`
- `japandi`

The script queries [Openverse](https://openverse.org/), which indexes openly
licensed media. It does not use OpenAI, require an API key, or launch a Google
Cloud job.

## License and quality review

Openverse supplies attribution and license metadata, but does not verify it.
The downloader records that metadata in `data/style_sources.jsonl`; review the
file and each source page before reusing any image. The default is `cc0` only,
which is the safest starting point. `--licenses cc0,by,by-sa` broadens the
search but can create attribution and share-alike obligations. Obtain legal
review and remove unsuitable files before a commercial release.

The labels are search-derived, not human-verified. Treat the output as a
starter set: inspect images, remove incorrect labels and duplicates, and add
images that you own or have licensed. A 20-image-per-style run is only a smoke
dataset, not a production-quality training corpus.

## Run from Cloud Shell

First inspect a few candidates without writing anything:

```bash
cd ~/furniture-ai-system
source .venv/bin/activate
python scripts/fetch_openverse_styles.py --dry-run
```

Then download a small CC0-only starter set:

```bash
python scripts/fetch_openverse_styles.py --per-style 20
```

To continue an interrupted run, use the same command. Existing image counts
and Openverse identifiers in the manifest are respected.

After visual review, train the classifier:

```bash
python training/train_room_classifier.py data/styles \
  --epochs 20 \
  --batch-size 16 \
  --output models/style_classifier/efficientnet_b0.pth
```

For a short CPU smoke test, use `--epochs 1 --limit 100 --no-pretrained`.
