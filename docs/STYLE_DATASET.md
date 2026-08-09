# Building a licensed style-classification dataset

Furniture AI uses an ImageFolder dataset for seven interior styles:

- `minimalist`
- `scandinavian`
- `industrial`
- `bohemian`
- `luxury`
- `mid_century_modern`
- `japandi`

Two import paths are available:

- `scripts/import_huggingface_styles.py` imports a large public Hugging Face
  dataset only when its Hub card declares an allowed license.
- `scripts/fetch_openverse_styles.py` creates a smaller, reviewable starter
  set from Openverse. Some Openverse deployments now require authentication,
  so a `401 Unauthorized` response means this path is unavailable anonymously.

## License and quality policy

The Hugging Face importer checks the live dataset card before downloading the
first image. It refuses datasets without a declared license and records the
dataset id, immutable revision, license, split, original row, normalized-image
SHA-256 and chosen style in `data/style_sources.jsonl`.

The importer deliberately does not map broad labels such as `Modern` or
`Japanese` to the narrower `minimalist` or `japandi` classes. If a qualified
reviewer approves a source taxonomy, pass an explicit JSON mapping through
`--label-map`. This decision remains visible in version control and the source
label remains in the manifest.

Openverse supplies attribution and license metadata, but does not verify it.
The downloader records that metadata in `data/style_sources.jsonl`; review the
file and each source page before reusing any image. The default is `cc0` only,
which is the safest starting point. `--licenses cc0,by,by-sa` broadens the
search but can create attribution and share-alike obligations. Obtain legal
review and remove unsuitable files before a commercial release.

Search-derived or mapped labels are weak labels, not human-verified ground
truth. Inspect samples, remove incorrect labels and duplicates, and keep a
human-reviewed validation and test set. A 20-image-per-style run is only a
smoke dataset, not a production-quality training corpus.

## Inspect a Hugging Face dataset

The dry run reads Hub metadata and Dataset Viewer rows, but writes no files:

```bash
cd ~/furniture-ai-system
source .venv/bin/activate
python scripts/import_huggingface_styles.py OWNER/DATASET --dry-run --max-rows 500
```

If the dataset card has no license, the command exits before image downloads.
If the dataset exposes multiple subsets, add both `--subset NAME` and
`--split train`.

After the license and taxonomy review, import all compatible rows:

```bash
python scripts/import_huggingface_styles.py OWNER/DATASET \
  --subset default \
  --split train \
  --label-map config/hf_style_label_map.example.json
```

The importer is resumable: normalized-image hashes already present in the
manifest are skipped.

## Small Openverse starter

First inspect a few candidates without writing anything:

```bash
cd ~/furniture-ai-system
source .venv/bin/activate
python scripts/fetch_openverse_styles.py --dry-run
```

Then download a small CC0-only starter set if anonymous access is available:

```bash
python scripts/fetch_openverse_styles.py --per-style 20
```

To continue an interrupted run, use the same command. Existing image counts
and Openverse identifiers in the manifest are respected.

## Train on an NVIDIA GPU

Do not place a large dataset in the small persistent Cloud Shell home disk.
Use a GPU VM or managed training job with a sufficiently large attached disk
or object-storage mount. Verify the runtime first:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

After visual review, train the classifier with mixed precision, pinned-memory
data loading, TF32 on supported NVIDIA GPUs, and stratified validation:

```bash
python training/train_room_classifier.py data/styles \
  --device cuda \
  --amp \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 8 \
  --output models/style_classifier/efficientnet_b0.pth
```

If GPU memory is exhausted, lower `--batch-size` before changing the model.
For a short CPU smoke test, use `--device cpu --epochs 1 --limit 100
--no-pretrained`.
