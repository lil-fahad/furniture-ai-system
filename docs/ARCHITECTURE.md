# Architecture

The final system is a modular monolith rather than a collection of repositories or runtime submodules.

1. `image_io.py` validates bytes, MIME type, dimensions, and image decoding.
2. `floorplan.py` extracts enclosed room regions using deterministic OpenCV operations.
3. `openai_service.py` optionally refines room semantics and creates a design brief.
4. `layout.py` places catalog items using geometric constraints and stable candidate ordering.
5. `storage.py` persists bookings in SQLite.
6. `api.py` exposes the complete system through one FastAPI application.
7. `models.py` verifies local checkpoints through one unified manifest.
8. `model_bundle.py` validates and installs allowlisted files from the external professional archive.

Large weights live under the ignored directory `models/professional/installed/`. The committed files under `models/professional/` contain only source metadata, checksums, expected sizes, revisions, licenses, and documented evaluation results.

Heavy ML dependencies are optional. The core application remains operational without PyTorch checkpoints, the professional bundle, or an OpenAI key.
