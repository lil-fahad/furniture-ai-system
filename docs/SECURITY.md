# Security

- `OPENAI_API_KEY` is read only from the process environment or GitHub Actions secrets.
- Health endpoints expose only a boolean configuration state.
- Uploaded images are limited by byte size, decoded type, MIME type, and pixel count; the pixel-count and decompression-bomb checks run before the image is decoded, and violations return `422`.
- Production mode requires a separate `SERVICE_API_KEY` of at least 24 characters for write and design endpoints (enforced at startup; `docker-compose.yml` sets `ENVIRONMENT=production`).
- CORS wildcards are rejected.
- Local model files are ignored by Git and can be verified through SHA-256 in `models/manifest.json`.
- PyTorch state dictionaries are loaded with `weights_only=True` and strict key matching.
- CI scans tracked text for common secret patterns.

A key shared in chat, an issue, a commit, or a text file must be revoked even if it is later deleted.
