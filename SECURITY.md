# Security Policy

## Scope

FurnitureAI handles floor-plan images, supplier catalog information, model artifacts, and optional AI API calls. Security controls are treated as release requirements, not optional hardening.

## Secrets

- Never commit API keys, OAuth tokens, private keys, or credentials.
- `OPENAI_API_KEY` is read only from runtime configuration/secrets.
- Health endpoints expose configuration state only, never secret values.
- Logs, test output, and artifacts must not contain credentials.

## Input security

- Uploaded images are size- and pixel-bounded before decoding.
- Pillow decompression-bomb protection is enabled.
- Only supported image formats are accepted.
- Geometry is validated before layout operations.
- External AI output is treated as untrusted input and validated against local contracts.

## Supplier data

A supplier can be discovered before it is authorized. Discovery does not make a supplier catalog production-eligible. Product records must carry authorization and provenance information before publication.

## Model security

- Model files are SHA-256 pinned.
- Model loading uses safe weight loading where supported.
- Unverified model artifacts must not be silently promoted to production.

## Cloud and cost safety

Training jobs must perform quota, budget, and duplicate-run checks before submitting paid GPU work. A failed or uncertain submission must not be retried blindly.

## Reporting

Report suspected vulnerabilities privately to the repository owner rather than publishing exploit details in an issue. Include the affected component, reproduction steps, impact, and suggested mitigation when safe to provide them.
