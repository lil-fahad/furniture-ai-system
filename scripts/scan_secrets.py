from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "OpenAI project key": re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    "OpenAI legacy key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "Private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
IGNORED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}

findings: list[str] = []
for path in ROOT.rglob("*"):
    if not path.is_file() or any(part in IGNORED for part in path.parts):
        continue
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".glb", ".pth", ".pt"}:
        continue
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for label, pattern in PATTERNS.items():
        if pattern.search(content):
            findings.append(f"{label}: {path.relative_to(ROOT)}")
if findings:
    raise SystemExit("Potential secrets detected:\n" + "\n".join(findings))
print("No committed secret patterns detected")
