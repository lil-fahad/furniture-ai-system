from __future__ import annotations

import argparse
import json
from pathlib import Path

from furniture_ai.model_bundle import (
    DEFAULT_INSTALL_ROOT,
    DEFAULT_SPEC_PATH,
    install_bundle,
    validate_bundle_spec,
    verify_bundle_archive,
    verify_installed_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install or verify the professional model bundle")
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--destination", type=Path, default=DEFAULT_INSTALL_ROOT)
    parser.add_argument("--check-spec", action="store_true")
    parser.add_argument("--verify-archive", action="store_true")
    parser.add_argument("--verify-installed", action="store_true")
    args = parser.parse_args()

    if args.check_spec:
        result = validate_bundle_spec(args.spec)
    elif args.verify_installed:
        result = verify_installed_bundle(args.destination, args.spec)
    elif args.verify_archive:
        if args.archive is None:
            parser.error("archive is required with --verify-archive")
        result = verify_bundle_archive(args.archive, args.spec)
    else:
        if args.archive is None:
            parser.error("archive is required for installation")
        result = install_bundle(args.archive, args.destination, args.spec).to_dict()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
