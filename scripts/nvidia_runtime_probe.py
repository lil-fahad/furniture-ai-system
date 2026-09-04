from __future__ import annotations

import argparse
import json

from furniture_ai.nvidia_acceleration import resolve_nvidia_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Report FurnitureAI NVIDIA inference capability.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()

    profile = resolve_nvidia_runtime(
        args.device,
        precision=args.precision,
        enable_torch_compile=args.torch_compile,
    )
    print(json.dumps(profile.as_public_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
