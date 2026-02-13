#!/usr/bin/env python3
"""
Package a Parakeet CTC ONNX export into a Handy-compatible model directory.

Output directory structure:
  <out_dir>/
    model.onnx
    vocab.txt
    config.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


def extract_vocab_from_nemo(nemo_path: Path, out_vocab_path: Path) -> None:
    if not nemo_path.exists():
        raise FileNotFoundError(f"Missing .nemo checkpoint: {nemo_path}")

    with tarfile.open(nemo_path, "r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        vocab_member = None

        for candidate_name in ("tokenizer.vocab", "vocab.txt"):
            vocab_member = next(
                (member for member in members if Path(member.name).name == candidate_name),
                None,
            )
            if vocab_member is not None:
                break

        if vocab_member is None:
            available = ", ".join(sorted(Path(member.name).name for member in members))
            raise FileNotFoundError(
                f"Could not find tokenizer.vocab/vocab.txt inside {nemo_path}. "
                f"Archive files: {available}"
            )

        extracted = archive.extractfile(vocab_member)
        if extracted is None:
            raise RuntimeError(f"Failed to read {vocab_member.name} from {nemo_path}")

        out_vocab_path.write_bytes(extracted.read())


def write_default_config(config_path: Path) -> None:
    config = {
        "model_file": "model.onnx",
        "vocab_file": "vocab.txt",
        "sample_rate": 16000,
        "features": 80,
        "n_fft": 512,
        "window_size": 0.025,
        "window_stride": 0.01,
        "normalize": "per_feature",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Handy Parakeet CTC model bundle (model.onnx + vocab.txt + config.json)."
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to INT8/FP32 ONNX file to deploy",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (will be created if needed)",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="Optional existing vocab.txt path (if omitted, extracted from --nemo)",
    )
    parser.add_argument(
        "--nemo",
        default=None,
        help="Path to .nemo checkpoint (used to extract vocab when --vocab is not provided)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model.onnx/vocab.txt/config.json files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX file: {onnx_path}")

    if args.vocab is None and args.nemo is None:
        raise ValueError("Provide either --vocab or --nemo to supply vocab.txt")

    out_dir.mkdir(parents=True, exist_ok=True)

    out_model = out_dir / "model.onnx"
    out_vocab = out_dir / "vocab.txt"
    out_config = out_dir / "config.json"

    if not args.force:
        existing = [path for path in (out_model, out_vocab, out_config) if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Refusing to overwrite existing files ({joined}). Re-run with --force."
            )

    shutil.copy2(onnx_path, out_model)

    if args.vocab:
        vocab_path = Path(args.vocab).expanduser().resolve()
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab path: {vocab_path}")
        shutil.copy2(vocab_path, out_vocab)
    else:
        nemo_path = Path(args.nemo).expanduser().resolve()
        extract_vocab_from_nemo(nemo_path, out_vocab)

    write_default_config(out_config)

    print("[INFO] Handy CTC bundle ready:")
    for path in (out_model, out_vocab, out_config):
        print(f"  - {path} ({path.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
