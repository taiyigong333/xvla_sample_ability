"""
Generate an X-VLA meta.json from a folder of HDF5 files.

Typical usage for scheme B:

python doc/5_generate_meta_from_hdf5_folder.py ^
  --input_dir F:/data/xvla/ur_set_a ^
  --output_path F:/research/vla_team/X-VLA/meta/robomind_ur_multi/robomind_ur_set_a.json ^
  --dataset_name robomind-ur-set-a ^
  --robot_type robomind-ur ^
  --observation_key observations/images/cam_high ^
  --language_instruction_key language_instruction

This script only generates the meta JSON file. It does not modify HDF5 files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py


def _iter_hdf5_files(input_dir: Path, recursive: bool) -> list[Path]:
    patterns = ("*.hdf5", "*.h5")
    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(input_dir.rglob(pattern))
        else:
            files.extend(input_dir.glob(pattern))
    files = sorted({p.resolve() for p in files if p.is_file()})
    return files


def _has_dataset(f: h5py.File, key: str) -> bool:
    try:
        obj = f[key]
        return isinstance(obj, h5py.Dataset)
    except KeyError:
        return False


def _validate_one_file(
    path: Path,
    observation_keys: Iterable[str],
    language_instruction_key: str,
    allow_language_attr: bool,
) -> list[str]:
    errors: list[str] = []
    with h5py.File(path, "r") as f:
        if not _has_dataset(f, "puppet/end_effector"):
            errors.append("missing dataset: puppet/end_effector")
        if not _has_dataset(f, "puppet/joint_position"):
            errors.append("missing dataset: puppet/joint_position")

        for key in observation_keys:
            if not _has_dataset(f, key):
                errors.append(f"missing dataset: {key}")

        if not _has_dataset(f, language_instruction_key):
            if allow_language_attr and language_instruction_key in f.attrs:
                pass
            else:
                errors.append(
                    f"missing language dataset: {language_instruction_key}"
                    + (" (attribute exists but dataset is still required by current training code)"
                       if language_instruction_key in f.attrs else "")
                )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate X-VLA meta.json from a folder of HDF5 files."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .hdf5/.h5 files")
    parser.add_argument("--output_path", type=str, required=True, help="Output meta.json path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Meta dataset_name, e.g. robomind-ur-set-a")
    parser.add_argument("--robot_type", type=str, default="robomind-ur", help="Robot type, e.g. robomind-ur")
    parser.add_argument(
        "--observation_key",
        type=str,
        nargs="+",
        default=["observations/images/cam_high"],
        help="One or more HDF5 image dataset keys",
    )
    parser.add_argument(
        "--language_instruction_key",
        type=str,
        default="language_instruction",
        help="HDF5 dataset key for language instruction",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively search input_dir for HDF5 files",
    )
    parser.add_argument(
        "--allow_language_attr",
        action="store_true",
        default=False,
        help="Allow language key to exist as an HDF5 attribute during validation. "
             "Note: current training code still expects a dataset, not an attribute.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Abort if any file fails validation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output_path).resolve()
    observation_keys = list(args.observation_key)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist or is not a directory: {input_dir}")

    files = _iter_hdf5_files(input_dir, recursive=args.recursive)
    if not files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found under: {input_dir}")

    invalid: list[tuple[Path, list[str]]] = []
    valid_paths: list[str] = []

    for path in files:
        errors = _validate_one_file(
            path=path,
            observation_keys=observation_keys,
            language_instruction_key=args.language_instruction_key,
            allow_language_attr=args.allow_language_attr,
        )
        if errors:
            invalid.append((path, errors))
        else:
            valid_paths.append(path.as_posix())

    if invalid:
        print("Validation warnings/errors:")
        for path, errors in invalid:
            print(f"- {path.as_posix()}")
            for err in errors:
                print(f"  - {err}")
        if args.strict:
            raise RuntimeError("Validation failed in strict mode.")

    if not valid_paths:
        raise RuntimeError("No valid HDF5 files passed validation.")

    meta = {
        "dataset_name": args.dataset_name,
        "robot_type": args.robot_type,
        "observation_key": observation_keys,
        "language_instruction_key": args.language_instruction_key,
        "datalist": valid_paths,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Found {len(files)} HDF5 files.")
    print(f"Valid files written to meta: {len(valid_paths)}")
    print(f"Output meta path: {output_path.as_posix()}")
    if invalid and not args.strict:
        print(f"Skipped invalid files: {len(invalid)}")


if __name__ == "__main__":
    main()
