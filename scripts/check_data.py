#!/usr/bin/env python3
"""Check local dataset paths, optionally verifying bundled-data SHA-256 hashes."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEN_DATA = "Generation/itspm_generation/table1_data"

# Each dataset is satisfied by any one complete group of paths.
DATASETS = {
    "p12-classification": [[
        "data/P12data/processed_data/PTdict_list.npy",
        "data/P12data/processed_data/arr_outcomes.npy",
        "data/P12data/splits/phy12_split1.npy",
    ]],
    "pam": [[
        "data/PAMAP2data/processed_data/PTdict_list.npy",
        "data/PAMAP2data/processed_data/arr_outcomes.npy",
        "data/PAMAP2data/splits/PAMAP2_split_1.npy",
    ]],
    "p19": [[
        "data/P19data/processed_data/PT_dict_list_6.npy",
        "data/P19data/processed_data/arr_outcomes_6.npy",
        "data/P19data/splits/phy19_split1_new.npy",
    ]],
    "p12-regression": [
        [f"data/physionet/processed/set-{part}_0.0.pt" for part in "abc"],
        [f"data/physionet/raw/set-{part}.tar.gz" for part in "abc"],
    ],
    "activity": [["data/activity/processed/data.pt"],
                 ["data/activity/raw/ConfLongDemo_JSI.txt"]],
    "ushcn": [["data/ushcn/ushcn.pt"], ["data/ushcn/processed/ushcn.pt"],
              ["data/ushcn/raw/small_chunked_sporadic.csv"]],
    "mimic": [["data/mimic/mimic.pt"], ["data/mimic/processed/mimic.pt"],
              ["data/mimic/raw/full_dataset.csv"]],
    "taxi": [[f"TPP/taxi/{split}.pkl" for split in ("train", "dev", "test")]],
    "stocks": [[f"{GEN_DATA}/stock_data.csv"]],
    "energy": [[f"{GEN_DATA}/energy_data.csv"]],
    "mujoco": [[f"{GEN_DATA}/mujoco_training_36.pt"]],
    "sines": [[]],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bundled", "all", *DATASETS], default="bundled")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to check.")
    parser.add_argument("--verify", action="store_true", help="Also hash every bundled file.")
    args = parser.parse_args(argv)
    selected = (["activity", "taxi", "mujoco"] if args.dataset == "bundled" else
                list(DATASETS) if args.dataset == "all" else [args.dataset])
    ok = True
    for name in selected:
        alternatives = DATASETS[name]
        present = any(all((args.root / path).is_file() for path in group) for group in alternatives)
        print(f"{'OK' if present else 'MISSING'} {name}")
        if not present:
            ok = False
            for index, group in enumerate(alternatives, 1):
                missing = [path for path in group if not (args.root / path).is_file()]
                print(f"  Option {index}: supply " + ", ".join(missing))
    if args.verify:
        manifest = json.loads((args.root / "DATA_MANIFEST.json").read_text())
        for relative, expected in manifest.items():
            path = args.root / relative
            if not path.is_file():
                print(f"MISSING bundled file: {relative}")
                ok = False
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            valid = path.stat().st_size == expected["bytes"] and digest == expected["sha256"]
            print(f"{'VERIFIED' if valid else 'MISMATCH'} {relative}")
            ok = ok and valid
    if not ok:
        print("See docs/DATASETS.md for sources and preparation instructions.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
