import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import yaml
import subprocess


SPLIT_KEYS = ["train", "val", "valid", "test"]


def find_yaml_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for name in ("data.yaml", "dataset.yaml"):
        files.extend(root.rglob(name))
    return files


def read_yaml(p: Path) -> dict:
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] YAML okunamadı: {p} -> {e}")
        return {}


def split_names_in_yaml(data: dict) -> List[Tuple[str, str]]:
    # returns list of (split_key, path_str)
    out: List[Tuple[str, str]] = []
    for key in SPLIT_KEYS:
        if key in data and data[key]:
            val = data[key]
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, str):
                        out.append((key, v))
            elif isinstance(val, str):
                out.append((key, val))
    return out


def labels_dir_exists_for_split(dataset_root: Path, split: str) -> Path | None:
    """Find labels directory for a given split.

    Tries both layouts: <root>/<split>/labels and <root>/labels/<split>.
    Additionally resolves val/valid synonyms bidirectionally.
    """
    candidates = [split]
    # Handle val/valid synonyms
    if split == "val":
        candidates.append("valid")
    elif split == "valid":
        candidates.append("val")

    for cand in candidates:
        p1 = dataset_root / cand / "labels"
        p2 = dataset_root / "labels" / cand
        if p1.exists():
            return p1
        if p2.exists():
            return p2
    return None


def run_remap(dataset_root: Path, split: str, backup: bool = True, force_backup: bool = False) -> int:
    cmd = [
        sys.executable,
        str(Path("tools") / "remap_single_dataset.py"),
        "--dataset-root", str(dataset_root),
        "--split", split,
    ]
    if backup:
        cmd.append("--backup")
    if force_backup:
        cmd.append("--force-backup")
    print(f"[APPLY] {dataset_root} split={split}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] Remap failed: {dataset_root} split={split}\n{proc.stderr}")
    else:
        if proc.stdout.strip():
            print(proc.stdout)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--force-backup", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] Root bulunamadı: {root}")
        sys.exit(1)

    yaml_files = find_yaml_files(root)
    if not yaml_files:
        print(f"[INFO] YAML bulunamadı: {root}")
        sys.exit(0)

    total_targets = 0
    applied = 0

    for yml in yaml_files:
        dataset_root = yml.parent
        data = read_yaml(yml)
        splits = split_names_in_yaml(data)
        # split anahtarı bazlı çalışacağız (train/val/valid/test)
        seen_split = set()
        for split_key, _ in splits:
            if split_key == "val":
                normalized = "val"
            elif split_key == "valid":
                normalized = "valid"
            else:
                normalized = split_key
            if normalized in seen_split:
                continue
            seen_split.add(normalized)

            labels_dir = labels_dir_exists_for_split(dataset_root, normalized)
            if labels_dir is None:
                print(f"[SKIP] {dataset_root} split={normalized} -> labels yok")
                continue

            total_targets += 1
            rc = run_remap(dataset_root, normalized, backup=(not args.no_backup), force_backup=args.force_backup)
            if rc == 0:
                applied += 1

    print(f"[SUMMARY] hedef split: {total_targets}, uygulanan: {applied}")


if __name__ == "__main__":
    main()
