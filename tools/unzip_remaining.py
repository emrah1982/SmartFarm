import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / 'datasets' / 'roboflow_dataset'


def unzip_all(target: Path) -> int:
    count = 0
    for zpath in target.rglob('*.zip'):
        try:
            out_dir = zpath.with_suffix('')
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"🗜️  Extracting: {zpath} -> {out_dir}")
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(out_dir)
            count += 1
        except Exception as e:
            print(f"❌ Failed to extract {zpath}: {e}")
    return count


def main():
    if not TARGET_DIR.exists():
        print(f"⚠️ Target directory not found: {TARGET_DIR}")
        sys.exit(0)
    extracted = unzip_all(TARGET_DIR)
    print(f"\n✅ Done. Extracted {extracted} zip file(s).")


if __name__ == '__main__':
    main()
