import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roboflow_api_helper import download_from_config_yaml

YAML_PATH = ROOT / 'config_datasets.yaml'
DATASET_DIR = 'datasets'


def iter_dataset_names(cfg: dict):
    datasets_root = (cfg or {}).get('datasets') or {}
    for group, items in datasets_root.items():
        if not isinstance(items, dict):
            continue
        for name, entry in items.items():
            # Yalnızca SDK ile indirilecek kayıtları döndür (roboflow_canonical zorunlu)
            if isinstance(entry, dict) and 'roboflow_canonical' in entry:
                yield group, name


def main():
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    results = []
    for group, name in iter_dataset_names(cfg):
        ok = False
        try:
            ok = bool(download_from_config_yaml(name, yaml_path=str(YAML_PATH), dataset_dir=DATASET_DIR, format_name='yolov11'))
        except Exception as e:
            print(f"ERROR {group}.{name}: {e}")
            ok = False
        results.append((group, name, ok))

    print('\n=== RESULTS ===')
    for g, n, ok in results:
        print(f"{g}.{n}: {'OK' if ok else 'FAIL'}")


if __name__ == '__main__':
    sys.exit(main())
