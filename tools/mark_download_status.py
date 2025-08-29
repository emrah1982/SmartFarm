import sys
from pathlib import Path
from datetime import datetime
import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / 'config_datasets.yaml'
DATASET_DIR = ROOT / 'datasets' / 'roboflow_dataset'

# Project import
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roboflow_api_helper import download_from_config_yaml  # type: ignore


def iter_datasets(cfg: dict):
    datasets_root = (cfg or {}).get('datasets') or {}
    for group, items in datasets_root.items():
        if not isinstance(items, dict):
            continue
        for name, entry in items.items():
            if isinstance(entry, dict) and entry.get('roboflow_canonical'):
                yield group, name, entry


def main():
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cfg = yaml.safe_load(YAML_PATH.read_text(encoding='utf-8'))

    results = []
    changed = False

    for group, name, entry in iter_datasets(cfg):
        ok = False
        err = None
        try:
            ok = bool(
                download_from_config_yaml(
                    name,
                    yaml_path=str(YAML_PATH),
                    dataset_dir=str(DATASET_DIR),
                    format_name='yolov11'
                )
            )
        except Exception as e:
            ok = False
            err = str(e)
        # Mark status into YAML entry
        entry['last_download_status'] = 'OK' if ok else 'FAIL'
        entry['last_download_time'] = ts
        if err:
            entry['last_download_error'] = err
        else:
            entry.pop('last_download_error', None)
        changed = True
        results.append((group, name, ok))
        print(f"{group}.{name}: {'OK' if ok else 'FAIL'}")

    if changed:
        YAML_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
        print(f"\nYAML updated with last_download_status at {ts}")

    # Summary
    ok_count = sum(1 for _,_,k in results if k)
    fail_count = sum(1 for _,_,k in results if not k)
    print(f"\nSUMMARY -> OK: {ok_count}, FAIL: {fail_count}, TOTAL: {len(results)}")


if __name__ == '__main__':
    main()
