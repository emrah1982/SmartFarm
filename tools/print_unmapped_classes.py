import yaml
from pathlib import Path

def main():
    cfg_path = Path('config_datasets.yaml')
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

    # Collect expected classes from datasets
    all_expected = set()
    for group_name, group in (cfg.get('datasets') or {}).items():
        if not isinstance(group, dict):
            continue
        for ds_name, ds in group.items():
            if isinstance(ds, dict) and isinstance(ds.get('expected_classes'), list):
                for c in ds['expected_classes']:
                    if isinstance(c, str):
                        all_expected.add(c)

    # Build mapping sources
    cm = cfg.get('class_mapping') or {}
    sub_classes_all = set()
    keywords_map = {}
    for main, info in cm.items():
        subs = set(info.get('sub_classes') or [])
        sub_classes_all |= subs
        keywords_map[main] = [k.lower() for k in (info.get('keywords') or [])]

    mapped = set()
    unmapped = set()

    for c in sorted(all_expected):
        if c in sub_classes_all:
            mapped.add(c)
            continue
        cl = c.lower()
        found = False
        for main, kws in keywords_map.items():
            if any(k in cl for k in kws):
                mapped.add(c)
                found = True
                break
        if not found:
            unmapped.add(c)

    print(f"TOTAL_EXPECTED: {len(all_expected)}")
    print(f"MAPPED: {len(mapped)}")
    print(f"UNMAPPED: {len(unmapped)}")
    if unmapped:
        print("\nUNMAPPED_LIST:")
        for c in sorted(unmapped):
            print("-", c)

if __name__ == '__main__':
    main()
