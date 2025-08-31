import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


def _normalize_name(s: str) -> str:
    return ''.join(ch.lower() for ch in s.strip().replace('-', ' ').replace('_', ' ').split())


def load_alias_canonical_map(aliases_path: Path) -> Dict[str, str]:
    if not aliases_path.exists():
        return {}
    with aliases_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    mapping: Dict[str, str] = {}
    alias_root = data.get('aliases', {}) or {}
    for canonical, variants in alias_root.items():
        if not isinstance(variants, list):
            continue
        for v in variants:
            mapping[_normalize_name(str(v))] = str(canonical)
        # canonical ismi de kendisine eşle
        mapping[_normalize_name(str(canonical))] = str(canonical)
    return mapping


def load_dataset_names(dataset_root: Path) -> List[str]:
    data_yaml = dataset_root / 'data.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError(f'data.yaml bulunamadı: {data_yaml}')
    with data_yaml.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    names = data.get('names')
    if isinstance(names, dict):
        # {id:name}
        names = [v for k, v in sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])]
    if not isinstance(names, list):
        raise ValueError('data.yaml içinde names listesi bulunamadı')
    return [str(n) for n in names]


def build_old_to_new_map(dataset_names: List[str], aliases_map: Dict[str, str], class_ids_path: Path) -> Tuple[Dict[int, int], Dict[str, int]]:
    with class_ids_path.open('r', encoding='utf-8') as f:
        cid = json.load(f)
    new_names: List[str] = cid['names']
    name_to_new_id: Dict[str, int] = {n: i for i, n in enumerate(new_names)}

    mapping: Dict[int, int] = {}
    for old_id, raw_name in enumerate(dataset_names):
        norm = _normalize_name(raw_name)
        canonical = aliases_map.get(norm, raw_name)
        # Eğer canonical da alias içinde kendine dönmüyorsa normalize et ve aynı kalır
        new_id = name_to_new_id.get(canonical)
        if new_id is None:
            # Son çare: normalize eşleşme dene
            # normalize isimleri haritala
            norm_to_new = {_normalize_name(n): i for i, n in enumerate(new_names)}
            new_id = norm_to_new.get(_normalize_name(canonical))
        if new_id is None:
            # Bulunamadıysa, veri kaybını önlemek için eski index'i koru
            new_id = old_id
        mapping[old_id] = int(new_id)
    return mapping, {n: i for i, n in enumerate(new_names)}


def remap_labels(labels_dir: Path, id_map: Dict[int, int], backup: bool) -> Tuple[int, int, List[Path]]:
    changed_files = 0
    total_files = 0
    touched_spider_files: List[Path] = []

    for txt in sorted(labels_dir.glob('*.txt')):
        total_files += 1
        orig = txt.read_text(encoding='utf-8', errors='ignore').strip().splitlines()
        new_lines: List[str] = []
        changed = False
        contains_new_spider = False
        for line in orig:
            if not line.strip():
                continue
            parts = line.split()
            try:
                cls = int(float(parts[0]))
            except Exception:
                new_lines.append(line)
                continue
            new_cls = id_map.get(cls, cls)
            if new_cls != cls:
                changed = True
            parts[0] = str(new_cls)
            new_lines.append(' '.join(parts))
        if changed:
            if backup:
                shutil.copy2(txt, txt.with_suffix('.txt.bak'))
            txt.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
            changed_files += 1
        # Spider dosyası tespiti yeni id üzerinden sonra yapılacak (dışarıdan geçilecek)
    return total_files, changed_files, []


def list_spider_files(labels_dir: Path, spider_id: int) -> List[Path]:
    spider_files: List[Path] = []
    for txt in sorted(labels_dir.glob('*.txt')):
        hit = False
        for line in txt.read_text(encoding='utf-8', errors='ignore').splitlines():
            if not line.strip():
                continue
            parts = line.split()
            try:
                cls = int(float(parts[0]))
            except Exception:
                continue
            if cls == spider_id:
                hit = True
                break
        if hit:
            spider_files.append(txt)
    return spider_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    ap.add_argument('--backup', action='store_true')
    args = ap.parse_args()

    root = Path(args.dataset_root)
    labels_dir = root / args.split / 'labels'
    if not labels_dir.exists():
        raise FileNotFoundError(f'Labels klasörü bulunamadı: {labels_dir}')

    aliases_path = Path('config') / 'class_aliases.yaml'
    class_ids_path = Path('config') / 'class_ids.json'

    dataset_names = load_dataset_names(root)
    aliases_map = load_alias_canonical_map(aliases_path)
    id_map, new_name_to_id = build_old_to_new_map(dataset_names, aliases_map, class_ids_path)

    total, changed, _ = remap_labels(labels_dir, id_map, backup=args.backup)

    spider_id = new_name_to_id.get('Spider')
    spider_files = []
    if spider_id is not None:
        spider_files = list_spider_files(labels_dir, spider_id)
        out_path = Path('config') / 'spider_files.yaml'
        data = {
            'dataset': str(root),
            'split': args.split,
            'spider_id': spider_id,
            'files': [str(p.relative_to(Path('.'))) if (Path('.') in p.parents) else str(p) for p in spider_files],
        }
        out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')

    print(f"[BİLGİ] Toplam dosya: {total}, Değişen: {changed}")
    print(f"[BİLGİ] Spider (id={spider_id}) dosya sayısı: {len(spider_files)}")


if __name__ == '__main__':
    main()
