#!/usr/bin/env python3
"""
YOLO Annotation Remapper to Master Class List

Amaç:
- Farklı kaynaklardan indirilmiş YOLO datasetlerinin annotation (labels/*.txt) dosyalarını,
  üst dizindeki master_data.yaml içindeki sınıf isimlerine göre yeniden numaralandırmak.

Özet Akış:
1) --root ile verilen kök içinde master_data.yaml bulunur ve sınıf isimleri okunur.
2) root altındaki her dataset klasörü (images/, labels/, data.yaml içeren) için:
   - dataset'e ait data.yaml'dan sınıf isimleri okunur
   - eski_id -> class_name -> master_id eşlemesi çıkarılır
   - labels/**/*.txt içindeki her satırın ilk sütunu (class_id) master_id ile değiştirilir
3) Rapor: güncellenen/atlanan dosya sayıları ve uyarılar yazdırılır.

Notlar:
- Sadece ilk sütun (class_id) yeniden yazılır; diğer koordinatlar aynen korunur.
- master listede bulunmayan etiket tespit edilirse uyarı verilir ve orijinal id korunur (veri kaybını önlemek için).
- Python 3 uyumlu; pathlib ve argparse kullanır. YAML için PyYAML (yaml) kullanır.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml  # PyYAML
except ImportError as e:
    print("[HATA] PyYAML (yaml) paketini kurmanız gerekiyor: pip install pyyaml", file=sys.stderr)
    raise


# ------------------------- Yardımcı Fonksiyonlar ------------------------- #

def load_yaml_names(yaml_path: Path) -> List[str]:
    """Verilen YOLO data.yaml dosyasından sınıf isimlerini (names) okur.

    Beklenen yapılar:
    - V5/V8 tarzı: names: ["a", "b", ...]
    - Bazı varyantlarda: names: {0: "a", 1: "b", ...}
    - Bazı projelerde: classes: ["a", "b", ...] (yedek plan)
    """
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = []
    if isinstance(data.get("names"), list):
        names = data["names"]
    elif isinstance(data.get("names"), dict):
        # dict index'e göre sırala
        items = sorted(((int(k), v) for k, v in data["names"].items()), key=lambda x: x[0])
        names = [v for _, v in items]
    elif isinstance(data.get("classes"), list):
        names = data["classes"]
    else:
        raise ValueError(f"names/classes alanı bulunamadı: {yaml_path}")

    # Tüm isimleri string'e zorla
    return [str(x) for x in names]


def get_label_dirs_from_data_yaml(data_yaml: Path) -> List[Path]:
    """data.yaml içindeki train/val/test yollarından labels dizinlerini türetir.

    Mantık:
    - data.yaml'da sıklıkla 'train', 'val', 'test' alanları görüntü dizinlerine işaret eder.
    - Yol '.../images' içeriyorsa, 'images' -> 'labels' dönüşümü yapılır.
    - Yol relatif ise, data.yaml'ın bulunduğu klasörü baz alarak mutlaklaştırılır.
    - Mevcut olan benzersiz dizinler döndürülür.
    - Hiçbiri bulunamazsa boş liste döner (çağıran yer fallback yapabilir).
    """
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    base = data_yaml.parent
    label_dirs: List[Path] = []

    for key in ("train", "val", "test"):
        p = data.get(key)
        if not p:
            continue
        # Liste veya string olabilir (bazı formatlar birden fazla yol destekleyebilir)
        candidates = p if isinstance(p, list) else [p]
        for cand in candidates:
            cand_str = str(cand)
            # Relatif yolu mutlaklaştır
            cand_path = (base / cand_str).resolve() if not Path(cand_str).is_absolute() else Path(cand_str)
            # images -> labels dönüşümü
            if "images" in cand_path.parts:
                parts = list(cand_path.parts)
                parts = [("labels" if part == "images" else part) for part in parts]
                lbl_dir = Path(*parts)
            else:
                # images değilse kardeş labels klasörünü dene
                # Örn: .../train => .../labels/train
                lbl_dir = cand_path.parent / "labels" / cand_path.name
            if lbl_dir.is_dir():
                label_dirs.append(lbl_dir)

    # Benzersiz ve mevcut olanlar
    uniq: List[Path] = []
    seen = set()
    for p in label_dirs:
        if p.exists():
            rp = str(p)
            if rp not in seen:
                seen.add(rp)
                uniq.append(p)
    return uniq


def build_id_maps(dataset_names: List[str], master_names: List[str]) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Dataset sınıf isimlerinden master sınıf isimlerine id eşlemesi üretir.

    Dönüş:
    - old_to_new: eski_id -> yeni_master_id (bulunamazsa eşleme yok)
    - old_to_name: eski_id -> sınıf adı (loglama için)
    """
    name_to_master = {name: idx for idx, name in enumerate(master_names)}
    old_to_new: Dict[int, int] = {}
    old_to_name: Dict[int, str] = {}

    for old_id, name in enumerate(dataset_names):
        old_to_name[old_id] = name
        if name in name_to_master:
            old_to_new[old_id] = name_to_master[name]
    return old_to_new, old_to_name


def find_datasets(root: Path) -> List[Path]:
    """root altında images/, labels/ ve data.yaml bulunduran dataset klasörlerini bulur.
    master_data.yaml dosyasının bulunduğu kökü dataset olarak saymayız.
    """
    datasets: List[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "labels").is_dir() and (p / "images").is_dir():
            # data.yaml veya dataset.yaml ismi değişebilir; ikisini de kontrol et
            if (p / "data.yaml").exists() or (p / "dataset.yaml").exists():
                datasets.append(p)
    return datasets


def iter_label_files(labels_dir: Path) -> List[Path]:
    """labels/ altında .txt uzantılı tüm dosyaları (alt klasörler dahil) listeler."""
    return sorted(labels_dir.rglob("*.txt"))


def remap_file(path: Path, old_to_new: Dict[int, int], old_to_name: Dict[int, str],
               unknown_action: str = "keep") -> Tuple[bool, int, int]:
    """Tek bir YOLO txt label dosyasını yeniden numaralandır.

    Parametreler:
    - unknown_action: "keep" => master'da bulunmayan class'ı olduğu gibi bırak
                      "skip" => bu satırı atla (dahil etme)

    Dönüş: (changed, updated_lines, skipped_lines)
    """
    changed = False
    updated, skipped = 0, 0

    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    new_lines: List[str] = []
    for ln in lines:
        if not ln.strip():
            new_lines.append(ln)
            continue
        parts = ln.split()
        try:
            old_id = int(parts[0])
        except (ValueError, IndexError):
            # Beklenmedik satır; aynen bırak
            new_lines.append(ln)
            continue

        if old_id in old_to_new:
            new_id = old_to_new[old_id]
            if new_id != old_id:
                parts[0] = str(new_id)
                changed = True
            new_lines.append(" ".join(parts))
            updated += 1
        else:
            # Master'da bulunmayan sınıf adı
            cname = old_to_name.get(old_id, f"<unknown:{old_id}>")
            print(f"[UYARI] {path}: '{cname}' (id={old_id}) master listede bulunamadı.")
            if unknown_action == "skip":
                changed = True
                skipped += 1
                continue
            else:
                # keep
                new_lines.append(ln)

    if changed:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines and not new_lines[-1].endswith("\n") else ""))

    return changed, updated, skipped


# ------------------------- Komut Satırı Arayüzü ------------------------- #

def run(root: Path, unknown_action: str = "keep", backup: bool = False) -> None:
    master_yaml = root / "master_data.yaml"
    if not master_yaml.exists():
        print(f"[HATA] master_data.yaml bulunamadı: {master_yaml}")
        sys.exit(1)

    try:
        master_names = load_yaml_names(master_yaml)
    except Exception as e:
        print(f"[HATA] master_data.yaml okunamadı: {e}")
        sys.exit(1)

    datasets = find_datasets(root)
    if not datasets:
        print(f"[BİLGİ] Dataset klasörü bulunamadı: {root}")
        return

    total_files = 0
    changed_files = 0
    total_updated = 0
    total_skipped = 0

    for d in datasets:
        data_yaml = d / "data.yaml"
        if not data_yaml.exists():
            alt = d / "dataset.yaml"
            if alt.exists():
                data_yaml = alt
            else:
                print(f"[UYARI] data.yaml bulunamadı, atlanıyor: {d}")
                continue

        try:
            dataset_names = load_yaml_names(data_yaml)
        except Exception as e:
            print(f"[UYARI] {data_yaml} okunamadı, atlanıyor: {e}")
            continue

        old_to_new, old_to_name = build_id_maps(dataset_names, master_names)

        # data.yaml içinden labels dizinlerini topla
        label_dirs = get_label_dirs_from_data_yaml(data_yaml)
        # Hiç bulunamazsa eski davranış: dataset kökünde labels/
        if not label_dirs:
            fallback = d / "labels"
            if fallback.exists():
                label_dirs = [fallback]
            else:
                print(f"[UYARI] labels dizini bulunamadı (data.yaml üzerinden de tespit edilemedi), atlanıyor: {d}")
                continue

        # Yedekleme ve remap tüm label dizinleri için
        for labels_dir in label_dirs:
            # Yedekleme
            if backup:
                for p in iter_label_files(labels_dir):
                    bak = p.with_suffix(p.suffix + ".bak")
                    if not bak.exists():
                        try:
                            bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
                        except Exception as e:
                            print(f"[UYARI] Yedekleme başarısız ({p}): {e}")

            for lbl in iter_label_files(labels_dir):
                total_files += 1
                changed, updated, skipped = remap_file(lbl, old_to_new, old_to_name, unknown_action=unknown_action)
                if changed:
                    changed_files += 1
                total_updated += updated
                total_skipped += skipped

    # Rapor
    print("\n=== ÖZET ===")
    print(f"Kök dizin: {root}")
    print(f"Toplam label dosyası: {total_files}")
    print(f"Güncellenen dosya: {changed_files}")
    print(f"Toplam güncellenen satır: {total_updated}")
    print(f"Toplam atlanan satır: {total_skipped}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="YOLO annotation dosyalarını master_data.yaml'a göre yeniden numaralandırır.")
    parser.add_argument("--root", type=str, default="datasets",
                        help="Dataset kök dizini (master_data.yaml burada olmalı). Varsayılan: datasets")
    parser.add_argument("--unknown-action", choices=["keep", "skip"], default="keep",
                        help="Master'da olmayan sınıf bulunan satırlar: keep = olduğu gibi bırak, skip = satırı atla.")
    parser.add_argument("--backup", action="store_true", help="labels/*.txt dosyalarını .bak olarak yedekle")

    args = parser.parse_args(argv)
    run(root=Path(args.root).resolve(), unknown_action=args.unknown_action, backup=args.backup)


if __name__ == "__main__":
    main()
