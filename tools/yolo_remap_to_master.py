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
from typing import Dict, List, Tuple, Any
import json
import os
from datetime import datetime

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


def mapping_to_jsonable(dataset_path: Path, old_to_new: Dict[int, int], old_to_name: Dict[int, str], master_names: List[str]) -> Dict[str, Any]:
    """Kullanıcı müdahalesi için düzenlenebilir JSON yapı üretir."""
    items: List[Dict[str, Any]] = []
    for old_id in sorted(old_to_name.keys()):
        name = old_to_name[old_id]
        new_id = old_to_new.get(old_id, None)
        new_name = master_names[new_id] if isinstance(new_id, int) and 0 <= new_id < len(master_names) else None
        items.append({
            "old_id": old_id,
            "name": name,
            "new_id": new_id,   # kullanıcı değiştirebilir (None bırakabilir)
            "new_name": new_name
        })
    return {
        "dataset": str(dataset_path),
        "mapping": items,
        "master_names": master_names,
        "note": "new_id alanlarını düzenleyin. None ise eşlenmez (keep/skip seçiminize göre işlem görür)."
    }


def save_mapping_json(out_path: Path, data: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_mapping_json(path: Path) -> Dict[int, int]:
    """Kullanıcı tarafından düzenlenmiş JSON/YAML dosyasından old_id->new_id haritasını okur."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    result: Dict[int, int] = {}
    for item in data.get("mapping", []) or []:
        try:
            old_id = int(item.get("old_id"))
        except Exception:
            continue
        new_id = item.get("new_id")
        if isinstance(new_id, int):
            result[old_id] = new_id
    return result


def remap_file(path: Path, old_to_new: Dict[int, int], old_to_name: Dict[int, str],
               unknown_action: str = "keep", write: bool = True) -> Tuple[bool, int, int]:
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

    if changed and write:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines and not new_lines[-1].endswith("\n") else ""))

    return changed, updated, skipped


# ------------------------- Komut Satırı Arayüzü ------------------------- #

def _export_master_ids_to_configs(master_names: List[str]) -> None:
    """Master sınıf isimlerinden class id listesini JSON olarak yaz.

    Öncelik: DriveManager mevcutsa ve aktif timestamp/configs dizini tespit edilebiliyorsa
    oraya yazar. Aksi halde yerelde ./configs/class_ids.json konumuna yazar.
    """
    payload = {
        "generated_at": datetime.now().isoformat(),
        "names": master_names,
        "id_to_name": [{"id": i, "name": n} for i, n in enumerate(master_names)],
    }

    # 1) DriveManager ile dene
    out_path: Path | None = None
    try:
        from drive_manager import DriveManager  # type: ignore
        dm = DriveManager()
        if dm.authenticate() and dm.load_drive_config():
            cfg_dir = dm.get_configs_dir()
            if cfg_dir:
                out_path = Path(cfg_dir) / "class_ids.json"
    except Exception:
        # Drive entegrasyonu yoksa sessizce geç
        out_path = None

    # 2) Yerel fallback
    if out_path is None:
        out_path = Path.cwd() / "configs" / "class_ids.json"

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[BİLGİ] class_ids.json yazıldı: {out_path}")
    except Exception as e:
        print(f"[UYARI] class_ids.json yazılamadı: {e}")

def print_mapping(old_to_new: Dict[int, int], old_to_name: Dict[int, str], master_names: List[str]) -> None:
    print("Eski -> Yeni (isim)")
    for old_id in sorted(old_to_name.keys()):
        name = old_to_name[old_id]
        if old_id in old_to_new:
            new_id = old_to_new[old_id]
            new_name = master_names[new_id] if 0 <= new_id < len(master_names) else "?"
            print(f"  {old_id:>2} -> {new_id:>2}  ({name} -> {new_name})")
        else:
            print(f"  {old_id:>2} ->  --  ({name} -> <UNMAPPED>)")


def run(root: Path, unknown_action: str = "keep", backup: bool = False,
        apply: bool = False, interactive: bool = False, preview_limit: int = 5,
        export_mapping_dir: Path | None = None, use_mapping: Path | None = None) -> None:
    master_yaml = root / "master_data.yaml"
    if not master_yaml.exists():
        print(f"[HATA] master_data.yaml bulunamadı: {master_yaml}")
        sys.exit(1)

    try:
        master_names = load_yaml_names(master_yaml)
    except Exception as e:
        print(f"[HATA] master_data.yaml okunamadı: {e}")
        sys.exit(1)

    # Master sınıf id listesini Drive configs içine yazmaya çalış
    try:
        _export_master_ids_to_configs(master_names)
    except Exception as e:
        print(f"[UYARI] class_ids.json dışa aktarımı başarısız: {e}")

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

        # Kullanıcı müdahalesi: export mapping skeleton
        if export_mapping_dir is not None:
            export_path = export_mapping_dir / f"{d.name}.mapping.yaml"
            try:
                payload = mapping_to_jsonable(d, old_to_new, old_to_name, master_names)
                save_mapping_json(export_path, payload)
                print(f"[BİLGİ] Mapping dışa aktarıldı: {export_path}")
            except Exception as e:
                print(f"[UYARI] Mapping dışa aktarılamadı ({d}): {e}")

        # Kullanıcıdan mapping yükleme
        if use_mapping is not None:
            mapping_path = use_mapping
            if mapping_path.is_dir():
                cand = mapping_path / f"{d.name}.mapping.yaml"
                if cand.exists():
                    mapping_path = cand
                else:
                    print(f"[UYARI] {d.name} için mapping dosyası bulunamadı: {cand}")
                    mapping_path = None  # type: ignore
            if mapping_path and mapping_path.exists():
                try:
                    override = load_mapping_json(mapping_path)
                    # override sadece verilen old_id'leri değiştirir
                    old_to_new.update(override)
                    print(f"[BİLGİ] Mapping yüklendi ve uygulandı: {mapping_path}")
                except Exception as e:
                    print(f"[UYARI] Mapping yüklenemedi ({mapping_path}): {e}")

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

        # Ön izleme (dry-run) istatistikleri
        would_change_files = 0
        would_update_lines = 0
        would_skip_lines = 0
        examples: List[Path] = []
        for labels_dir in label_dirs:
            for lbl in iter_label_files(labels_dir):
                ch, up, sk = remap_file(lbl, old_to_new, old_to_name, unknown_action=unknown_action, write=False)
                if ch:
                    would_change_files += 1
                    if len(examples) < preview_limit:
                        examples.append(lbl)
                would_update_lines += up
                would_skip_lines += sk

        # Özet ve eşleme tablosu
        print(f"\n--- DATASET: {d} ---")
        print_mapping(old_to_new, old_to_name, master_names)
        print(f"Bulunan label dizinleri: {[str(x) for x in label_dirs]}")
        print(f"Değişecek dosya (tahmini): {would_change_files}")
        print(f"Güncellenecek satır (tahmini): {would_update_lines}")
        print(f"Atlanacak satır (tahmini): {would_skip_lines}")
        if examples:
            print("Örnek değişecek dosyalar:")
            for ex in examples:
                print(f"  - {ex}")

        # Uygulama kararı
        do_apply = False
        if apply:
            do_apply = True
        elif interactive:
            ans = input("Uygulansın mı? [y/N]: ").strip().lower()
            do_apply = ans in ("y", "yes")

        if not do_apply:
            # Bu dataset için yazma yok, genel sayaca sadece önizleme toplamlari eklemiyoruz
            continue

        # Yedekleme ve gerçek uygulama
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
                changed, updated, skipped = remap_file(lbl, old_to_new, old_to_name, unknown_action=unknown_action, write=True)
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
    parser.add_argument("--apply", action="store_true", help="Önizleme yerine değişiklikleri uygula (yaz)")
    parser.add_argument("--interactive", action="store_true", help="Her dataset için önizleme göster ve onay iste")
    parser.add_argument("--preview-limit", type=int, default=5, help="Önizlemede örnek gösterilecek dosya sayısı")
    parser.add_argument("--export-mapping-dir", type=str, default=None, help="Eşleme şablonlarını bu klasöre {dataset}.mapping.yaml olarak yaz")
    parser.add_argument("--use-mapping", type=str, default=None, help="Eşleme dosyası (yaml/json) ya da klasörü (içinde {dataset}.mapping.yaml) kullan")

    args = parser.parse_args(argv)
    run(
        root=Path(args.root).resolve(),
        unknown_action=args.unknown_action,
        backup=args.backup,
        apply=args.apply,
        interactive=args.interactive,
        preview_limit=args.preview_limit,
        export_mapping_dir=(Path(args.export_mapping_dir).resolve() if args.export_mapping_dir else None),
        use_mapping=(Path(args.use_mapping).resolve() if args.use_mapping else None),
    )


if __name__ == "__main__":
    main()
