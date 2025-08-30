#!/usr/bin/env python3
# colab_drive_quick_test.py
# Colab'da Google Drive entegrasyonunu ve periyodik/final yüklemeleri hızlıca doğrulamak için mini test.
# Not: Bu betik Colab ortamında çalıştırılmak üzere tasarlanmıştır.

import os
import sys
import time
from datetime import datetime


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive_if_needed():
    if not in_colab():
        print("[UYARI] Colab dışında çalıştırıyorsunuz. Bu test Colab için tasarlanmıştır.")
        return None
    from google.colab import drive
    mount_point = "/content/drive"
    if not os.path.isdir(mount_point):
        os.makedirs(mount_point, exist_ok=True)
    drive.mount(mount_point, force_remount=True)
    base = os.path.join(mount_point, "MyDrive")
    print(f"[INFO] Drive mount edildi: {base}")
    return base


def ensure_repo_on_colab():
    # Çoğu Colab senaryosunda repo /content/SmartFarm olarak klonlanır/kopyalanır.
    # Mevcut çalışma dizininden de çalışabilir.
    repo_candidates = [
        "/content/SmartFarm",
        "/content/SmartFarm-main",
        os.getcwd(),
    ]
    for c in repo_candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "training.py")):
            print(f"[INFO] Repo bulundu: {c}")
            return c
    raise RuntimeError("Repo dizini bulunamadı. Lütfen projeyi /content içine kopyalayın veya çalışma dizininizi ayarlayın.")


def run_quick_test():
    drive_base = mount_drive_if_needed()
    repo_dir = ensure_repo_on_colab()

    sys.path.insert(0, repo_dir)

    # Gerekli modüller
    from drive_manager import DriveManager
    import model_downloader
    import training

    # 1) Drive entegrasyonunu etkinleştir
    dm = DriveManager(colab_mode=True)
    dm.activate_drive_integration(base_drive_dir=drive_base)

    # Seçilen timestamp kökü ve alt klasörleri
    ts_root = dm.active_timestamp_dir
    models_dir = os.path.join(ts_root, "models")
    logs_dir = os.path.join(ts_root, "logs")
    ckpt_dir = os.path.join(ts_root, "checkpoints")

    print("\n===== Drive Zaman Damgası Klasörleri =====")
    print("Timestamp root:", ts_root)
    print("models/:", models_dir)
    print("logs/:", logs_dir)
    print("checkpoints/:", ckpt_dir)

    # 2) Küçük bir model indir (yolo11s.pt) => otomatik olarak timestamp/models altına yönlenmeli
    print("\n[ADIM] Model indiriliyor (yolo11s detection, small)...")
    try:
        model_downloader.download_specific_model_type(
            model_type="detection",
            model_size="s",
            save_dir=ts_root,  # yönlendirme ile <ts>/models altına inmeli
            language="tr",
        )
    except Exception as e:
        print("[HATA] Model indirme başarısız:", e)

    # 3) Çok kısa bir eğitim (örnek): 6 epoch, her 3 epoch'ta checkpoint upload kontrolü.
    # training.train_model benzeri bir API varsa onu kullanın; yoksa basit training runner'ına parametre geçin.
    # Burada varsayılan dataset.yaml ile minimal bir çalışma deneriz.
    data_yaml = os.path.join(repo_dir, "dataset.yaml")
    if not os.path.isfile(data_yaml):
        print("[UYARI] dataset.yaml bulunamadı. Lütfen veri kümesini ayarlayın veya yol sağlayın.")
    else:
        print("\n[ADIM] Mini eğitim başlıyor (epoch=6). Drive periyodik yükleme aralığı=3")
        try:
            results = training.train_model(
                data_config=data_yaml,
                model_path=os.path.join(models_dir, "yolo11s.pt"),
                epochs=6,
                img_size=640,
                batch_size=4,
                device=0 if os.environ.get("COLAB_GPU") else "cpu",
                project_dir=os.path.join("/content", "SmartFarm", "runs", "train"),
                save_to_drive=True,
                drive_manager=dm,
                drive_save_interval=3,  # her 3 epoch'ta yükleme
                language="tr",
            )
            print("[INFO] Eğitim sonuçları:", results)
        except Exception as e:
            print("[HATA] Eğitim çağrısı başarısız:", e)

    # 4) Drive içeriklerini listele: models/, checkpoints/ ve logs/
    def ls(path):
        try:
            if not os.path.isdir(path):
                print(f"[YOK] {path}")
                return
            entries = sorted(os.listdir(path))
            print(f"\n[LS] {path} ({len(entries)} öge)")
            for name in entries[:50]:
                print(" -", name)
        except Exception as e:
            print(f"[HATA] Listeleme hatası {path}:", e)

    print("\n===== Drive İçeriklerinin Hızlı Kontrolü =====")
    ls(models_dir)
    ls(ckpt_dir)
    ls(os.path.join(ckpt_dir, "weights"))
    ls(logs_dir)

    # Kritik dosyalar beklenenler:
    # - <ts>/models/best.pt ve last.pt (final yedek)
    # - <ts>/checkpoints/best.pt ve last.pt (periyodik ve final)
    # - <ts>/checkpoints/weights/ (yerel runs/train/exp/weights kopyası)
    # - <ts>/logs/ indirme ve eğitim logları

    print("\n[ÖNERİ] Yukarıdaki listelerde best.pt ve last.pt dosyalarının hem models/ hem checkpoints/ altında bulunduğunu doğrulayın.")
    print("[ÖNERİ] Ayrıca checkpoints/weights/ klasörünün kopyalandığını ve logs/ altında kayıtların oluştuğunu kontrol edin.")


if __name__ == "__main__":
    run_quick_test()
