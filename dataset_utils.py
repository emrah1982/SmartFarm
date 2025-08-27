#!/usr/bin/env python3
# dataset_utils.py - Dataset management for YOLO11 training

import os
import yaml
import urllib.request
import zipfile
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import requests

# dataset_utils.py dosyasındaki download_dataset fonksiyonunu bu şekilde değiştirin:

def build_roboflow_download_url(src_url: str, api_key: str | None, split_config: dict | None) -> str:
    """Build a robust Roboflow download URL for both Universe and API endpoints.

    Supported patterns:
    - Universe DS link: https://universe.roboflow.com/ds/<hash> (use key=, optional split)
    - Universe download link: https://universe.roboflow.com/<ws>/<project>/download/<version>/<format>
      (ensure format in path; append key= and optional split)
    - API endpoint: https://api.roboflow.com/... (use api_key= and format=yolov8)
    """
    parsed = urlparse(src_url)
    host = parsed.netloc.lower()

    # Helper: build split string
    split_str = None
    if split_config and all(k in split_config for k in ("train", "test", "val")):
        split_str = f"{split_config['train']}-{split_config['test']}-{split_config['val']}"

    # Case 1: Universe
    if "universe.roboflow.com" in host:
        path = parsed.path.rstrip('/')
        existing_q = parse_qs(parsed.query)

        # a) DS short link: /ds/<hash>
        if path.startswith("/ds/"):
            # Keep path as-is, merge query params
            params = dict((k, v[:]) for k, v in existing_q.items())
            # Ensure format is present (yolov8 default)
            if "format" not in params:
                params["format"] = ["yolov8"]
            if api_key:
                params["key"] = [api_key]
            if split_str:
                params["split"] = [split_str]
            new_query = urlencode({k: v[0] for k, v in params.items()}, doseq=True)
            new_url = urlunparse((parsed.scheme, parsed.netloc, path, "", new_query, ""))
            return new_url

        # b) Download link: /<ws>/<project>/download/<version>[/<format>]
        if "/download/" in path:
            parts = path.split('/')
            # Ensure last part is a known format; default to yolov8
            known_formats = {"yolov5", "yolov7", "yolov8", "yolo", "voc", "coco"}
            if parts[-1] not in known_formats:
                parts.append("yolov8")
                path = "/".join(filter(None, parts))

            params = dict((k, v[:]) for k, v in existing_q.items())
            if api_key:
                params["key"] = [api_key]
            if split_str:
                params["split"] = [split_str]
            new_query = urlencode({k: v[0] for k, v in params.items()}, doseq=True)
            new_url = urlunparse((parsed.scheme, parsed.netloc, path, "", new_query, ""))
            return new_url

        # Fallback for other universe links: pass through with key/split
        params = dict((k, v[:]) for k, v in existing_q.items())
        if api_key:
            params["key"] = [api_key]
        if split_str:
            params["split"] = [split_str]
        new_query = urlencode({k: v[0] for k, v in params.items()}, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, path, "", new_query, ""))

    # Case 2: API endpoint
    if "api.roboflow.com" in host:
        path = parsed.path
        q = parse_qs(parsed.query)
        q["format"] = ["yolov8"]  # Use YOLOv8 by default
        if api_key:
            q["api_key"] = [api_key]
        if split_str:
            q["split"] = [split_str]
        new_query = urlencode({k: v[0] for k, v in q.items()}, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, path, "", new_query, ""))

    # Case 3: Unknown host – keep as-is, optionally append api_key/split
    q = parse_qs(parsed.query)
    if api_key and "api_key" not in q and "key" not in q:
        # Prefer 'api_key' for non-universe unknowns
        q["api_key"] = [api_key]
    if split_str:
        q["split"] = [split_str]
    new_query = urlencode({k: v[0] for k, v in q.items()}, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", new_query, ""))


def _resolve_universe_ds_to_canonical(ds_url: str, session: requests.Session | None = None) -> tuple[str | None, str | None, str | None]:
    """Try to resolve a universe short link /ds/<hash> to (workspace, project, version) by reading the HTML.
    Returns (workspace, project, version) or (None, None, None) if it cannot be resolved.
    """
    try:
        sess = session or requests.Session()
        resp = sess.get(ds_url, timeout=60, allow_redirects=True)
        if resp.status_code != 200:
            return (None, None, None)
        html = resp.text
        import re
        # Look for canonical download link pattern: /<ws>/<project>/download/<version>
        m = re.search(r"https?://universe\.roboflow\.com/([\w-]+)/([\w-]+)/download/(\d+)", html)
        if not m:
            # Also try dataset page form: /<ws>/<project>/dataset/(\d+)
            m2 = re.search(r"https?://universe\.roboflow\.com/([\w-]+)/([\w-]+)/dataset/(\d+)", html)
            if not m2:
                return (None, None, None)
            return (m2.group(1), m2.group(2), m2.group(3))
        return (m.group(1), m.group(2), m.group(3))
    except Exception:
        return (None, None, None)


def _build_api_endpoint_url(workspace: str, project: str, version: str, api_key: str, split_config: dict | None) -> str:
    """Build Roboflow API download endpoint using workspace/project/version and api_key."""
    from urllib.parse import urlencode
    q = {
        "api_key": api_key,
        "format": "yolov8",
    }
    if split_config and all(k in split_config for k in ("train", "test", "val")):
        q["split"] = f"{split_config['train']}-{split_config['test']}-{split_config['val']}"
    # API path: https://api.roboflow.com/dataset/{workspace}/{project}/{version}
    # Some docs show dataset identifier as {workspace}/{project}
    path = f"/dataset/{workspace}/{project}/{version}"
    return f"https://api.roboflow.com{path}?{urlencode(q)}"

def download_dataset(url, dataset_dir='datasets/roboflow_dataset', api_key=None, split_config=None):
    """Download YOLO formatted dataset from Roboflow with improved error handling and API key support"""
    print(f'📥 Dataset indiriliyor: {url}')

    # Create target directory
    os.makedirs(dataset_dir, exist_ok=True)

    # Build robust download URL
    download_url = build_roboflow_download_url(url, api_key, split_config)
    # ÖNLEYİCİ DÖNÜŞÜM: Universe kısa bağlantıları (/ds/<hash>) çoğunlukla HTML döndürür.
    # ZIP'i garantiye almak için mümkünse API endpoint URL'sine dönüştürelim.
    try:
        parsed0 = urlparse(download_url)
        if "universe.roboflow.com" in parsed0.netloc and parsed0.path.startswith("/ds/"):
            q0 = parse_qs(parsed0.query)
            effective_key = api_key or (q0.get("key", [None])[0])
            if effective_key:
                ws, prj, ver = _resolve_universe_ds_to_canonical(url)
                if ws and prj and ver:
                    api_like = _build_api_endpoint_url(ws, prj, ver, effective_key, split_config)
                    print(f"🔁 DS link API endpoint'e dönüştürüldü: {api_like[:100]}...")
                    download_url = api_like
    except Exception:
        pass
    if api_key:
        print(f"🔑 API key aktif (ilk 10): {api_key[:10]}...")
    if split_config:
        print(f"📊 Bölümleme: train={split_config.get('train')} test={split_config.get('test')} val={split_config.get('val')}")
    
    zip_path = os.path.join(dataset_dir, 'dataset.zip')

    # Try multiple download methods
    for attempt in range(3):
        try:
            print(f"Download attempt {attempt + 1}/3")
            
            # Method 1: requests library (more reliable)
            import requests
            # Browser benzeri headers (public dataset için)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
            print(f"🔗 İndirme URL'si: {download_url[:100]}...")
            
            # Session kullanarak cookie ve redirect yönetimi
            session = requests.Session()
            session.headers.update(headers)
            # Bazı Roboflow yönlendirmeleri Referer bekleyebilir
            try:
                session.headers["Referer"] = url
            except Exception:
                pass
            
            response = session.get(download_url, timeout=300, stream=True, allow_redirects=True)
            print(f"🔎 HTTP durum: {response.status_code}")
            
            # Detaylı hata kontrolü
            if response.status_code == 403:
                print(f"❌ 403 Forbidden Hatası!")
                print(f"🔑 Muhtemel nedenler:")
                print(f"   • API anahtarı eksik veya geçersiz")
                print(f"   • Dataset private ve erişim izni yok")
                print(f"   • Rate limit aşıldı")
                print(f"💡 Çözüm önerileri:")
                print(f"   • API key ekleyin: download_dataset(url, api_key='your_key')")
                print(f"   • Dataset'in public olduğundan emin olun")
                print(f"   • Birkaç dakika bekleyip tekrar deneyin")
                # 403 için format fallback denemesi (universe + ds/download linkleri)
                from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
                parsed_dl = urlparse(download_url)
                if "universe.roboflow.com" in parsed_dl.netloc:
                    q = parse_qs(parsed_dl.query)
                    current_format = q.get("format", ["yolov8"])[0]
                    alt_format = "yolov5" if current_format != "yolov5" else "yolov8"
                    q["format"] = [alt_format]
                    # Ayrıca split parametresini kaldırıp deneyelim
                    if "split" in q:
                        del q["split"]
                    new_query = urlencode({k: v[0] for k, v in q.items()}, doseq=True)
                    download_url = urlunparse((parsed_dl.scheme, parsed_dl.netloc, parsed_dl.path, "", new_query, ""))
                    print(f"🔁 403 sonrası alternatif format ile yeniden denenecek: format={alt_format}")
                    # Bir sonraki döngü denemesine geç
                    continue
                # Eğer API key varsa, universe ds linklerini API endpoint'e çözümlemeyi dene
                if api_key:
                    ws, prj, ver = (None, None, None)
                    try:
                        ws, prj, ver = _resolve_universe_ds_to_canonical(url, session)
                    except Exception:
                        pass
                    if ws and prj and ver:
                        api_url = _build_api_endpoint_url(ws, prj, ver, api_key, split_config)
                        print(f"🔁 403 sonrası API endpoint ile denenecek: {api_url}")
                        download_url = api_url
                        continue
                raise requests.exceptions.HTTPError(f"403 Forbidden - API key gerekli olabilir")
            elif response.status_code == 404:
                print(f"❌ 404 Not Found - Dataset bulunamadı")
                raise requests.exceptions.HTTPError(f"Dataset bulunamadı: {url}")
            elif response.status_code == 429:
                print(f"❌ 429 Too Many Requests - Rate limit aşıldı")
                print(f"⏳ 60 saniye bekleyip tekrar deneyin")
                raise requests.exceptions.HTTPError(f"Rate limit aşıldı")
            
            response.raise_for_status()
            
            # Check if response is actually a ZIP file
            content_type = response.headers.get('content-type', '')
            if 'zip' not in content_type and 'octet-stream' not in content_type:
                print(f"⚠️ Beklenmeyen content-type: {content_type}")
                # HTML dönerse içerisinden .zip linki çekmeyi dene
                try:
                    text_snippet = response.text[:2000]
                    import re
                    zip_links = re.findall(r'href=["\']([^"\']+\.zip)["\']', text_snippet, flags=re.IGNORECASE)
                    if zip_links:
                        candidate = zip_links[0]
                        # Mutlak URL değilse ana host ile birleştir
                        from urllib.parse import urljoin
                        candidate_url = urljoin(download_url, candidate)
                        print(f"🔁 HTML içinden zip bulundu, yeniden denenecek: {candidate_url}")
                        download_url = candidate_url
                        # Bir sonraki denemeye geç
                        continue
                    else:
                        # Orijinal URL'yi dene ve redirect son URL'yi kullan
                        print("🔁 HTML döndü, orijinal URL üzerinden redirect takip edilecek")
                        head = session.get(url, timeout=60, allow_redirects=True)
                        final_url = head.url
                        if final_url.endswith('.zip'):
                            print(f"🔁 Redirect ile zip bulundu: {final_url}")
                            download_url = final_url
                            continue
                except Exception as _:
                    pass
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end="")
            
            print(f'\nDownload completed: {zip_path}')
            
            # Validate ZIP file
            if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
                print("Downloaded file is too small or missing")
                continue
            
            # Test ZIP file integrity
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    if len(file_list) == 0:
                        print("ZIP file is empty")
                        continue
                    print(f"ZIP contains {len(file_list)} files")
            except zipfile.BadZipFile:
                print("Invalid ZIP file, retrying...")
                continue

            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print(f'Archive extracted: {dataset_dir}')

            # Remove ZIP file
            os.remove(zip_path)

            # If extracted content is a single top-level directory, flatten it
            try:
                entries = [e for e in os.listdir(dataset_dir) if e not in ['.DS_Store']]
                if len(entries) == 1:
                    only_entry = os.path.join(dataset_dir, entries[0])
                    if os.path.isdir(only_entry):
                        print(f"🧹 Tek üst klasör tespit edildi: {entries[0]} -> düzleştiriliyor...")
                        for item in os.listdir(only_entry):
                            src = os.path.join(only_entry, item)
                            dst = os.path.join(dataset_dir, item)
                            if os.path.exists(dst):
                                # Merge: if directory, move contents; if file, overwrite
                                if os.path.isdir(src) and os.path.isdir(dst):
                                    for sub in os.listdir(src):
                                        shutil.move(os.path.join(src, sub), os.path.join(dst, sub))
                                else:
                                    try:
                                        os.remove(dst)
                                    except Exception:
                                        pass
                                    shutil.move(src, dst)
                            else:
                                shutil.move(src, dst)
                        # Remove the now-empty top-level directory
                        try:
                            os.rmdir(only_entry)
                        except Exception:
                            pass
            except Exception as _:
                pass

            # Fix directory structure
            fix_directory_structure(dataset_dir)

            # Update and save dataset YAML
            update_dataset_yaml(dataset_dir)

            return True
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Hatası (Deneme {attempt + 1}/3): {e}")
            if "403" in str(e):
                print(f"🔑 API anahtarı sorunu tespit edildi")
                if attempt == 2:
                    print(f"💡 Son çözüm önerisi: Roboflow hesabınızdan yeni API key alın")
                    return False
            elif "404" in str(e):
                print(f"📂 Dataset bulunamadı - URL'yi kontrol edin")
                return False
            elif "429" in str(e):
                print(f"⏳ Rate limit - 60 saniye bekleniyor...")
                import time
                time.sleep(60)
                continue
        except requests.exceptions.RequestException as e:
            print(f'❌ Requests hatası (Deneme {attempt + 1}/3): {e}')
        except zipfile.BadZipFile:
            print(f'❌ Bozuk ZIP dosyası (Deneme {attempt + 1}/3)')
        except Exception as e:
            print(f"❌ Genel hata (Deneme {attempt + 1}/3): {e}")
            
        if attempt == 2:  # Last attempt
            print("❌ Tüm indirme denemeleri başarısız")
            return False
        print("⏳ 5 saniye bekleyip tekrar deneniyor...")
        import time
        time.sleep(5)

    print(f'❌ All download attempts failed for {url}')
    return False

def fix_directory_structure(dataset_dir):
    """Fix directory structure to match YOLO11 expectations"""
    print(f"Checking and fixing directory structure...")

    # Roboflow structure: dataset/train/images and dataset/valid/images
    # YOLO11 expected structure: dataset/images/train and dataset/images/val

    # 1. First check existing structure and report
    train_images_dir = os.path.join(dataset_dir, 'train', 'images')
    valid_images_dir = os.path.join(dataset_dir, 'valid', 'images')
    test_images_dir = os.path.join(dataset_dir, 'test', 'images')

    train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
    valid_labels_dir = os.path.join(dataset_dir, 'valid', 'labels')
    test_labels_dir = os.path.join(dataset_dir, 'test', 'labels')

    # Expected YOLO11 structure
    yolo_images_dir = os.path.join(dataset_dir, 'images')
    yolo_labels_dir = os.path.join(dataset_dir, 'labels')

    yolo_train_images = os.path.join(yolo_images_dir, 'train')
    yolo_val_images = os.path.join(yolo_images_dir, 'val')
    yolo_test_images = os.path.join(yolo_images_dir, 'test')

    yolo_train_labels = os.path.join(yolo_labels_dir, 'train')
    yolo_val_labels = os.path.join(yolo_labels_dir, 'val')
    yolo_test_labels = os.path.join(yolo_labels_dir, 'test')

    # Check directory existence
    has_train = os.path.exists(train_images_dir) and os.path.exists(train_labels_dir)
    has_valid = os.path.exists(valid_images_dir) and os.path.exists(valid_labels_dir)
    has_test = os.path.exists(test_images_dir) and os.path.exists(test_labels_dir)

    print(f"Current structure:")
    print(f"  Train folder exists: {has_train}")
    print(f"  Valid folder exists: {has_valid}")
    print(f"  Test folder exists: {has_test}")

    # Create YOLO11 structure directories
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # 2. Copy train folder
    if has_train:
        print(f"Organizing train data...")
        # If folder doesn't exist or source has files, copy them
        if not os.path.exists(yolo_train_images) or len(os.listdir(train_images_dir)) > 0:
            # Copy images
            os.makedirs(yolo_train_images, exist_ok=True)
            for img_file in os.listdir(train_images_dir):
                src_path = os.path.join(train_images_dir, img_file)
                dst_path = os.path.join(yolo_train_images, img_file)
                shutil.copy2(src_path, dst_path)

            # Copy labels
            os.makedirs(yolo_train_labels, exist_ok=True)
            for label_file in os.listdir(train_labels_dir):
                src_path = os.path.join(train_labels_dir, label_file)
                dst_path = os.path.join(yolo_train_labels, label_file)
                shutil.copy2(src_path, dst_path)

            print(f"  Train data copied: {len(os.listdir(yolo_train_images))} images, {len(os.listdir(yolo_train_labels))} labels")

    # 3. Copy valid folder (as val)
    if has_valid:
        print(f"Organizing validation data...")
        # If folder doesn't exist or source has files, copy them
        if not os.path.exists(yolo_val_images) or len(os.listdir(valid_images_dir)) > 0:
            # Copy images
            os.makedirs(yolo_val_images, exist_ok=True)
            for img_file in os.listdir(valid_images_dir):
                src_path = os.path.join(valid_images_dir, img_file)
                dst_path = os.path.join(yolo_val_images, img_file)
                shutil.copy2(src_path, dst_path)

            # Copy labels
            os.makedirs(yolo_val_labels, exist_ok=True)
            for label_file in os.listdir(valid_labels_dir):
                src_path = os.path.join(valid_labels_dir, label_file)
                dst_path = os.path.join(yolo_val_labels, label_file)
                shutil.copy2(src_path, dst_path)

            print(f"  Validation data copied as 'val': {len(os.listdir(yolo_val_images))} images, {len(os.listdir(yolo_val_labels))} labels")

    # 4. Copy test folder (if exists)
    if has_test:
        print(f"Organizing test data...")
        # If folder doesn't exist or source has files, copy them
        if not os.path.exists(yolo_test_images) or len(os.listdir(test_images_dir)) > 0:
            # Copy images
            os.makedirs(yolo_test_images, exist_ok=True)
            for img_file in os.listdir(test_images_dir):
                src_path = os.path.join(test_images_dir, img_file)
                dst_path = os.path.join(yolo_test_images, img_file)
                shutil.copy2(src_path, dst_path)

            # Copy labels
            os.makedirs(yolo_test_labels, exist_ok=True)
            for label_file in os.listdir(test_labels_dir):
                src_path = os.path.join(test_labels_dir, label_file)
                dst_path = os.path.join(yolo_test_labels, label_file)
                shutil.copy2(src_path, dst_path)

            print(f"  Test data copied: {len(os.listdir(yolo_test_images))} images, {len(os.listdir(yolo_test_labels))} labels")

    # 5. Report updated structure
    print(f"\nUpdated structure:")
    for root, dirs, files in os.walk(dataset_dir):
        # Only show images and labels folders
        if 'images' in root or 'labels' in root:
            level = root.replace(dataset_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level <= 2:  # Only list first 3 levels
                sub_dirs = [d for d in dirs if d in ['train', 'val', 'test']]
                for d in sub_dirs:
                    sub_path = os.path.join(root, d)
                    file_count = len([f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f))])
                    print(f"{indent}    {d}/ ({file_count} files)")

def update_dataset_yaml(dataset_dir):
    """Read data.yaml from downloaded dataset and reconfigure it"""
    source_yaml = os.path.join(dataset_dir, 'data.yaml')
    target_yaml = 'dataset.yaml'

    try:
        # Read original YAML
        with open(source_yaml, 'r') as f:
            data = yaml.safe_load(f)

        # Preserve class information
        class_names = data.get('names', [])
        nc = data.get('nc', len(class_names))

        # Create new configuration
        updated_data = {
            'path': os.path.abspath(dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test' if os.path.exists(os.path.join(dataset_dir, 'images/test')) else '',
            'nc': nc,
            'names': class_names
        }

        # Preserve other metadata if exists
        if 'roboflow' in data:
            updated_data['roboflow'] = data['roboflow']

        # Save new YAML
        with open(target_yaml, 'w') as f:
            yaml.dump(updated_data, f, sort_keys=False)

        print(f'Dataset configuration updated: {target_yaml}')
        print(f'Classes: {updated_data["names"]}')

        # For safety, show the updated dataset configuration
        print(f'Updated configuration:')
        for key, value in updated_data.items():
            print(f"  {key}: {value}")

        return True
    except Exception as e:
        print(f'YAML update error: {e}')
        return False

def analyze_dataset(dataset_dir):
    """Analyze dataset and provide statistics"""
    print(f"\n===== Dataset Analysis =====")
    
    try:
        # Get image and label counts
        train_img_dir = os.path.join(dataset_dir, 'images', 'train')
        val_img_dir = os.path.join(dataset_dir, 'images', 'val')
        test_img_dir = os.path.join(dataset_dir, 'images', 'test')
        
        train_label_dir = os.path.join(dataset_dir, 'labels', 'train')
        val_label_dir = os.path.join(dataset_dir, 'labels', 'val')
        test_label_dir = os.path.join(dataset_dir, 'labels', 'test')
        
        # Count files if directories exist
        train_img_count = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
        val_img_count = len(os.listdir(val_img_dir)) if os.path.exists(val_img_dir) else 0
        test_img_count = len(os.listdir(test_img_dir)) if os.path.exists(test_img_dir) else 0
        
        train_label_count = len(os.listdir(train_label_dir)) if os.path.exists(train_label_dir) else 0
        val_label_count = len(os.listdir(val_label_dir)) if os.path.exists(val_label_dir) else 0
        test_label_count = len(os.listdir(test_label_dir)) if os.path.exists(test_label_dir) else 0
        
        # Print statistics
        print(f"Training set: {train_img_count} images, {train_label_count} labels")
        print(f"Validation set: {val_img_count} images, {val_label_count} labels")
        print(f"Test set: {test_img_count} images, {test_label_count} labels")
        print(f"Total images: {train_img_count + val_img_count + test_img_count}")
        
        # Check for label class distribution
        if os.path.exists('dataset.yaml'):
            with open('dataset.yaml', 'r') as f:
                data = yaml.safe_load(f)
            
            class_names = data.get('names', [])
            print(f"\nClass names: {class_names}")
            
            # Count instances of each class in the training set
            if os.path.exists(train_label_dir) and class_names:
                class_counts = {name: 0 for name in class_names}
                
                # Sample up to 50 label files to get class distribution
                sample_count = min(50, train_label_count)
                sample_files = os.listdir(train_label_dir)[:sample_count]
                
                for label_file in sample_files:
                    file_path = os.path.join(train_label_dir, label_file)
                    with open(file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:  # YOLO format: class x y w h
                                try:
                                    class_idx = int(parts[0])
                                    if 0 <= class_idx < len(class_names):
                                        class_counts[class_names[class_idx]] += 1
                                except (ValueError, IndexError):
                                    pass
                
                print("\nClass distribution (based on sample):")
                for name, count in class_counts.items():
                    print(f"  {name}: {count}")
                    
        return {
            'train_count': train_img_count,
            'val_count': val_img_count,
            'test_count': test_img_count,
            'total_count': train_img_count + val_img_count + test_img_count
        }
    except Exception as e:
        print(f"Dataset analysis error: {e}")
        return None

def check_dataset_integrity(dataset_dir):
    """Check dataset integrity - ensure all images have corresponding labels"""
    print("\n===== Dataset Integrity Check =====")
    
    issues_found = 0
    
    try:
        # Check train set
        train_img_dir = os.path.join(dataset_dir, 'images', 'train')
        train_label_dir = os.path.join(dataset_dir, 'labels', 'train')
        
        if os.path.exists(train_img_dir) and os.path.exists(train_label_dir):
            train_images = {os.path.splitext(f)[0] for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}
            train_labels = {os.path.splitext(f)[0] for f in os.listdir(train_label_dir) if f.endswith('.txt')}
            
            # Find images without labels
            img_without_label = train_images - train_labels
            if img_without_label:
                print(f"Found {len(img_without_label)} training images without labels")
                if len(img_without_label) <= 5:  # Show the first few only
                    print(f"  Missing labels for: {', '.join(list(img_without_label))}")
                issues_found += len(img_without_label)
            
            # Find labels without images
            label_without_img = train_labels - train_images
            if label_without_img:
                print(f"Found {len(label_without_img)} training labels without images")
                if len(label_without_img) <= 5:  # Show the first few only
                    print(f"  Extra labels for: {', '.join(list(label_without_img))}")
                issues_found += len(label_without_img)
        
        # Check validation set
        val_img_dir = os.path.join(dataset_dir, 'images', 'val')
        val_label_dir = os.path.join(dataset_dir, 'labels', 'val')
        
        if os.path.exists(val_img_dir) and os.path.exists(val_label_dir):
            val_images = {os.path.splitext(f)[0] for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}
            val_labels = {os.path.splitext(f)[0] for f in os.listdir(val_label_dir) if f.endswith('.txt')}
            
            # Find images without labels
            img_without_label = val_images - val_labels
            if img_without_label:
                print(f"Found {len(img_without_label)} validation images without labels")
                issues_found += len(img_without_label)
            
            # Find labels without images
            label_without_img = val_labels - val_images
            if label_without_img:
                print(f"Found {len(label_without_img)} validation labels without images")
                issues_found += len(label_without_img)
        
        if issues_found == 0:
            print("No dataset integrity issues found!")
        else:
            print(f"Total issues found: {issues_found}")
            print("Consider fixing these issues for better training results.")
        
        return issues_found
    except Exception as e:
        print(f"Dataset integrity check error: {e}")
        return -1

if __name__ == "__main__":
    # If run directly, test functionality
    test_dir = "datasets/test_dataset"
    os.makedirs(test_dir, exist_ok=True)
    print("This module provides dataset management functions.")
    print("To test, provide a Roboflow URL as argument.")
