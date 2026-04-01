import os
import json
import sys
import subprocess
from pathlib import Path

def setup_directories():
    """Şartname gereksinimleri ve proje düzeni için gerekli dizinleri oluşturur."""
    dirs = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'output/detections',
        'output/odometry',
        'tests',
        'scripts'
    ]
    print("[1/4] Dizin yapısı oluşturuluyor...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  - {d} oluşturuldu.")

def download_models():
    """YOLOv8n modelini (veya diğerlerini) indirir."""
    print("\n[2/4] Model kontrolü ve indirme...")
    model_path = Path("yolov8n.pt")
    if not model_path.exists():
        print("  - yolov8n.pt bulunamadı. İndiriliyor...")
        try:
            from ultralytics import YOLO
            # YOLO nesnesi oluşturmak modeli otomatik indirir
            YOLO("yolov8n.pt")
            print("  - yolov8n.pt başarıyla indirildi.")
        except ImportError:
            print("  [HATA] 'ultralytics' kütüphanesi yüklü değil. Önce requirements.txt yüklenmeli.")
    else:
        print("  - yolov8n.pt zaten mevcut.")

def generate_default_config():
    """Yarışma istemcisi için varsayılan config.json oluşturur."""
    print("\n[3/4] Varsayılan yapılandırma oluşturuluyor...")
    config = {
        "server_url": "http://127.0.0.1:5000",
        "username": "team_skyguard",
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.35,
        "img_size": 1280,
        "save_logs": True
    }
    
    config_path = "config.json"
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print("  - config.json oluşturuldu.")
    else:
        print("  - config.json zaten mevcut, atlanıyor.")

def check_environment():
    """Python sürümünü ve temel paketleri kontrol eder."""
    print("\n[4/4] Ortam kontrolü...")
    print(f"  - Python sürümü: {sys.version.split()[0]}")
    
    try:
        import cv2
        import numpy
        import ultralytics
        print("  - Temel kütüphaneler (OpenCV, NumPy, Ultralytics) yüklü.")
    except ImportError as e:
        print(f"  [UYARI] Eksik paketler var: {e}")
        print("  Lütfen 'pip install -r requirements.txt' komutunu çalıştırın.")

if __name__ == "__main__":
    print("="*50)
    print("   SkyGuard AI - Kapsamlı Kurulum Sihirbazı")
    print("="*50)
    
    setup_directories()
    download_models()
    generate_default_config()
    check_environment()
    
    print("\n" + "="*50)
    print("   Kurulum İşlemi Tamamlandı!")
    print("="*50)
