import sys
import os
import json
import time

def test_imports():
    print("[1/4] Kütüphane içe aktarma test ediliyor...")
    try:
        import cv2
        import numpy as np
        import requests
        import ultralytics
        from ultralytics import YOLO
        print("  - [OK] Tüm kütüphaneler yüklü.")
        return True
    except ImportError as e:
        print(f"  - [HATA] Eksik kütüphane: {e}")
        return False

def test_model_loading():
    print("\n[2/4] Model yükleme test ediliyor...")
    try:
        from ultralytics import YOLO
        model_path = "yolov8n.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"  - [OK] YOLOv8n modeli başarıyla yüklendi.")
            return True
        else:
            print(f"  - [UYARI] {model_path} bulunamadı.")
            return False
    except Exception as e:
        print(f"  - [HATA] Model hatası: {e}")
        return False

def test_config():
    print("\n[3/4] Yapılandırma dosyası kontrol ediliyor...")
    if os.path.exists("config.json"):
        try:
            with open("config.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"  - [OK] config.json okundu: Server: {config.get('server_url')}")
            return True
        except Exception as e:
            print(f"  - [HATA] config.json bozuk: {e}")
            return False
    else:
        print("  - [HATA] config.json bulunamadı.")
        return False

def test_server_connection():
    print("\n[4/4] Sunucu bağlantısı kontrol ediliyor...")
    if os.path.exists("config.json"):
        with open("config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        url = config.get("server_url")
        try:
            import requests
            resp = requests.get(f"{url}/api/ping", timeout=3)
            if resp.status_code == 200:
                print(f"  - [OK] Sunucuya ({url}) başarıyla bağlanıldı.")
                return True
            else:
                print(f"  - [UYARI] Sunucu yanıtı: {resp.status_code}")
                return False
        except Exception:
            print(f"  - [BİLGİ] Sunucuya ({url}) bağlanılamadı. Mock server kapalı olabilir.")
            return False
    return False

if __name__ == "__main__":
    print("="*50)
    print("   SkyGuard AI - Doğrulama Betiği")
    print("="*50)
    
    results = [
        test_imports(),
        test_model_loading(),
        test_config(),
        test_server_connection()
    ]
    
    print("\n" + "="*50)
    if all(results[:3]): # Server connection is optional for setup success
        print("   DOĞRULAMA BAŞARILI!")
        print("   Sistem yarışma için hazır.")
    else:
        print("   DOĞRULAMA BAŞARISIZ!")
        print("   Lütfen yukarıdaki hataları kontrol edin.")
    print("="*50)
