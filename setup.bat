@echo off
echo ==========================================
echo      SkyGuard AI Ortam Kurulumu
echo ==========================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Python yuklu degil veya PATH'e eklenmemis.
    pause
    exit /b
)

echo [BILGI] Sanal ortam olusturuluyor...
python -m venv venv

echo [BILGI] Sanal ortam aktif ediliyor...
call venv\Scripts\activate

echo [BILGI] Bagimliliklar yukleniyor...
pip install -r requirements.txt

echo [BILGI] Kapsamli kurulum sihirbazi calistiriliyor...
python setup_project.py

echo.
echo ==========================================
echo        Kurulum Tamamlandi!
echo ==========================================
echo.
echo Kontrol panelini calistirmak icin:
echo    streamlit run dashboard.py
echo.
echo Ana gorev dongusunu calistirmak icin:
echo    python competition_entry.py
echo.
pause
