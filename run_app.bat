@echo off
REM Double-click to start the MD Notes Enricher UI using the "env" virtual environment.
cd /d "%~dp0"
if not exist "env\Scripts\python.exe" (
  echo Virtual environment "env" not found. Create it with:
  echo   python -m venv env
  echo   env\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
"env\Scripts\python.exe" -m streamlit run app.py
pause
