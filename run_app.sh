#!/usr/bin/env bash
# Start the MD Notes Enricher UI using the "env" virtual environment.
cd "$(dirname "$0")" || exit 1
if [ -x "env/Scripts/python.exe" ]; then PY="env/Scripts/python.exe"   # Git Bash on Windows
elif [ -x "env/bin/python" ]; then PY="env/bin/python"                 # macOS / Linux
else
  echo 'Virtual environment "env" not found. Create it with:'
  echo '  python -m venv env && env/bin/python -m pip install -r requirements.txt'
  read -r -p "Press Enter to close..." _; exit 1
fi
exec "$PY" -m streamlit run app.py
