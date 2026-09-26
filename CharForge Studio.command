#!/bin/bash
# Double-click to start CharForge Studio (http://localhost:8830). Close this window to stop it.
cd "$(dirname "$0")"
for py in ".venv/bin/python" "../.venv/bin/python" "$(command -v python3)"; do
  if [ -x "$py" ] && "$py" -c "import numpy" 2>/dev/null; then exec "$py" charforge.py studio; fi
done
echo "No Python with CharForge's requirements found. Set it up once:"
echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
read -n 1 -s -r -p "Press any key to close."
