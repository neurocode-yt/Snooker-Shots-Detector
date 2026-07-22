import sys
from pathlib import Path
import uvicorn

# Ensure project root is in Python path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from apps.api.main import app

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Starting Snooker AI Server...")
    print("  Access Web UI:  http://127.0.0.1:8000/")
    print("  API Docs:       http://127.0.0.1:8000/docs")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
