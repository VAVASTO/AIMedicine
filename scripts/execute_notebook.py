from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/demo.ipynb"
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=90, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(nb, path)
print("Executed notebook saved:", path)
