"""Read-only compatibility smoke test for the documented H2GCN Docker image.

Does not access or evaluate training/test labels, and does not write files.
"""
import importlib
import platform
import sys
from pathlib import Path

EXPECTED = {
    "tensorflow": "2.2.0",
    "numpy": "1.18.5",
    "scipy": "1.4.1",
    "networkx": "2.4",
    "sklearn": "0.23.2",
}


def main():
    print("Python:", sys.version.split()[0])
    print("OS:", platform.system(), "architecture:", platform.machine())
    if sys.version_info[:2] != (3, 8):
        raise SystemExit("FAIL: H2GCN image must use Python 3.8")
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise SystemExit("FAIL: expected Linux x86-64/amd64 environment")

    for module_name, required_version in EXPECTED.items():
        mod = importlib.import_module(module_name)
        actual = mod.__version__
        print("{}: {}".format(module_name, actual))
        if actual != required_version:
            raise SystemExit("FAIL: {} expected {}, got {}".format(
                module_name, required_version, actual))

    project = Path.cwd()
    if not (project / "h2gcn" / "datasets" / "npz.py").is_file():
        raise SystemExit("FAIL: run with the repository mounted at /workspace")

    sys.path.insert(0, str(project / "h2gcn"))
    from datasets.npz import NPZData
    print("NPZData loader: {}".format(NPZData.__name__))
    print("PASS: required packages and local NPZData import verified")
    print("NOTE: model training and clean-room metric replication not tested")


if __name__ == "__main__":
    main()
