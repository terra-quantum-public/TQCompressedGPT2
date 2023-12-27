from setuptools import setup, find_packages
from pathlib import Path

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "requirements.txt", "r") as file:
    required_packages = []
    for line in file.readlines():
        line = line.strip()

        if line.startswith("#") or len(line) == 0:
            continue

        if "https://" in line:
            tail = line.rsplit("/", 1)[1]
            tail = tail.split("#")[0]
            line = tail.replace("@", "==").replace(".git", "")

        if line == "-e .":
            continue

        required_packages.append(line)

setup(
    name="TQCompressedGPT2",
    version="0.1",
    packages=find_packages(),
    install_requires=[required_packages],
)
