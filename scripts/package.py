"""Audit and archive the complete self-contained Portable; never includes test weights."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "dist" / "Portable"
VERSION = "1.0.0"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def main():
    required = ["NInferEZ.Converter.exe", "NInferEZ.Converter.Backend.exe", "runtime/python/python.exe",
                "app/backend/app.py", "app/vendor/tools/convert/__main__.py", "app/inspector/ninfer-inspect.exe",
                "README.md", "LICENSE", "licenses/microsoft.netcore.app.runtime.win-x64-LICENSE.TXT",
                "docs/distribution/dotnet-runtime.json"]
    required.append("runtime/python/python312.zip")
    for item in required:
        if not (PORTABLE / item).is_file():
            raise RuntimeError("Missing package dependency: " + item)
    files = sorted(path for path in PORTABLE.rglob("*") if path.is_file())
    unexpected = [path for path in files if path.suffix.lower() in (".ninfer", ".gguf", ".safetensors", ".pyc", ".pdb", ".dmp") or path.name in ("pyvenv.cfg",)]
    if unexpected:
        raise RuntimeError("Public package contains weights or a nonportable environment: " + str(unexpected))
    root_files = sorted(p.name for p in PORTABLE.iterdir() if p.is_file())
    root_dirs = sorted(p.name for p in PORTABLE.iterdir() if p.is_dir())
    if root_files != sorted(["NInferEZ.Converter.exe", "NInferEZ.Converter.Backend.exe", "README.md", "LICENSE"]):
        raise RuntimeError("Release root must contain only the two EXEs, README and LICENSE: " + str(root_files))
    if root_dirs != ["app", "docs", "licenses", "runtime"]:
        raise RuntimeError("Unexpected release root directories: " + str(root_dirs))
    folder_counts = [(len(list(folder.glob('*'))), folder) for folder in PORTABLE.rglob('*') if folder.is_dir()]
    largest = max(folder_counts, key=lambda item: item[0])
    records = PORTABLE / "docs" / "distribution"
    site = PORTABLE / "runtime" / "python" / "Lib" / "site-packages"
    dependencies = []
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        dependencies.append({"name": distribution.metadata["Name"], "version": distribution.version})
    dependencies.sort(key=lambda item: item["name"].lower())
    framework = json.loads((records / "dotnet-runtime.json").read_text())["runtimeOptions"]["includedFrameworks"]
    identity = {"product": "NInferEZ Converter", "version": VERSION, "platform": "windows-x64", "portable": True,
                "conversionRuntime": "cpu", "python": "3.12.10", "dotnet": framework, "dependencies": dependencies,
                "engineRevision": "51cdf17108d28d7b772dfcc56a80dcf5de517291", "codeSigned": False,
                "modelWeightsIncluded": False, "runtimeQualification": "See docs/QA.md",
                "layoutVersion": 2, "dotnetPackaging": "compressed single-file EXEs",
                "rootFiles": root_files, "rootDirectories": root_dirs}
    (records / "package-manifest.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    checksum_file = records / "SHA256SUMS.txt"
    files = sorted(p for p in PORTABLE.rglob("*") if p.is_file() and p != checksum_file)
    checksum_file.write_text("".join(f"{digest(p)}  {p.relative_to(PORTABLE).as_posix()}\n" for p in files), encoding="utf-8")
    archive = ROOT / "dist" / f"NInferEZ-Converter-{VERSION}-Portable-win-x64.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
        for path in sorted(p for p in PORTABLE.rglob("*") if p.is_file()):
            bundle.write(path, (Path("NInferEZ Converter") / path.relative_to(PORTABLE)).as_posix())
    checksum = digest(archive)
    Path(str(archive) + ".sha256").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    summary = {"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum,
               "portableFiles": len(files) + 1, "portableBytes": sum(p.stat().st_size for p in files),
               "rootFiles": root_files, "rootDirectories": root_dirs,
               "largestDirectory": {"path": largest[1].relative_to(PORTABLE).as_posix(), "entries": largest[0]}}
    (ROOT / "build" / "package-audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
