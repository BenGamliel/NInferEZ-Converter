"""Use Python's canonical stdlib ZIP for flat modules and encodings.

Package directories, native extensions and filesystem-based resources retain their
original layout. Keep os.py/site.py as prefix landmarks. Never move site-packages.
"""
import os
from pathlib import Path
import shutil
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'dist/Portable/runtime/python'


def main():
    runtime = RUNTIME.resolve(strict=True)
    library = runtime / 'Lib'
    archive = runtime / 'python312.zip'
    sources = [p for p in library.glob('*.py') if p.name not in ('os.py', 'site.py')]
    encodings = library / 'encodings'
    if encodings.exists():
        sources.extend(p for p in encodings.rglob('*.py') if '__pycache__' not in p.parts)
    if sources:
        contents = {}
        if archive.exists():
            with zipfile.ZipFile(archive) as previous:
                contents = {name: previous.read(name) for name in previous.namelist()}
        for path in sources:
            if not path.resolve().is_relative_to(runtime):
                raise RuntimeError('Unsafe runtime source path')
            contents[path.relative_to(library).as_posix()] = path.read_bytes()
        temporary = runtime / ('.stdlib-' + uuid.uuid4().hex + '.zip')
        try:
            # Startup encodings must not depend on importing zlib first.
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as bundle:
                for name, data in sorted(contents.items()):
                    bundle.writestr(name, data)
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
        for path in sources:
            path.unlink()
    # Stale bytecode from the build machine is redundant; the host runs with -B.
    caches = sorted(library.rglob('__pycache__'), key=lambda p: len(p.parts), reverse=True)
    for cache in caches:
        if not cache.resolve().is_relative_to(runtime):
            raise RuntimeError('Unsafe runtime cache path')
        shutil.rmtree(cache)
    if encodings.exists() and not any(encodings.iterdir()):
        encodings.rmdir()
    print('Flat stdlib modules and encodings bundled; package/resource directories retained.')


if __name__ == '__main__':
    main()
