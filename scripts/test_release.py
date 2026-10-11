"""Exercise the released EXEs after relocation, with no global Python/.NET paths."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'vendor')]
from fixtures import checkpoint


def main():
    archive = ROOT / 'dist/NInferEZ-Converter-1.0.0-Portable-win-x64.zip'
    with tempfile.TemporaryDirectory(prefix='relocate-', dir=ROOT / 'build') as location:
        extracted = Path(location) / 'portable עברית'
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None, 'ZIP integrity failure'
            bundle.extractall(extracted)
        portable = extracted / 'NInferEZ Converter'
        root_files = sorted(p.name for p in portable.iterdir() if p.is_file())
        assert root_files == sorted(['NInferEZ.Converter.exe', 'NInferEZ.Converter.Backend.exe', 'README.md', 'LICENSE'])
        assert sorted(p.name for p in portable.iterdir() if p.is_dir()) == ['app', 'docs', 'licenses', 'runtime']
        hashes = (portable / 'docs/distribution/SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
        for line in hashes:
            expected, name = line.split('  ', 1)
            value = hashlib.sha256((portable / name).read_bytes()).hexdigest()
            assert value == expected, 'Checksum mismatch: ' + name
        env = os.environ.copy()
        env['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32')
        env['PYTHONHOME'] = env['PYTHONPATH'] = env['DOTNET_ROOT'] = str(Path(location) / 'absent')
        env['DOTNET_BUNDLE_EXTRACT_BASE_DIR'] = str(Path(location) / 'dotnet-cache')
        backend = portable / 'NInferEZ.Converter.Backend.exe'
        cap = subprocess.run([str(backend), 'capabilities'], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding='utf-8', timeout=60)
        assert cap.returncode == 0, cap.stdout + cap.stderr
        capabilities = json.loads(cap.stdout)
        assert capabilities['devices'] == ['cpu']
        source, _, _ = checkpoint(Path(location) / 'source')
        output = Path(location) / 'backend עברית.ninfer'
        request = {'source': str(source), 'output': str(output), 'profile': 'auto', 'maxFileBytes': 32768}
        result = subprocess.run([str(backend), 'convert'], input=json.dumps(request), cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding='utf-8', timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert events[-1]['success'] and output.is_file()
        assert any(e['type'] == 'progress' and e['percent'] == 100 for e in events)
        screenshot = ROOT / 'build/relocated-frontend.png'
        gui_output = Path(location) / 'frontend עברית.ninfer'
        gui = subprocess.run([str(portable / 'NInferEZ.Converter.exe'), '--smoke-convert',
                              str(source), str(gui_output), str(screenshot)], cwd=ROOT, env=env,
                             capture_output=True, timeout=90)
        assert gui.returncode == 0, gui.stderr.decode('utf-8', 'replace')
        assert gui_output.is_file() and screenshot.is_file()
        summary = {'zipIntegrity': True, 'checkedHashes': len(hashes), 'relocatedUnicodeDirectory': True,
                   'differentWorkingDirectory': True, 'globalRuntimePathsExcluded': True,
                   'backendConversion': True, 'frontendConversion': True, 'cpuOnly': True,
                   'rootFileCount': len(root_files), 'singleFileDotnet': True}
        (ROOT / 'build/relocation-results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
