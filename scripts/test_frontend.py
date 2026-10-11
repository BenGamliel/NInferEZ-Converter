"""Exercise the real Wizard and Manual handlers against the standalone backend."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "vendor")]
from fixtures import checkpoint
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts


def main():
    exe = ROOT / "dist/Portable/NInferEZ.Converter.exe"
    with tempfile.TemporaryDirectory(prefix="ui-tests-", dir=ROOT / "build") as temporary:
        location = Path(temporary)
        env = os.environ.copy()
        env["DOTNET_BUNDLE_EXTRACT_BASE_DIR"] = str(location / "dotnet-cache")
        def run(*args):
            result = subprocess.run([str(exe), *map(str, args)], cwd=location, env=env, capture_output=True, timeout=120)
            error = Path(str(args[-1]) + ".error.txt")
            assert result.returncode == 0 and not error.exists(), error.read_text() if error.exists() else result.stderr.decode("utf-8", "replace")
            assert Path(args[-1]).is_file(), "Missing GUI snapshot"
        src, _, _ = checkpoint(location / "מודל ראשון")
        wizard_out = location / "wizard עברית.ninfer"
        run("--smoke-convert", src, wizard_out, ROOT / "build/wizard-conversion.png")
        with Artifact(wizard_out) as a:
            assert any(obj.format == "q4_g64_fp16" for obj in a.objects if hasattr(obj, "format"))
        missing, _, _ = checkpoint(location / "missing")
        resources = location / "resource files"; resources.mkdir()
        for role in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
            shutil.move(missing / role, resources / role)
        recovered_out = location / "recovered.ninfer"
        run("--smoke-recovery", missing, resources, recovered_out, ROOT / "build/wizard-recovery.png")
        assert not (missing / "config.json").exists(), "Wizard modified source resources"
        src2, _, _ = checkpoint(location / "manual", moe=True, vision=True, mtp=True)
        manual_out = location / "manual.ninfer"
        request = {"source": str(src2), "output": str(manual_out), "profile": "q6", "components": ["text", "vision", "mtp"],
                   "rowsPerChunk": 7, "maxFileBytes": 32768,
                   "tensorOverrides": [{"selector": "text/token_embedding", "format": "q8_g32_fp16", "method": "grouped_absmax"}],
                   "name": "Manual MoE + Vision + MTP"}
        request_file = location / "manual-request.json"; request_file.write_text(json.dumps(request), encoding="utf-8")
        run("--smoke-request", request_file, ROOT / "build/manual-conversion.png")
        with Artifact(manual_out) as a:
            assert set(a.directory.components) == {"text", "vision", "mtp"}
            assert len(a.directory.files) > 1
            assert a.by_id[binding_parts(a.directory.bindings["text/token_embedding"], a.by_id)[0][0]].format == "q8_g32_fp16"
        repacked = location / "wizard repacked.ninfer"
        run("--smoke-convert", manual_out, repacked, ROOT / "build/wizard-repack.png")
        manual_repacked = location / "manual repacked.ninfer"
        repack_request = location / "manual-repack.json"
        repack_request.write_text(json.dumps({"source": str(manual_out), "output": str(manual_repacked), "profile": "preserve", "maxFileBytes": 65536,
                                            "name": "Checkpoint-only options must not alter repack metadata", "resourceFolder": "unused"}), encoding="utf-8")
        run("--smoke-request", repack_request, ROOT / "build/manual-repack.png")
        with Artifact(manual_out) as original:
            for path in (repacked, manual_repacked):
                with Artifact(path) as copy:
                    assert original.directory.metadata == copy.directory.metadata
                    assert original.directory.components == copy.directory.components
                    for obj in original.objects:
                        assert original.read_object(obj.id) == copy.read_object(obj.id)
        summary = {"wizardExplicitChoice": True, "wizardConversion": True, "missingFilesRecovery": True,
                   "sourceResourcesUnmodified": True, "manualRequestImport": True, "manualMoeVisionMtp": True,
                   "manualTensorOverrides": True, "outputSharding": True, "unicodePaths": True,
                   "wizardArtifactRepack": True, "manualArtifactRepack": True, "repackRetainsAllObjects": True}
        (ROOT / "build/frontend-tests.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
