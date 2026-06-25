from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/model_runtime.py"
EVALUATE = ROOT / "scripts/evaluate.py"
FAKE = ROOT / "tests/fake_server.py"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(argv: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, check=False, timeout=kwargs.pop("timeout", 90), **kwargs)


class RuntimeFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / ".claude").mkdir()
        (self.root / "tasks/demo/targets/vllm-kunlun").mkdir(parents=True)
        (self.root / "runs").mkdir()
        shutil.copytree(ROOT / "templates", self.root / "templates")
        (self.root / "scripts").mkdir()
        shutil.copy2(RUNTIME, self.root / "scripts/model_runtime.py")
        (self.root / "tests").mkdir()
        shutil.copy2(FAKE, self.root / "tests/fake_server.py")
        (self.root / "tasks/demo/model.yaml").write_text("name: Demo\npath: /models/demo\n", encoding="utf-8")
        (self.root / "tasks/demo/targets/vllm-kunlun/target.yaml").write_text(
            f"engine: vllm-kunlun\nhardware: p800\ntarget_repo: {self.root}\nruntime:\n  python: {sys.executable}\n  tensor_parallel_size: 1\n",
            encoding="utf-8",
        )
        self.port = free_port()
        self.cli("init", "demo/vllm-kunlun")
        runbook = self.root / "tasks/demo/targets/vllm-kunlun/runbook"
        (runbook / "env.sh").write_text(
            f'''#!/usr/bin/env bash
MODEL_HOST=127.0.0.1
MODEL_PORT={self.port}
MODEL_BASE_URL=http://${{MODEL_HOST}}:${{MODEL_PORT}}
MODEL_STARTUP_TIMEOUT=15
MODEL_CHECK_TIMEOUT=10
MODEL_VALIDATE_TIMEOUT=10
MODEL_BENCHMARK_TIMEOUT=10
MODEL_SHUTDOWN_TIMEOUT=3
MODEL_READY_INTERVAL=0.1
MODEL_READY_PROBE_TIMEOUT=5
''',
            encoding="utf-8",
        )
        (runbook / "start.sh").write_text(
            '''#!/usr/bin/env bash
set -euo pipefail
exec "$RUNTIME_PYTHON" "$CONTROL_ROOT/tests/fake_server.py" --host "$MODEL_HOST" --port "$MODEL_PORT"
''',
            encoding="utf-8",
        )
        (runbook / "ready.sh").write_text(
            '''#!/usr/bin/env bash
set -euo pipefail
"$RUNTIME_PYTHON" - <<'PY2'
import os,urllib.request
urllib.request.urlopen(os.environ["MODEL_BASE_URL"]+"/health",timeout=2).read()
PY2
''',
            encoding="utf-8",
        )
        (runbook / "checks/smoke.sh").write_text(
            '''#!/usr/bin/env bash
set -euo pipefail
"$RUNTIME_PYTHON" - <<'PY2'
import os,urllib.request
req=urllib.request.Request(os.environ["MODEL_BASE_URL"]+"/v1/chat/completions",data=b"{}",headers={"Content-Type":"application/json"})
open(os.path.join(os.environ["RUN_DIR"],"smoke.json"),"wb").write(urllib.request.urlopen(req,timeout=2).read())
PY2
''',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.cli("stop", "demo/vllm-kunlun", check=False)
        self.temp.cleanup()

    def cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = run([sys.executable, str(self.root / "scripts/model_runtime.py"), "--repo-root", str(self.root), *args])
        if check and proc.returncode:
            raise AssertionError(proc.stdout + proc.stderr)
        return proc


class RuntimeTests(RuntimeFixture):
    def test_engine_overlay_and_sync(self) -> None:
        start = self.root / "tasks/demo/targets/vllm-kunlun/runbook/start.sh"
        start.write_text("custom-start\n", encoding="utf-8")
        result = json.loads(self.cli("init", "demo/vllm-kunlun").stdout)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(start.read_text(encoding="utf-8"), "custom-start\n")
        self.assertTrue(any(item["path"] == "start.sh" and item["action"] == "preserved" for item in result["actions"]))

    def test_smoke_cleanup(self) -> None:
        data = json.loads(self.cli("run", "demo/vllm-kunlun", "--check", "smoke").stdout)
        self.assertEqual(data["status"], "passed")
        self.assertEqual(data["cleanup"]["status"], "passed")
        self.assertTrue((Path(data["run_dir"]) / "smoke.json").exists())

    def test_persistent_lifecycle(self) -> None:
        data = json.loads(self.cli("serve", "demo/vllm-kunlun").stdout)
        self.assertEqual(data["status"], "passed")
        self.assertEqual(json.loads(self.cli("status", "demo/vllm-kunlun").stdout)["status"], "running")
        self.assertEqual(json.loads(self.cli("exec", "demo/vllm-kunlun", "--check", "smoke").stdout)["status"], "passed")
        self.assertEqual(json.loads(self.cli("stop", "demo/vllm-kunlun").stdout)["status"], "passed")


class StageTests(RuntimeFixture):
    def setUp(self) -> None:
        super().setUp()
        shutil.copy2(EVALUATE, self.root / "scripts/evaluate.py")
        target = self.root / "tasks/demo/targets/vllm-kunlun"
        (self.root / "tasks/demo/model-analysis.md").write_text("---\nstatus: passed\nrevision: 1\n---\n", encoding="utf-8")
        (target / "assessment.md").write_text("---\nstatus: passed\nresult: adaptation_required\nrevision: 1\n---\n", encoding="utf-8")
        (target / "implementation.md").write_text("---\nstatus: passed\n---\n", encoding="utf-8")
        runbook = target / "runbook"
        (runbook / "checks/validate.sh").write_text(
            '''#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$RUN_DIR/validation"
printf '{"status":"passed","cases":[{"id":"greedy","required":true,"status":"passed"}]}\n' > "$RUN_DIR/validation/result.json"
''', encoding="utf-8")
        (runbook / "checks/benchmark.sh").write_text(
            '''#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$RUN_DIR/benchmark"
printf '{"completed":10,"request_throughput":2.5,"output_throughput":100.0,"mean_ttft_ms":30.0,"mean_tpot_ms":8.0}\n' > "$RUN_DIR/benchmark/result.json"
''', encoding="utf-8")

    def stage(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = run([sys.executable, str(self.root / "scripts/evaluate.py"), "--repo-root", str(self.root), *args])
        if check and proc.returncode:
            raise AssertionError(proc.stdout + proc.stderr)
        return proc

    def test_validate_and_benchmark(self) -> None:
        validation = json.loads(self.stage("validate", "demo/vllm-kunlun").stdout)
        self.assertEqual(validation["status_hint"], "passed")
        target = self.root / "tasks/demo/targets/vllm-kunlun"
        (target / "validation.md").write_text("---\nstatus: passed\n---\n", encoding="utf-8")
        benchmark = json.loads(self.stage("benchmark", "demo/vllm-kunlun").stdout)
        self.assertEqual(benchmark["status_hint"], "passed")
        self.assertEqual(benchmark["benchmark_result"]["metrics"]["request_throughput_rps"], 2.5)


class HelperTests(unittest.TestCase):
    def test_model_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"demo","architectures":["Demo"]}', encoding="utf-8")
            output = root / "facts.json"
            proc = run([sys.executable, str(ROOT / "scripts/model.py"), "inspect", "--model-path", str(model), "--output", str(output)])
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(json.loads(output.read_text())["json"]["config.json"]["model_type"], "demo")

    def test_implement_scope_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            run(["git", "init", "-q"], cwd=repo)
            run(["git", "config", "user.email", "test@example.com"], cwd=repo)
            run(["git", "config", "user.name", "Test"], cwd=repo)
            (repo / "allowed").mkdir()
            (repo / "allowed/x.py").write_text("X=1\n")
            (repo / "README.md").write_text("x\n")
            run(["git", "add", "."], cwd=repo)
            run(["git", "commit", "-qm", "base"], cwd=repo)
            head = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
            (repo / "allowed/x.py").write_text("X=2\n")
            out = root / "check"
            proc = run([
                sys.executable, str(ROOT / "scripts/implement.py"), "check",
                "--target-repo", str(repo), "--base-ref", head,
                "--run-dir", str(out), "--allow", "allowed/**", "--",
                sys.executable, "-c", "from pathlib import Path; assert Path('allowed/x.py').exists()",
            ])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            (repo / "README.md").write_text("changed\n")
            proc = run([
                sys.executable, str(ROOT / "scripts/implement.py"), "scope",
                "--target-repo", str(repo), "--base-ref", head,
                "--output", str(root / "scope.json"), "--allow", "allowed/**",
            ])
            self.assertEqual(proc.returncode, 4)


if __name__ == "__main__":
    unittest.main()
