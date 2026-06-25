#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ShellToolTests(unittest.TestCase):
    def run_shell(self, text: str, run_dir: Path) -> subprocess.CompletedProcess[str]:
        script = run_dir.parent / "test.sh"
        script.write_text(text, encoding="utf-8")
        env = os.environ.copy()
        env.update({"CONTROL_ROOT": str(PACKAGE_ROOT), "RUN_DIR": str(run_dir)})
        run_dir.mkdir(parents=True)
        return subprocess.run(["bash", str(script)], text=True, capture_output=True, env=env, timeout=90)

    def test_validation_pass_and_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            passed = self.run_shell(
                '''#!/usr/bin/env bash
set -euo pipefail
source "$CONTROL_ROOT/scripts/validation/lib.sh"
validation_init
validation_case deterministic required -- bash -lc 'echo ok'
validation_mark multimodal optional not_applicable 'text model'
validation_finish
''',
                base / "pass",
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            summary = json.loads((base / "pass/validation/summary.json").read_text())
            self.assertEqual(summary["status"], "passed")

            partial = self.run_shell(
                '''#!/usr/bin/env bash
set -euo pipefail
source "$CONTROL_ROOT/scripts/validation/lib.sh"
validation_init
validation_case deterministic required -- bash -lc 'true'
validation_case batch optional -- bash -lc 'false'
validation_finish
''',
                base / "partial",
            )
            self.assertEqual(partial.returncode, 65)
            summary = json.loads((base / "partial/validation/summary.json").read_text())
            self.assertEqual(summary["status"], "partial")

    def test_benchmark_aggregation_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            completed = self.run_shell(
                '''#!/usr/bin/env bash
set -euo pipefail
source "$CONTROL_ROOT/scripts/benchmark/lib.sh"
benchmark_init
benchmark_case serving required --warmup 0 --repeat 1 -- bash -lc 'printf "{\\"metrics\\":{\\"output_throughput\\":{\\"value\\":100,\\"unit\\":\\"tok/s\\",\\"direction\\":\\"higher\\"},\\"median_ttft_ms\\":20}}\\n" > "$BENCHMARK_SAMPLE_FILE"'
benchmark_expect serving output_throughput higher 120 median tok/s
benchmark_finish
''',
                base / "bench",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((base / "bench/benchmark/summary.json").read_text())
            self.assertEqual(summary["status"], "passed")
            self.assertFalse(summary["target_met"])
            self.assertEqual(summary["cases"][0]["valid_samples"], 1)

    def test_compare_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text('{"x": 1.0, "id": "a"}', encoding="utf-8")
            (root / "b.json").write_text('{"x": 1.0001, "id": "b"}', encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(PACKAGE_ROOT / "scripts/validation/compare_json.py"), "--reference", str(root / "a.json"), "--actual", str(root / "b.json"), "--abs-tol", "0.001", "--ignore", "id"],
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_compare_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference.json").write_text("[1.0, 2.0, 3.0]", encoding="utf-8")
            (root / "actual.json").write_text("[1.0, 2.0001, 3.0]", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts/validation/compare_arrays.py"),
                    "--reference", str(root / "reference.json"),
                    "--actual", str(root / "actual.json"),
                    "--abs-tol", "0.001",
                    "--rel-tol", "0.001",
                    "--output", str(root / "comparison.json"),
                ],
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((root / "comparison.json").read_text())
            self.assertTrue(result["passed"])

    def test_benchmark_comparison_requires_same_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {
                "workload_fingerprint": "same",
                "cases": [{"name": "serve", "metrics": {"throughput": {"median": 100, "direction": "higher"}}}],
            }
            (root / "baseline.json").write_text(json.dumps({**common, "execution_fingerprint": "a"}), encoding="utf-8")
            (root / "current.json").write_text(json.dumps({**common, "execution_fingerprint": "b"}), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts/benchmark/compare.py"),
                    "--baseline", str(root / "baseline.json"),
                    "--current", str(root / "current.json"),
                ],
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertFalse(result["comparable"])
            self.assertIn("runtime_definition_changed", result["incomparable_reasons"])


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / ".claude").mkdir()
        (self.root / "scripts/validation").mkdir(parents=True)
        (self.root / "scripts/benchmark").mkdir(parents=True)
        (self.root / "tests").mkdir()
        shutil.copy2(PACKAGE_ROOT / "scripts/model_runtime.py", self.root / "scripts/model_runtime.py")
        for source in (PACKAGE_ROOT / "scripts/validation").iterdir():
            if source.is_file():
                shutil.copy2(source, self.root / "scripts/validation" / source.name)
        for source in (PACKAGE_ROOT / "scripts/benchmark").iterdir():
            if source.is_file():
                shutil.copy2(source, self.root / "scripts/benchmark" / source.name)
        shutil.copy2(PACKAGE_ROOT / "tests/fake_server.py", self.root / "tests/fake_server.py")
        target_dir = self.root / "tasks/demo/targets/engine"
        target_dir.mkdir(parents=True)
        (self.root / "tasks/demo/model.yaml").write_text("name: Demo\nmodel_path: /models/demo\nrevision: test\n", encoding="utf-8")
        (target_dir / "target.yaml").write_text(f"engine: fake\nhardware: cpu\ntarget_repo: .\nruntime:\n  python: {sys.executable}\n  tensor_parallel_size: 1\n", encoding="utf-8")
        self.port = free_port()
        init = self.run_cli("init", "demo/engine")
        self.assertEqual(init.returncode, 0, init.stderr)
        runbook = target_dir / "runbook"
        (runbook / "env.sh").write_text(
            f'''#!/usr/bin/env bash
MODEL_HOST=127.0.0.1
MODEL_PORT={self.port}
MODEL_BASE_URL=http://${{MODEL_HOST}}:${{MODEL_PORT}}
MODEL_STARTUP_TIMEOUT=30
MODEL_CHECK_TIMEOUT=20
MODEL_CHECK_VALIDATE_TIMEOUT=20
MODEL_CHECK_BENCHMARK_TIMEOUT=30
MODEL_SHUTDOWN_TIMEOUT=3
MODEL_READY_INTERVAL=0.1
MODEL_READY_PROBE_TIMEOUT=2
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
curl -fsS --max-time 1 "$MODEL_BASE_URL/health" >/dev/null
''',
            encoding="utf-8",
        )
        (runbook / "checks/validate.sh").write_text(
            '''#!/usr/bin/env bash
set -euo pipefail
source "$CONTROL_ROOT/scripts/validation/lib.sh"
validation_init
validation_case api required -- curl -fsS --max-time 2 "$MODEL_BASE_URL/health"
validation_finish
''',
            encoding="utf-8",
        )
        (runbook / "checks/benchmark.sh").write_text(
            '''#!/usr/bin/env bash
set -euo pipefail
source "$CONTROL_ROOT/scripts/benchmark/lib.sh"
benchmark_init
benchmark_case serving required --warmup 0 --repeat 1 -- bash -lc 'printf "{\\"output_throughput\\":100,\\"median_ttft_ms\\":20}\\n" > "$BENCHMARK_SAMPLE_FILE"'
benchmark_expect serving output_throughput higher 120 median tok/s
benchmark_finish
''',
            encoding="utf-8",
        )
        for path in runbook.rglob("*.sh"):
            path.chmod(0o755)

    def tearDown(self) -> None:
        try:
            self.run_cli("stop", "demo/engine")
        finally:
            self.temp.cleanup()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(self.root / "scripts/model_runtime.py"), "--repo-root", str(self.root), *args]
        stdout_path = self.root / "last-cli.stdout"
        stderr_path = self.root / "last-cli.stderr"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, text=True, stdout=stdout, stderr=stderr, timeout=120)
        return subprocess.CompletedProcess(
            command,
            completed.returncode,
            stdout_path.read_text(encoding="utf-8"),
            stderr_path.read_text(encoding="utf-8"),
        )

    def test_validate_runtime(self) -> None:
        completed = self.run_cli("run", "demo/engine", "--check", "validate")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "passed")
        run_dir = Path(payload["run_dir"])
        self.assertEqual(json.loads((run_dir / "validation/summary.json").read_text())["status"], "passed")
        self.assertEqual(payload["cleanup"]["status"], "passed")

    def test_benchmark_runtime_target_unmet(self) -> None:
        completed = self.run_cli("run", "demo/engine", "--check", "benchmark")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "passed")
        summary = json.loads((Path(payload["run_dir"]) / "benchmark/summary.json").read_text())
        self.assertFalse(summary["target_met"])


if __name__ == "__main__":
    unittest.main()
