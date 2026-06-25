#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CLI_SOURCE = PACKAGE_ROOT / "scripts" / "model_runtime.py"
FAKE_SERVER_SOURCE = PACKAGE_ROOT / "tests" / "fake_server.py"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / ".claude").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "tests").mkdir()
        shutil.copy2(CLI_SOURCE, self.root / "scripts" / "model_runtime.py")
        shutil.copy2(FAKE_SERVER_SOURCE, self.root / "tests" / "fake_server.py")
        target_dir = self.root / "tasks" / "demo" / "targets" / "engine"
        target_dir.mkdir(parents=True)
        (self.root / "tasks" / "demo" / "model.yaml").write_text(
            "name: Demo\nmodel_path: /models/demo\nrevision: test\n", encoding="utf-8"
        )
        (target_dir / "target.yaml").write_text(
            "engine: fake\nhardware: cpu\ntarget_repo: .\nruntime:\n  python: " + sys.executable + "\n  tensor_parallel_size: 1\n",
            encoding="utf-8",
        )
        self.port = free_port()
        self.run_cli("init", "demo/engine")
        runbook = target_dir / "runbook"
        (runbook / "env.sh").write_text(
            f'''#!/usr/bin/env bash\nMODEL_HOST=127.0.0.1\nMODEL_PORT={self.port}\nMODEL_BASE_URL=http://${{MODEL_HOST}}:${{MODEL_PORT}}\nMODEL_STARTUP_TIMEOUT=10\nMODEL_CHECK_TIMEOUT=10\nMODEL_SHUTDOWN_TIMEOUT=3\nMODEL_READY_INTERVAL=0.1\nMODEL_READY_PROBE_TIMEOUT=6\n''',
            encoding="utf-8",
        )
        (runbook / "start.sh").write_text(
            '''#!/usr/bin/env bash\nset -euo pipefail\nexec "$RUNTIME_PYTHON" "$CONTROL_ROOT/tests/fake_server.py" --host "$MODEL_HOST" --port "$MODEL_PORT"\n''',
            encoding="utf-8",
        )
        (runbook / "ready.sh").write_text(
            '''#!/usr/bin/env bash\nset -euo pipefail\n"$RUNTIME_PYTHON" - <<'PY2'\nimport os, urllib.request\nurllib.request.urlopen(os.environ["MODEL_BASE_URL"] + "/health", timeout=1).read()\nPY2\n''',
            encoding="utf-8",
        )
        (runbook / "checks" / "smoke.sh").write_text(
            '''#!/usr/bin/env bash\nset -euo pipefail\n"$RUNTIME_PYTHON" - <<'PY2'\nimport json, os, urllib.request\nreq=urllib.request.Request(os.environ["MODEL_BASE_URL"] + "/v1/chat/completions", data=b"{}", headers={"Content-Type":"application/json"})\nbody=urllib.request.urlopen(req, timeout=2).read()\nopen(os.path.join(os.environ["RUN_DIR"], "smoke.json"), "wb").write(body)\nPY2\n''',
            encoding="utf-8",
        )
        (runbook / "checks" / "validate.sh").write_text(
            '''#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p "$RUN_DIR/validation"\nprintf 'passed\\n' > "$RUN_DIR/validation/result.txt"\n''',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        try:
            self.run_cli("stop", "demo/engine", check=False)
        finally:
            self.temp.cleanup()

    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(self.root / "scripts" / "model_runtime.py"),
            "--repo-root",
            str(self.root),
            *args,
        ]
        return subprocess.run(command, text=True, capture_output=True, check=check, timeout=30)

    def test_ephemeral_smoke_and_cleanup(self) -> None:
        completed = self.run_cli("run", "demo/engine", "--check", "smoke")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "passed")
        run_dir = Path(payload["run_dir"])
        self.assertTrue((run_dir / "smoke.json").exists())
        self.assertEqual(payload["cleanup"]["status"], "passed")

    def test_validation_named_check(self) -> None:
        completed = self.run_cli("run", "demo/engine", "--check", "validate")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "passed")
        self.assertTrue((Path(payload["run_dir"]) / "validation" / "result.txt").exists())

    def test_persistent_lifecycle(self) -> None:
        serve = self.run_cli("serve", "demo/engine")
        payload = json.loads(serve.stdout)
        self.assertEqual(payload["status"], "passed")
        status = json.loads(self.run_cli("status", "demo/engine").stdout)
        self.assertEqual(status["status"], "running")
        execute = json.loads(self.run_cli("exec", "demo/engine", "--check", "smoke").stdout)
        self.assertEqual(execute["status"], "passed")
        stopped = json.loads(self.run_cli("stop", "demo/engine").stdout)
        self.assertEqual(stopped["status"], "passed")
        status2 = json.loads(self.run_cli("status", "demo/engine").stdout)
        self.assertEqual(status2["status"], "stopped")

    def test_init_does_not_overwrite(self) -> None:
        completed = self.run_cli("init", "demo/engine", check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["reason"], "runbook_exists")


if __name__ == "__main__":
    unittest.main()
