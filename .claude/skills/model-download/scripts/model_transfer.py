#!/usr/bin/env python3
"""Download model weights and upload them to BOS."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Iterable


DEFAULT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".json",
    ".txt",
    ".model",
    ".tiktoken",
    ".py",
    ".md",
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "outputs"


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def which_any(candidates: Iterable[str]) -> str | None:
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    return None


HF_HOSTS = ("huggingface.co", "hf.co")
MS_HOSTS = ("modelscope.cn", "www.modelscope.cn")

# path segments that are viewer routes, not part of the repo id
_HF_ROUTE_SEGMENTS = {"tree", "blob", "resolve", "commits", "commit", "raw"}
_MS_ROUTE_SEGMENTS = {"summary", "files", "tree", "blob", "resolve"}


def parse_model_url(value: str) -> tuple[str, str] | None:
    """Return (source, model_id) if value is an HF/ModelScope model URL, else None."""
    text = value.strip()
    if "://" not in text and "/" in text and any(
        text.split("/", 1)[0].endswith(host) or text.split("/", 1)[0] == host
        for host in HF_HOSTS + MS_HOSTS
    ):
        text = "https://" + text
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)/(.*)$", text)
    if not match:
        return None
    host = match.group(1).lower()
    path = match.group(2).strip("/")
    if not path:
        return None
    segments = path.split("/")

    if host in HF_HOSTS:
        # optional "models/" prefix, then org/model, then viewer routes
        if segments and segments[0] in {"models", "datasets", "spaces"}:
            segments = segments[1:]
        repo: list[str] = []
        for seg in segments:
            if seg in _HF_ROUTE_SEGMENTS:
                break
            repo.append(seg)
        model_id = "/".join(repo[:2]) if len(repo) >= 2 else "/".join(repo)
        return ("hf", model_id) if model_id else None

    if host in MS_HOSTS:
        if segments and segments[0] == "models":
            segments = segments[1:]
        repo = []
        for seg in segments:
            if seg in _MS_ROUTE_SEGMENTS:
                break
            repo.append(seg)
        model_id = "/".join(repo[:2]) if len(repo) >= 2 else "/".join(repo)
        return ("modelscope", model_id) if model_id else None

    return None


def resolve_source_and_model(args: argparse.Namespace) -> None:
    """If --model-id is a full HF/ModelScope URL, extract the id and infer --source."""
    if not args.model_id:
        return
    parsed = parse_model_url(args.model_id)
    if parsed is None:
        if not args.source:
            raise SystemExit(
                "cannot auto-detect source: --model-id is not a Hugging Face / ModelScope URL; "
                "pass --source hf|modelscope, or use a full model URL"
            )
        return
    url_source, model_id = parsed
    if args.source and args.source != url_source:
        raise SystemExit(
            f"--source={args.source} conflicts with URL host (looks like {url_source}); "
            "omit --source or pass a matching one"
        )
    args.source = url_source
    args.model_id = model_id
    print(f"[INFO] auto-detected source={url_source}, model id={model_id}", flush=True)


def run(cmd: list[str], env: dict[str, str], log_path: Path | None = None) -> int:
    print("[CMD]", " ".join(cmd), flush=True)
    if log_path is None:
        return subprocess.run(cmd, env=env).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def build_env(proxy: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if proxy:
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
    return env


def bcecmd_sync_command(local_dir: Path, bos_path: str | None, concurrency: int) -> str:
    target = bos_path or "<BOS_model_path>"
    return f"bcecmd bos sync {local_dir} {target} --concurrency {concurrency}"


def archive_path_for(local_dir: Path, pack: str) -> Path:
    """Deterministic archive path next to the local dir, e.g. <name>.tar[.gz]."""
    suffix = ".tar.gz" if pack == "tar.gz" else ".tar"
    return local_dir.parent / f"{local_dir.name}{suffix}"


def bos_archive_target(bos_path: str | None, archive: Path) -> str:
    base = (bos_path or "<BOS_model_path>").rstrip("/")
    return f"{base}/{archive.name}"


def bcecmd_cp_command(archive: Path, bos_path: str | None, concurrency: int) -> str:
    target = bos_archive_target(bos_path, archive)
    return f"bcecmd bos cp {archive} {target} --concurrency {concurrency}"


def upload_command_hint(
    local_dir: Path, bos_path: str | None, concurrency: int, pack: str | None
) -> str:
    """The exact bcecmd line the user should run: cp the archive when packing, else sync the dir."""
    if pack:
        return bcecmd_cp_command(archive_path_for(local_dir, pack), bos_path, concurrency)
    return bcecmd_sync_command(local_dir, bos_path, concurrency)


def build_archive(local_dir: Path, archive: Path, pack: str) -> None:
    """Pack local_dir into archive; the dir name is the top-level entry inside the archive."""
    mode = "w:gz" if pack == "tar.gz" else "w"
    print(f"[PACK] creating {archive} ({pack})", flush=True)
    start = time.time()
    tmp = archive.with_name(archive.name + ".part")
    with tarfile.open(tmp, mode) as tar:
        tar.add(local_dir, arcname=local_dir.name)
    tmp.replace(archive)
    size = archive.stat().st_size
    print(f"[PACK] done {archive} | {format_bytes(size)} | {time.time() - start:.2f}s", flush=True)


def manual_upload_hint(
    local_dir: Path, bos_path: str | None, concurrency: int, pack: str | None = None
) -> str:
    return "\n".join(
        [
            "[MANUAL UPLOAD REQUIRED] bcecmd not found; download continues but upload is skipped.",
            "Install and configure bcecmd, then upload it yourself:",
            f"  {upload_command_hint(local_dir, bos_path, concurrency, pack)}",
        ]
    )


def download_only_upload_hint(
    local_dir: Path, bos_path: str | None, concurrency: int, pack: str | None = None
) -> str:
    return "\n".join(
        [
            "[UPLOAD SKIPPED] download only (--no-upload). To upload to BOS later, run:",
            f"  {upload_command_hint(local_dir, bos_path, concurrency, pack)}",
        ]
    )


def preflight() -> int:
    checks = {
        "hf": which_any(["hf", "huggingface-cli"]),
        "modelscope": shutil.which("modelscope"),
        "bcecmd": shutil.which("bcecmd"),
    }
    for name, path in checks.items():
        print(f"{name}: {path or 'missing'}")

    bcecmd = checks["bcecmd"]
    if bcecmd:
        subprocess.run(["bcecmd", "bos", "--version"], check=False)
    else:
        print(
            manual_upload_hint(
                Path("<Local_model_path>"),
                None,
                64,
            )
        )
    return 0


def download_command(args: argparse.Namespace) -> list[str] | None:
    if args.source == "local":
        return None

    if args.source == "hf":
        hf = which_any(["hf", "huggingface-cli"])
        if hf is None:
            if args.dry_run:
                hf = "hf"
            else:
                raise RuntimeError("missing hf or huggingface-cli")
        if Path(hf).name == "hf":
            return [
                hf,
                "download",
                args.model_id,
                "--local-dir",
                args.local_dir,
                "--max-workers",
                str(args.max_workers),
            ]
        return [
            hf,
            "download",
            "--resume-download",
            args.model_id,
            "--local-dir",
            args.local_dir,
            "--max-workers",
            str(args.max_workers),
        ]

    if args.source == "modelscope":
        modelscope = shutil.which("modelscope")
        if modelscope is None:
            if args.dry_run:
                modelscope = "modelscope"
            else:
                raise RuntimeError("missing modelscope")
        return [
            modelscope,
            "download",
            "--model",
            args.model_id,
            "--local_dir",
            args.local_dir,
            "--max-workers",
            str(args.max_workers),
        ]

    raise RuntimeError(f"unknown source: {args.source}")


def bos_exists(bos_file: str, env: dict[str, str]) -> bool:
    result = subprocess.run(
        ["bcecmd", "bos", "ls", bos_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        return False
    output = result.stdout
    return "STANDARD" in output or "NORMAL" in output or "exit normal" in output


def stable_file(path: Path, wait_seconds: float) -> bool:
    try:
        size1 = path.stat().st_size
        time.sleep(wait_seconds)
        size2 = path.stat().st_size
        return size1 == size2 and size2 > 0
    except OSError:
        return False


def iter_upload_candidates(root: Path, suffixes: tuple[str, ...]) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in suffixes:
            yield path


def summarize_local_dir(root: Path, suffixes: tuple[str, ...]) -> tuple[int, int, int]:
    total_files = 0
    matched_files = 0
    total_bytes = 0
    if not root.exists():
        return total_files, matched_files, total_bytes
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        total_files += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
        if path.suffix in suffixes:
            matched_files += 1
    return total_files, matched_files, total_bytes


def progress_reporter(
    local_root: Path,
    suffixes: tuple[str, ...],
    start_time: float,
    stop_event: threading.Event,
    interval: int,
) -> None:
    if interval <= 0:
        return
    while not stop_event.wait(interval):
        total_files, matched_files, total_bytes = summarize_local_dir(local_root, suffixes)
        elapsed = int(time.time() - start_time)
        print(
            " ".join(
                [
                    "[PROGRESS]",
                    f"elapsed_seconds={elapsed}",
                    f"local_size={format_bytes(total_bytes)}",
                    f"files={total_files}",
                    f"upload_candidates={matched_files}",
                ]
            ),
            flush=True,
        )


def upload_one(
    path: Path,
    local_root: Path,
    bos_root: str,
    env: dict[str, str],
    concurrency: int,
) -> int:
    rel = path.relative_to(local_root).as_posix()
    bos_file = f"{bos_root.rstrip('/')}/{rel}"
    if bos_exists(bos_file, env):
        print(f"[SKIP] {rel} exists on BOS", flush=True)
        return 0
    start = time.time()
    result = run(
        [
            "bcecmd",
            "bos",
            "cp",
            str(path),
            bos_file,
            "--concurrency",
            str(concurrency),
        ],
        env,
    )
    elapsed = max(time.time() - start, 0.001)
    mib = path.stat().st_size / 1024 / 1024
    if result == 0:
        print(f"[DONE] {rel} | {elapsed:.2f}s | {mib / elapsed:.2f} MiB/s", flush=True)
    else:
        print(f"[ERROR] {rel}", flush=True)
    return result


def watch_upload(
    local_root: Path,
    bos_root: str,
    env: dict[str, str],
    stop_event: threading.Event,
    suffixes: tuple[str, ...],
    concurrency: int,
    log_path: Path,
) -> None:
    uploaded: set[Path] = set()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[INFO] watching {local_root}\n")
    while not stop_event.is_set():
        for path in iter_upload_candidates(local_root, suffixes):
            if path in uploaded:
                continue
            if not stable_file(path, 1):
                continue
            code = upload_one(path, local_root, bos_root, env, concurrency)
            if code == 0:
                uploaded.add(path)
        stop_event.wait(5)


def final_sync(local_dir: Path, bos_path: str, env: dict[str, str], concurrency: int) -> int:
    return run(
        [
            "bcecmd",
            "bos",
            "sync",
            str(local_dir),
            bos_path,
            "--concurrency",
            str(concurrency),
        ],
        env,
    )


def upload_archive(archive: Path, bos_path: str, env: dict[str, str], concurrency: int) -> int:
    return run(
        [
            "bcecmd",
            "bos",
            "cp",
            str(archive),
            bos_archive_target(bos_path, archive),
            "--concurrency",
            str(concurrency),
        ],
        env,
    )


def confirm_upload(args: argparse.Namespace, local_dir: Path) -> None:
    if not args.upload or args.yes or args.dry_run:
        return
    if args.pack:
        upload_line = f"Type YES to pack into {args.pack} and upload the archive to BOS:"
    else:
        upload_line = "Type YES to start uploading during download and final BOS sync:"
    print(
        "\n".join(
            [
                "[UPLOAD CONFIRMATION REQUIRED]",
                f"source={args.source}",
                f"model_id={args.model_id or ''}",
                f"local_dir={local_dir}",
                f"pack={args.pack or 'none'}",
                f"bos_path={args.bos_path}",
                upload_line,
            ]
        ),
        flush=True,
    )
    if not sys.stdin.isatty():
        raise SystemExit("upload confirmation required; pass --yes only when explicitly approved")
    answer = input("> ").strip()
    if answer != "YES":
        raise SystemExit("upload cancelled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face or ModelScope model weights and upload to BOS.",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--source", choices=["hf", "modelscope", "local"])
    parser.add_argument(
        "--model-id",
        help="repo id (org/model) or a full Hugging Face / ModelScope model URL",
    )
    parser.add_argument("--local-dir")
    parser.add_argument("--bos-path")
    parser.add_argument("--proxy")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--pack",
        choices=["tar", "tar.gz"],
        help="pack the downloaded dir into a tar/tar.gz archive and upload that archive instead of syncing files",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--upload-concurrency", type=int, default=64)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=60,
        help="seconds between progress reports; set 0 to disable",
    )
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip upload confirmation only when explicitly requested",
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.preflight:
        return
    if not args.source:
        raise SystemExit("--source is required")
    if not args.local_dir:
        raise SystemExit("--local-dir is required")
    if args.source != "local" and not args.model_id:
        raise SystemExit("--model-id is required for downloads")
    if args.upload and not args.bos_path:
        raise SystemExit("--bos-path is required when --upload is set")


def main() -> int:
    args = parse_args()
    if not args.preflight:
        resolve_source_and_model(args)
    validate(args)
    if args.preflight:
        return preflight()

    local_dir = Path(args.local_dir).expanduser().resolve()
    env = build_env(args.proxy)
    suffixes = tuple(args.suffixes or DEFAULT_SUFFIXES)

    upload_requested = args.upload
    if args.upload and shutil.which("bcecmd") is None:
        print(
            manual_upload_hint(local_dir, args.bos_path, args.upload_concurrency, args.pack),
            flush=True,
        )
        args.upload = False

    download = download_command(args)
    confirm_upload(args, local_dir)
    if args.dry_run:
        if download:
            print("download:", " ".join(download))
        if args.pack:
            archive = archive_path_for(local_dir, args.pack)
            print("pack:", f"{local_dir} -> {archive} ({args.pack})")
            if args.upload:
                print("upload:", f"pack then cp {archive} to {bos_archive_target(args.bos_path, archive)}")
        elif args.upload:
            print("upload:", f"watch {local_dir} then sync to {args.bos_path}")
        print("progress:", f"every {args.progress_interval}s" if args.progress_interval > 0 else "disabled")
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    upload_stop_event = threading.Event()
    progress_stop_event = threading.Event()
    watcher: threading.Thread | None = None
    progress: threading.Thread | None = None
    start = time.time()
    if args.progress_interval > 0:
        progress = threading.Thread(
            target=progress_reporter,
            args=(local_dir, suffixes, start, progress_stop_event, args.progress_interval),
            daemon=True,
        )
        progress.start()
    # Packing uploads a single archive built after download, so skip the
    # incremental file watcher; it only helps when syncing individual files.
    if args.upload and not args.pack:
        watcher = threading.Thread(
            target=watch_upload,
            args=(
                local_dir,
                args.bos_path,
                env,
                upload_stop_event,
                suffixes,
                args.upload_concurrency,
                OUTPUT_DIR / "watch-upload.log",
            ),
            daemon=True,
        )
        watcher.start()

    download_code = 0
    try:
        if download:
            download_code = run(download, env, OUTPUT_DIR / "download.log")
        else:
            print("[INFO] source=local, skipping download")
    finally:
        upload_stop_event.set()
        if watcher:
            watcher.join(timeout=10)

    archive: Path | None = None
    if args.pack:
        archive = archive_path_for(local_dir, args.pack)
        build_archive(local_dir, archive, args.pack)

    upload_code = 0
    if args.upload:
        if args.pack and archive is not None:
            upload_code = upload_archive(archive, args.bos_path, env, args.upload_concurrency)
        else:
            upload_code = final_sync(local_dir, args.bos_path, env, args.upload_concurrency)

    progress_stop_event.set()
    if progress:
        progress.join(timeout=10)

    if upload_requested and not args.upload:
        upload_status = "skipped_manual"
    elif args.upload:
        upload_status = "passed" if upload_code == 0 else "failed"
    else:
        upload_status = "disabled"

    elapsed = int(time.time() - start)
    print(
        "\n".join(
            [
                "[SUMMARY]",
                f"source={args.source}",
                f"model_id={args.model_id or ''}",
                f"local_dir={local_dir}",
                f"pack={args.pack or 'none'}",
                f"archive={archive or ''}",
                f"bos_path={args.bos_path or ''}",
                f"download_status={'passed' if download_code == 0 else 'failed'}",
                f"upload_status={upload_status}",
                f"output_dir={OUTPUT_DIR}",
                f"elapsed_seconds={elapsed}",
            ]
        ),
        flush=True,
    )
    if upload_requested and not args.upload:
        print(
            manual_upload_hint(local_dir, args.bos_path, args.upload_concurrency, args.pack),
            flush=True,
        )
    elif not upload_requested:
        print(
            download_only_upload_hint(local_dir, args.bos_path, args.upload_concurrency, args.pack),
            flush=True,
        )
    return download_code or upload_code


if __name__ == "__main__":
    sys.exit(main())
