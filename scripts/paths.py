#!/usr/bin/env python3
"""Canonical task and flat run timeline paths."""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Sequence

ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_id(value: str, label: str) -> str:
    if not ID_RE.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def parse_target_ref(value: str) -> tuple[str, str]:
    if value.count('/') != 1:
        raise ValueError('target must be <model-id>/<target-id>')
    model_id, target_id = value.split('/', 1)
    return validate_id(model_id, 'model id'), validate_id(target_id, 'target id')


def model_run_key(model_id: str) -> str:
    return validate_id(model_id, 'model id')


def target_run_key(model_id: str, target_id: str) -> str:
    return f"{validate_id(model_id, 'model id')}--{validate_id(target_id, 'target id')}"


def model_run_root(repo_root: Path, model_id: str) -> Path:
    return repo_root / 'runs' / model_run_key(model_id)


def target_run_root(repo_root: Path, model_id: str, target_id: str) -> Path:
    return repo_root / 'runs' / target_run_key(model_id, target_id)


def safe_part(value: str, label: str) -> str:
    value = re.sub(r'[^A-Za-z0-9._-]+', '-', value).strip('-')
    if not value:
        raise ValueError(f'empty {label}')
    return value


def run_name(stage: str, detail: str | None = None, stamp: str | None = None) -> str:
    parts = [stamp or dt.datetime.now().strftime('%Y%m%d-%H%M%S'), safe_part(stage, 'stage')]
    if detail:
        parts.append(safe_part(detail, 'detail'))
    return '-'.join(parts)


def unique_run_dir(run_root: Path, stage: str, detail: str | None = None) -> Path:
    base = run_root / run_name(stage, detail)
    path = base
    index = 1
    while path.exists():
        path = base.with_name(f'{base.name}-{index}')
        index += 1
    path.mkdir(parents=True)
    return path


def create_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.target:
        model_id, target_id = parse_target_ref(args.target)
        root = target_run_root(repo_root, model_id, target_id)
    elif args.model:
        root = model_run_root(repo_root, args.model)
    else:
        raise SystemExit('create-run requires --model or --target')
    print(unique_run_dir(root, args.stage, args.detail))
    return 0


def show_root(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.target:
        model_id, target_id = parse_target_ref(args.target)
        print(target_run_root(repo_root, model_id, target_id))
    elif args.model:
        print(model_run_root(repo_root, args.model))
    else:
        raise SystemExit('root requires --model or --target')
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Canonical repository run paths')
    p.add_argument('--repo-root', default='.')
    sub = p.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create-run')
    create.add_argument('--model')
    create.add_argument('--target')
    create.add_argument('--stage', required=True)
    create.add_argument('--detail')
    create.set_defaults(handler=create_run)
    root = sub.add_parser('root')
    root.add_argument('--model')
    root.add_argument('--target')
    root.set_defaults(handler=show_root)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == '__main__':
    raise SystemExit(main())
