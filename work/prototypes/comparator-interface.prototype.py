#!/usr/bin/env python3
"""PROTOTYPE ONLY: inspect the proposed comparator request/result states.

This does not perform numerical comparison and intentionally has no Torch
dependency. Production comparison is designed to call
torch.testing.assert_close on the P800 machine.
"""

from __future__ import annotations

import argparse
import json


BASE_REQUEST = {
    "spec_path": "migration-spec.md",
    "golden_sample_dir": "runs/golden-001/samples/sample-001",
    "actual_output": "in-memory P800 replay result",
    "precision_gate_source": "Contract",
    "backend": "torch.testing.assert_close",
}


SCENARIOS = {
    "pass": {
        "request": BASE_REQUEST,
        "result": {
            "sample_id": "sample-001",
            "passed": True,
            "backend": "torch.testing.assert_close",
            "precision_gate": {"atol": 0.01, "rtol": 0.01},
            "checks": {
                "structure_match": True,
                "dtype_match": True,
                "expected_finite": True,
                "actual_finite": True,
            },
            "tensors": [
                {
                    "path": "result",
                    "shape": [1, 6144],
                    "dtype": "torch.bfloat16",
                    "passed": True,
                    "max_abs_diff": 0.0078125,
                    "mismatch_count": 0,
                    "message": None,
                }
            ],
            "summary": "所有输出通过 Contract 固定容差",
        },
    },
    "structure": {
        "request": BASE_REQUEST,
        "result": {
            "sample_id": "sample-001",
            "passed": False,
            "backend": "torch.testing.assert_close",
            "precision_gate": {"atol": 0.01, "rtol": 0.01},
            "checks": {
                "structure_match": False,
                "dtype_match": None,
                "expected_finite": None,
                "actual_finite": None,
            },
            "tensors": [],
            "summary": "expected 为 tuple，actual 为 list；未进入数值比较",
        },
    },
    "nonfinite": {
        "request": BASE_REQUEST,
        "result": {
            "sample_id": "sample-001",
            "passed": False,
            "backend": "torch.testing.assert_close",
            "precision_gate": {"atol": 0.01, "rtol": 0.01},
            "checks": {
                "structure_match": True,
                "dtype_match": True,
                "expected_finite": True,
                "actual_finite": False,
            },
            "tensors": [
                {
                    "path": "result",
                    "shape": [1, 6144],
                    "dtype": "torch.bfloat16",
                    "passed": False,
                    "max_abs_diff": None,
                    "mismatch_count": None,
                    "message": "actual 含非有限值；未进入 assert_close",
                }
            ],
            "summary": "P800 输出不满足有限值要求",
        },
    },
    "numeric": {
        "request": BASE_REQUEST,
        "result": {
            "sample_id": "sample-001",
            "passed": False,
            "backend": "torch.testing.assert_close",
            "precision_gate": {"atol": 0.01, "rtol": 0.01},
            "checks": {
                "structure_match": True,
                "dtype_match": True,
                "expected_finite": True,
                "actual_finite": True,
            },
            "tensors": [
                {
                    "path": "result",
                    "shape": [1, 6144],
                    "dtype": "torch.bfloat16",
                    "passed": False,
                    "max_abs_diff": 0.03125,
                    "mismatch_count": 7,
                    "message": "Tensor-likes are not close",
                }
            ],
            "summary": "result 未通过 Contract 固定容差",
        },
    },
}


def show(name: str) -> None:
    print(f"\n=== {name} ===")
    print(json.dumps(SCENARIOS[name], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", choices=[*SCENARIOS, "all"])
    args = parser.parse_args()

    selected = args.scenario
    if selected is None:
        print("选择场景：")
        names = list(SCENARIOS)
        for index, name in enumerate(names, start=1):
            print(f"  {index}. {name}")
        selected = names[int(input("> ").strip()) - 1]

    if selected == "all":
        for name in SCENARIOS:
            show(name)
    else:
        show(selected)


if __name__ == "__main__":
    main()
