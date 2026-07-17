"""Shared names for the capture CLI and its SGLang plugin."""

import hashlib
import json
from typing import Any


CAPTURE_CONFIG_ENV = "MODEL_ADAPTATION_CAPTURE_CONFIG"
REPLAY_CONFIG_ENV = "MODEL_ADAPTATION_REPLAY_CONFIG"
CONFIG_SCHEMA = "golden-capture-config/v1"
STATE_SCHEMA = "golden-capture-state/v2"
CANDIDATE_SAMPLE_SCHEMA = "capture-candidate/v1"
CANDIDATE_SERIALIZATION = "candidate-torch-save/v2"
REPLAY_CONFIG_SCHEMA = "loaded-model-replay-config/v1"
REPLAY_RESULT_SCHEMA = "loaded-model-replay-result/v1"

OPERATOR_ID = "Step3p5MLP.forward"
ACTIVATION_GUARD = "self.limit is not None"
VALIDATION_TP_RANK = 0
PLUGIN_ENTRY_POINT = "model_adaptation_capture"
FORWARD_HOOK_TARGET = "sglang.srt.models.step3p5.Step3p5ForCausalLM.forward"
MLP_HOOK_TARGET = "sglang.srt.models.step3p5.Step3p5MLP.forward"
MODEL_RELATIVE_PATH = "python/sglang/srt/models/step3p5.py"
FIXED_PRECISION_GATE = {
    "comparator": "torch.testing.assert_close",
    "atol": 0.01,
    "rtol": 0.02,
    "require_exact_structure": True,
    "require_same_dtype": True,
    "require_finite": True,
    "check_stride": False,
}


def checkpoint_metadata(checkpoint_id: str, config_digest: str) -> dict[str, str]:
    model_path, separator, revision = checkpoint_id.rpartition("@")
    if not separator or not model_path or not revision:
        raise ValueError(
            "checkpoint id must use '<model-path>@<revision>'"
        )
    if not isinstance(config_digest, str) or not config_digest:
        raise ValueError("checkpoint config digest must be a non-empty string")
    return {
        "id": checkpoint_id,
        "model_path": model_path,
        "revision": revision,
        "config_digest": config_digest,
    }


def shape_id_for(shape: Any) -> str:
    if (
        not isinstance(shape, (list, tuple))
        or not shape
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in shape
        )
    ):
        raise ValueError("tensor shape must be a non-empty list of dimensions")
    encoded = json.dumps(
        {"x": list(shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
