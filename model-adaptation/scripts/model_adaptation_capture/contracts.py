"""Shared names for the capture CLI and its SGLang plugin."""

CAPTURE_CONFIG_ENV = "MODEL_ADAPTATION_CAPTURE_CONFIG"
CONFIG_SCHEMA = "golden-capture-config/v0"
STATE_SCHEMA = "golden-capture-state/v0"
CANDIDATE_SAMPLE_SCHEMA = "capture-candidate/v0"
CANDIDATE_SERIALIZATION = "candidate-torch-save/v1"

OPERATOR_ID = "activation.step_swiglu_with_limit"
PLUGIN_ENTRY_POINT = "model_adaptation_capture"
FORWARD_HOOK_TARGET = "sglang.srt.models.step3p5.Step3p5ForCausalLM.forward"
SWIGLU_HOOK_TARGET = "sglang.srt.models.step3p5_ops.step_swiglu_with_limit"
HELPER_RELATIVE_PATH = "python/sglang/srt/models/step3p5_ops.py"
