"""Shared names for the capture CLI and its SGLang plugin."""

CAPTURE_CONFIG_ENV = "MODEL_ADAPTATION_CAPTURE_CONFIG"
REPLAY_CONFIG_ENV = "MODEL_ADAPTATION_REPLAY_CONFIG"
CONFIG_SCHEMA = "golden-capture-config/v1"
STATE_SCHEMA = "golden-capture-state/v1"
CANDIDATE_SAMPLE_SCHEMA = "capture-candidate/v1"
CANDIDATE_SERIALIZATION = "candidate-torch-save/v2"
REPLAY_CONFIG_SCHEMA = "loaded-model-replay-config/v0"
REPLAY_RESULT_SCHEMA = "loaded-model-replay-result/v0"

OPERATOR_ID = "Step3p5MLP.forward"
ACTIVATION_GUARD = "self.limit is not None"
VALIDATION_TP_RANK = 0
PLUGIN_ENTRY_POINT = "model_adaptation_capture"
FORWARD_HOOK_TARGET = "sglang.srt.models.step3p5.Step3p5ForCausalLM.forward"
MLP_HOOK_TARGET = "sglang.srt.models.step3p5.Step3p5MLP.forward"
MODEL_RELATIVE_PATH = "python/sglang/srt/models/step3p5.py"
