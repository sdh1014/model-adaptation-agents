# KLX P800 Model Migration Automation

This context defines the language used to design a spec-driven tool for migrating model operators from the SGLang CUDA path to KLX P800.

## Language

**Migration Spec**:
The durable contract and progress record that keeps the agent aligned during execution and after context compression. It contains the goal, scope, current operator, acceptance criteria, decisions, and evidence pointers.
_Avoid_: Design document, prompt

**Contract**:
The human-approved, agent-read-only portion of the Migration Spec containing the goal, scope, success criteria, environment boundary, and stop rules.
_Avoid_: Working state, editable prompt

**Working State**:
The agent-maintained portion of the Migration Spec containing the current phase, gap queue, active operator, Run pointers, next action, decisions, and evidence pointers.
_Avoid_: Contract, chat history

**Migration Agent**:
The single decision-making actor that reads the Migration Spec, analyzes source and evidence, selects the next action, generates a permitted repair, and updates Working State.
_Avoid_: Multi-agent coordinator, deterministic runner

**Deterministic Tool**:
A small script invoked by the Migration Agent to capture, replay, or compare data without deciding scope, changing acceptance criteria, or selecting the next migration action.
_Avoid_: Autonomous agent, workflow engine

**Operator Gap**:
A model-path operation whose CUDA implementation exists but whose Kunlun implementation is absent or cannot reproduce the required behavior on P800.
_Avoid_: Generic support matrix, source-code difference

**Semantic Operator**:
The smallest model-path computation with explicit replayable inputs and outputs that can be replaced independently at its SGLang call site. It may contain several basic PyTorch operations.
_Avoid_: Every tensor primitive, entire decoder layer

**READY**:
A scanned semantic operator whose P800-executable implementation and required behavior can both be established from available evidence.

**CAPTURE_REQUIRED**:
A scanned semantic operator that lacks a Kunlun implementation, depends on a CUDA-only path, or cannot be judged semantically from source alone and therefore requires CUDA Golden Capture.

**NEEDS_HUMAN**:
A scanned path whose call relationship or semantic operator boundary cannot be determined reliably by the agent.

**CUDA Golden Capture**:
The one-time collection of replayable operator inputs and expected outputs from an instrumented SGLang CUDA run for the identified Operator Gaps.
_Avoid_: Repeated cross-device comparison

**Capture Session**:
The single CUDA-side execution that gathers all required Golden Samples, retaining one sample for each distinct execution phase, shape, dtype, layout, and non-tensor argument signature.
_Avoid_: One sample total, repeated CUDA visits

**Golden Sample**:
A self-replay-verified record of one Semantic Operator invocation, including the exact inputs, expected outputs, and execution context required for offline P800 replay.
_Avoid_: Output-only snapshot, arbitrary hidden-state dump

**Precision Gate**:
The human-approved validation rule in the Contract that requires matching output structure and dtype, finite values, and configured absolute and relative tolerances for every Demo shape.
_Avoid_: Agent-adjusted tolerance, diagnostic metric

**Run**:
An immutable record of one CUDA capture or P800 repair attempt, containing its inputs, commands, logs, results, and candidate patch when applicable.
_Avoid_: Output, current state

**Golden Run**:
The Run produced once on the CUDA machine and transferred to the P800 machine as the reusable reference for later repair attempts.
_Avoid_: Shared live CUDA service, repeated capture

**Handoff Bundle**:
The Migration Spec plus its referenced Golden Run and integrity manifest, generated and validated by the tool for a one-time human-managed transfer from the CUDA machine to the P800 machine.
_Avoid_: Remote execution session, automatic credentialed transfer

**P800 Repair Loop**:
The repeated process on the P800 machine that replays captured inputs, creates or selects a Kunlun implementation, compares results with the CUDA Golden Capture, and records evidence in the Migration Spec.
_Avoid_: CUDA-P800 online loop

**Repair Boundary**:
The agent's permitted repair scope: one Semantic Operator implemented through P800-executable Python, PyTorch, or existing xspeedgate operations, plus its focused replay test.
_Avoid_: Decoder-layer changes, model assembly, new native-kernel registration

**BLOCKED**:
A Working State indicating that progress requires work outside the Contract or Repair Boundary, so the agent must preserve evidence and stop instead of expanding scope.
_Avoid_: Failed attempt, silent fallback

**Demo Closure**:
The minimum proof that scans the complete Step-3.7-Flash operator path, records every discovered Operator Gap, and closes one real gap for no more than three observed shapes through the P800 Repair Loop. Closing every gap or covering every shape belongs to the tool's later operating goal, not the demo gate.
_Avoid_: Full Step-3.7 migration, single-op-only scan

**Solution Design Document**:
The deliverable of this wayfinding effort, describing the minimal tool, its runtime flow, artifact contracts, boundaries, and implementation plan.
_Avoid_: Migration Spec, runnable demo
