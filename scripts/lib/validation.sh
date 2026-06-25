#!/usr/bin/env bash
# shellcheck shell=bash

_validation_python() {
  if [[ -n "${RUNTIME_PYTHON:-}" && -x "${RUNTIME_PYTHON}" ]]; then printf '%s' "$RUNTIME_PYTHON"; else command -v python3 || command -v python; fi
}

validation_init() {
  : "${RUN_DIR:?RUN_DIR is required}"
  export VALIDATION_DIR="${VALIDATION_DIR:-$RUN_DIR/validation}"
  mkdir -p "$VALIDATION_DIR/cases"
  : > "$VALIDATION_DIR/cases.jsonl"
}

validation_case() {
  local id="${1:?case id required}" requirement="${2:?required|optional required}"
  shift 2
  [[ "${1:-}" == "--" ]] || { echo "validation_case requires -- before command" >&2; return 2; }
  shift
  [[ "$requirement" == "required" || "$requirement" == "optional" ]] || { echo "invalid requirement: $requirement" >&2; return 2; }
  local safe="${id//[^A-Za-z0-9._-]/_}"
  local dir="$VALIDATION_DIR/cases/$safe"
  mkdir -p "$dir"
  local started rc status
  started="$(date -Iseconds)"
  set +e
  "$@" >"$dir/stdout.log" 2>"$dir/stderr.log"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then status=passed; elif [[ $rc -eq 64 ]]; then status=blocked; else status=failed; fi
  "$( _validation_python )" - "$VALIDATION_DIR/cases.jsonl" "$id" "$requirement" "$status" "$rc" "$started" "$dir" "$@" <<'PY'
import json,sys,datetime
path,id_,req,status,rc,started,dir_,*argv=sys.argv[1:]
record={"id":id_,"required":req=="required","status":status,"exit_code":int(rc),"started_at":started,"finished_at":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),"argv":argv,"evidence":[f"cases/{dir_.split('/')[-1]}/stdout.log",f"cases/{dir_.split('/')[-1]}/stderr.log"]}
with open(path,"a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
PY
  return 0
}

validation_finish() {
  local py
  py="$(_validation_python)"
  "$py" - "$VALIDATION_DIR/cases.jsonl" "$VALIDATION_DIR/result.json" <<'PY'
import json,sys
source,out=sys.argv[1:]
cases=[]
with open(source,encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if line: cases.append(json.loads(line))
required=[c for c in cases if c.get("required")]
if not cases: status="blocked"; summary="no validation cases"
elif any(c["status"]=="failed" for c in required): status="failed"; summary="required case failed"
elif any(c["status"]=="blocked" for c in required): status="blocked"; summary="required case blocked"
elif any(c["status"] not in {"passed","skipped"} for c in cases): status="partial"; summary="optional or unknown cases remain"
else: status="passed"; summary="all required cases passed"
with open(out,"w",encoding="utf-8") as f: json.dump({"schema_version":1,"status":status,"summary":summary,"cases":cases,"deviations":[],"blockers":[]},f,ensure_ascii=False,indent=2)
print(status)
sys.exit(0 if status in {"passed","partial"} else (64 if status=="blocked" else 1))
PY
}
