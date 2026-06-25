#!/usr/bin/env bash
# shellcheck shell=bash

_benchmark_python() {
  if [[ -n "${RUNTIME_PYTHON:-}" && -x "${RUNTIME_PYTHON}" ]]; then printf '%s' "$RUNTIME_PYTHON"; else command -v python3 || command -v python; fi
}

benchmark_init() {
  : "${RUN_DIR:?RUN_DIR is required}"
  export BENCHMARK_DIR="${BENCHMARK_DIR:-$RUN_DIR/benchmark}"
  mkdir -p "$BENCHMARK_DIR/cases"
  : > "$BENCHMARK_DIR/cases.jsonl"
}

benchmark_case() {
  local id="${1:?case id required}" requirement="${2:?required|optional required}"
  shift 2
  local warmup=0 repeat=1
  while [[ $# -gt 0 && "${1:-}" != "--" ]]; do
    case "$1" in
      --warmup) warmup="$2"; shift 2 ;;
      --repeat) repeat="$2"; shift 2 ;;
      *) echo "unknown benchmark_case option: $1" >&2; return 2 ;;
    esac
  done
  [[ "${1:-}" == "--" ]] || { echo "benchmark_case requires -- before command" >&2; return 2; }
  shift
  [[ "$requirement" == "required" || "$requirement" == "optional" ]] || { echo "invalid requirement" >&2; return 2; }
  local safe="${id//[^A-Za-z0-9._-]/_}"
  local dir="$BENCHMARK_DIR/cases/$safe"
  mkdir -p "$dir/warmup" "$dir/samples"
  local i rc failed=0 blocked=0
  for ((i=1;i<=warmup;i++)); do
    export BENCHMARK_PHASE=warmup BENCHMARK_SAMPLE_INDEX="$i" BENCHMARK_SAMPLE_FILE="$dir/warmup/$i.json"
    set +e; "$@" >"$dir/warmup/$i.stdout.log" 2>"$dir/warmup/$i.stderr.log"; rc=$?; set -e
    [[ $rc -eq 0 ]] || { failed=1; break; }
  done
  if [[ $failed -eq 0 ]]; then
    for ((i=1;i<=repeat;i++)); do
      export BENCHMARK_PHASE=measure BENCHMARK_SAMPLE_INDEX="$i" BENCHMARK_SAMPLE_FILE="$dir/samples/$i.json"
      set +e; "$@" >"$dir/samples/$i.stdout.log" 2>"$dir/samples/$i.stderr.log"; rc=$?; set -e
      if [[ $rc -eq 64 ]]; then blocked=1; elif [[ $rc -ne 0 ]]; then failed=1; fi
    done
  fi
  local status=passed
  [[ $blocked -eq 0 ]] || status=blocked
  [[ $failed -eq 0 ]] || status=failed
  "$(_benchmark_python)" - "$BENCHMARK_DIR/cases.jsonl" "$id" "$requirement" "$status" "$warmup" "$repeat" "$dir" "$@" <<'PY'
import json,sys,datetime
path,id_,req,status,warmup,repeat,dir_,*argv=sys.argv[1:]
record={"id":id_,"required":req=="required","status":status,"warmup":int(warmup),"repeat":int(repeat),"argv":argv,"case_dir":dir_,"finished_at":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
with open(path,"a",encoding="utf-8") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
PY
  return 0
}

benchmark_finish() {
  "$(_benchmark_python)" - "$BENCHMARK_DIR/cases.jsonl" "$BENCHMARK_DIR/result.json" <<'PY'
import json,sys,glob,statistics,math,os
source,out=sys.argv[1:]
cases=[]
for line in open(source,encoding="utf-8"):
    line=line.strip()
    if line: cases.append(json.loads(line))
required=[c for c in cases if c.get("required")]
if not cases: status="blocked"
elif any(c["status"]=="failed" for c in required): status="failed"
elif any(c["status"]=="blocked" for c in required): status="blocked"
else: status="passed"
aggregated={}
for case in cases:
    values={}
    for path in sorted(glob.glob(os.path.join(case["case_dir"],"samples","*.json"))):
        try: data=json.load(open(path,encoding="utf-8"))
        except Exception: continue
        node=data.get("metrics",data) if isinstance(data,dict) else {}
        if not isinstance(node,dict): continue
        for key,value in node.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool): values.setdefault(key,[]).append(float(value))
    metrics={}
    for key,series in values.items():
        ordered=sorted(series)
        def pct(p):
            if not ordered:return None
            idx=max(0,min(len(ordered)-1,math.ceil(p*len(ordered))-1));return ordered[idx]
        metrics[key]={"count":len(series),"mean":statistics.fmean(series),"median":statistics.median(series),"p90":pct(.90),"p95":pct(.95),"p99":pct(.99),"min":min(series),"max":max(series)}
    aggregated[case["id"]]={"status":case["status"],"warmup":case["warmup"],"repeat":case["repeat"],"metrics":metrics}
primary_metrics={}
for case in cases:
    if case.get("required") and aggregated.get(case["id"],{}).get("metrics"):
        primary_metrics={k:v["mean"] for k,v in aggregated[case["id"]]["metrics"].items()}
        break
if not primary_metrics:
    for case in cases:
        if aggregated.get(case["id"],{}).get("metrics"):
            primary_metrics={k:v["mean"] for k,v in aggregated[case["id"]]["metrics"].items()}
            break
payload={"schema_version":1,"status":status,"cases":cases,"metrics":primary_metrics,"aggregated":aggregated,"target_met":None}
json.dump(payload,open(out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(status)
sys.exit(0 if status=="passed" else (64 if status=="blocked" else 1))
PY
}
