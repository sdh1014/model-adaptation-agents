#!/usr/bin/env python3
"""Execute adapt-validate or adapt-benchmark through the target Runbook."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from paths import target_run_root

EXIT_OK, EXIT_BLOCKED, EXIT_FAILED = 0, 2, 3
TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
VALID_STATES = {"passed", "failed", "blocked", "partial", "skipped", "unknown"}


def now_iso() -> str: return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
def timestamp() -> str: return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(json.dumps(dict(data),ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); os.replace(temp,path)


def sha256(path: Path) -> str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def discover_root(explicit: str|None) -> Path:
    if explicit: return Path(explicit).expanduser().resolve()
    current=Path.cwd().resolve()
    for candidate in (current,*current.parents):
        if (candidate/".claude").is_dir() and (candidate/"tasks").is_dir(): return candidate
    return current


def parse_target(value: str) -> tuple[str,str]:
    if not TARGET_RE.fullmatch(value): raise ValueError("target must be <model-id>/<target-id>")
    return tuple(value.split("/",1))  # type: ignore[return-value]


def paths_for(root: Path,target: str) -> dict[str,Path]:
    model,target_id=parse_target(target); model_dir=root/"tasks"/model; target_dir=model_dir/"targets"/target_id
    return {"model_yaml":model_dir/"model.yaml","analysis":model_dir/"model-analysis.md","target_yaml":target_dir/"target.yaml","assessment":target_dir/"assessment.md","implementation":target_dir/"implementation.md","validation":target_dir/"validation.md","benchmark":target_dir/"benchmark.md","runbook":target_dir/"runbook","run_root":target_run_root(root,model,target_id)}


def scalar(value: str) -> Any:
    value=value.strip()
    if value in {"","null","~"}: return None if value else ""
    if value.lower() in {"true","false"}: return value.lower()=="true"
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')): return value[1:-1]
    if re.fullmatch(r"-?\d+",value): return int(value)
    if re.fullmatch(r"-?\d+\.\d+",value): return float(value)
    return value


def simple_yaml(text: str) -> dict[str,Any]:
    result: dict[str, Any] = {}
    stack:list[tuple[int,dict[str,Any]]]=[(-1,result)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or ":" not in raw: continue
        indent=len(raw)-len(raw.lstrip(" ")); key,value=raw.strip().split(":",1)
        while len(stack)>1 and indent<=stack[-1][0]: stack.pop()
        parent=stack[-1][1]
        if not value.strip(): child: dict[str, Any] = {}; parent[key]=child; stack.append((indent,child))
        else: parent[key]=scalar(value)
    return result


def frontmatter(path: Path) -> dict[str,Any]:
    if not path.is_file(): return {}
    lines=path.read_text(encoding="utf-8",errors="replace").splitlines()
    if not lines or lines[0].strip()!="---": return {}
    body=[]
    for line in lines[1:]:
        if line.strip()=="---": break
        body.append(line)
    return simple_yaml("\n".join(body))


def git_fingerprint(path: str|None) -> dict[str,Any]|None:
    if not path: return None
    repo=Path(path); result={"path":str(repo),"exists":repo.exists()}
    if not repo.is_dir(): return result
    def run(*args:str): return subprocess.run(["git","-C",str(repo),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10,check=False)
    probe=run("rev-parse","--is-inside-work-tree")
    if probe.returncode: result["git"]=False; return result
    head=run("rev-parse","HEAD"); branch=run("branch","--show-current"); status=run("status","--porcelain=v1"); diff=run("diff","--binary","--no-ext-diff")
    result.update({"git":True,"head":head.stdout.decode().strip() if not head.returncode else None,"branch":branch.stdout.decode().strip() if not branch.returncode else None,"dirty":bool(status.stdout.strip()),"status_sha256":hashlib.sha256(status.stdout).hexdigest(),"diff_sha256":hashlib.sha256(diff.stdout).hexdigest()})
    return result


def input_fingerprint(root: Path,target: str,stage: str,check: str) -> dict[str,Any]:
    paths=paths_for(root,target); target_cfg={}
    try:
        import yaml  # type: ignore
        target_cfg=yaml.safe_load(paths["target_yaml"].read_text()) or {}
    except Exception: target_cfg=simple_yaml(paths["target_yaml"].read_text()) if paths["target_yaml"].is_file() else {}
    file_paths=[paths["model_yaml"],paths["analysis"],paths["target_yaml"],paths["assessment"],paths["implementation"]]
    if stage=="benchmark": file_paths.append(paths["validation"])
    file_paths += [paths["runbook"]/name for name in ("env.sh","start.sh","ready.sh","stop.sh",f"checks/{check}.sh")]
    hashes={str(path.relative_to(root)):sha256(path) for path in file_paths if path.is_file()}
    def repo_path(value: Any) -> str | None:
        if not value:
            return None
        path=Path(str(value)).expanduser()
        if not path.is_absolute(): path=(root/path).resolve()
        return str(path)
    return {"files":hashes,"frontmatter":{"analysis":frontmatter(paths["analysis"]),"assessment":frontmatter(paths["assessment"]),"implementation":frontmatter(paths["implementation"]),"validation":frontmatter(paths["validation"])},"repositories":{"target_repo":git_fingerprint(repo_path(target_cfg.get("target_repo"))),"upstream_repo":git_fingerprint(repo_path(target_cfg.get("upstream_repo")))},"python":{"executable":sys.executable,"version":sys.version.splitlines()[0]}}


def preconditions(root: Path,target: str,stage: str,check: str,allow_incomplete: bool,allow_unvalidated: bool) -> dict[str,Any]:
    paths=paths_for(root,target); required=[paths["model_yaml"],paths["target_yaml"],paths["assessment"]]
    missing=[str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing: return {"status":"blocked","reason":"missing_inputs","missing":missing}
    script=paths["runbook"]/"checks"/f"{check}.sh"
    if not script.is_file(): return {"status":"blocked","reason":"check_script_missing","path":str(script)}
    if "MODEL_RUN_NOT_CONFIGURED" in script.read_text(encoding="utf-8",errors="replace"): return {"status":"blocked","reason":"check_script_not_configured","path":str(script)}
    assess=frontmatter(paths["assessment"])
    if assess.get("status")!="passed": return {"status":"blocked","reason":"assessment_not_passed","observed":assess.get("status")}
    if stage=="validate" and not allow_incomplete:
        impl=frontmatter(paths["implementation"]); result=assess.get("result")
        if result!="already_supported" and impl.get("status") not in {"passed","not_required"}: return {"status":"blocked","reason":"implementation_not_complete","observed":impl.get("status") or "missing","hint":"use --allow-incomplete only for diagnostics"}
    if stage=="benchmark" and not allow_unvalidated:
        val=frontmatter(paths["validation"])
        if val.get("status")!="passed": return {"status":"blocked","reason":"validation_not_passed","observed":val.get("status") or "missing"}
    return {"status":"passed"}


def stage_dir(root: Path,target: str,stage: str,explicit: str|None) -> Path:
    if explicit:
        path=Path(explicit).expanduser(); path=path if path.is_absolute() else (root/path).resolve()
        if path.exists() and any(path.iterdir()): raise FileExistsError(f"stage run directory is not empty: {path}")
        path.mkdir(parents=True,exist_ok=True); return path
    base=paths_for(root,target)["run_root"]/f"{timestamp()}-{stage}"; path=base; i=1
    while path.exists(): path=Path(f"{base}-{i}"); i+=1
    path.mkdir(parents=True); return path


def invoke_runtime(root: Path,target: str,check: str,out: Path,against: bool,timeout: float|None) -> dict[str,Any]:
    runtime=root/"scripts/model_runtime.py"; runtime_dir=out/"runtime"
    argv=[sys.executable,str(runtime),"--repo-root",str(root),"exec" if against else "run",target,"--check",check,"--run-dir",str(runtime_dir)]
    if timeout is not None: argv += ["--check-timeout",str(timeout)]
    atomic_json(out/"command.json",{"argv":argv,"cwd":str(root)})
    proc=subprocess.run(argv,cwd=str(root),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    (out/"runtime-cli.stdout.log").write_text(proc.stdout,encoding="utf-8"); (out/"runtime-cli.stderr.log").write_text(proc.stderr,encoding="utf-8")
    payload=None
    try: payload=json.loads(proc.stdout)
    except json.JSONDecodeError: pass
    if (runtime_dir/"result.json").is_file():
        try: payload=json.loads((runtime_dir/"result.json").read_text())
        except json.JSONDecodeError: pass
    return {"returncode":proc.returncode,"payload":payload if isinstance(payload,dict) else {},"runtime_dir":str(runtime_dir),"stdout":str(out/"runtime-cli.stdout.log"),"stderr":str(out/"runtime-cli.stderr.log")}


def validation_result(runtime_dir: Path) -> dict[str,Any]:
    path=runtime_dir/"validation/result.json"
    if not path.is_file(): return {"status":"not_provided","path":str(path)}
    try: data=json.loads(path.read_text())
    except Exception as exc: return {"status":"invalid","path":str(path),"error":str(exc)}
    cases=[]
    for i,case in enumerate(data.get("cases",[]) if isinstance(data,dict) else [],1):
        if isinstance(case,dict):
            state=str(case.get("status") or "unknown"); cases.append({"id":str(case.get("id") or f"case-{i}"),"required":bool(case.get("required",True)),"status":state if state in VALID_STATES else "unknown","evidence":case.get("evidence",[])})
    required=[c for c in cases if c["required"]]
    if any(c["status"]=="failed" for c in required): derived="failed"
    elif any(c["status"]=="blocked" for c in required): derived="blocked"
    elif required and any(c["status"]!="passed" for c in required): derived="partial"
    elif cases and any(c["status"] not in {"passed","skipped"} for c in cases): derived="partial"
    elif cases: derived="passed"
    else: derived=str(data.get("status") or "unknown") if isinstance(data,dict) else "unknown"
    return {"status":"parsed","path":str(path),"derived_status":derived,"cases":cases,"summary":data.get("summary") if isinstance(data,dict) else None,"deviations":data.get("deviations",[]) if isinstance(data,dict) else [],"blockers":data.get("blockers",[]) if isinstance(data,dict) else []}

ALIASES={
"duration_s":("duration","benchmark_duration","benchmark_duration_s","duration_s"),"completed_requests":("completed","successful_requests","completed_requests"),"request_throughput_rps":("request_throughput","request_throughput_rps"),"input_throughput_tps":("input_throughput","input_token_throughput","input_throughput_tps"),"output_throughput_tps":("output_throughput","output_token_throughput","output_throughput_tps"),"total_token_throughput_tps":("total_token_throughput","total_throughput","total_token_throughput_tps"),"mean_ttft_ms":("mean_ttft_ms",),"median_ttft_ms":("median_ttft_ms","p50_ttft_ms"),"p90_ttft_ms":("p90_ttft_ms",),"p95_ttft_ms":("p95_ttft_ms",),"p99_ttft_ms":("p99_ttft_ms",),"mean_tpot_ms":("mean_tpot_ms",),"median_tpot_ms":("median_tpot_ms","p50_tpot_ms"),"p90_tpot_ms":("p90_tpot_ms",),"p95_tpot_ms":("p95_tpot_ms",),"p99_tpot_ms":("p99_tpot_ms",),"mean_itl_ms":("mean_itl_ms",),"p99_itl_ms":("p99_itl_ms",),"mean_e2e_latency_ms":("mean_e2e_latency_ms","mean_e2el_ms"),"p99_e2e_latency_ms":("p99_e2e_latency_ms","p99_e2el_ms"),"peak_memory_mb":("peak_memory_mb","peak_device_memory_mb","max_memory_mb")}
WORKLOAD=("backend","model","dataset_name","request_rate","max_concurrency","num_prompts","random_input_len","random_output_len","total_input_tokens","total_output_tokens","concurrency")


def numeric(value: Any) -> int|float|None:
    if isinstance(value,bool): return None
    if isinstance(value,(int,float)): return value
    if isinstance(value,str):
        try: return float(value)
        except ValueError: return None
    if isinstance(value,dict): return numeric(value.get("value"))
    return None


def dict_candidates(value: Any) -> list[dict[str,Any]]:
    if isinstance(value,list): return [x for x in value if isinstance(x,dict)]
    if not isinstance(value,dict): return []
    result=[value]
    for key in ("result","results","metrics","benchmark"):
        nested=value.get(key)
        if isinstance(nested,dict): result.append({**value,**nested})
        elif isinstance(nested,list): result.extend(x for x in nested if isinstance(x,dict))
    return result


def load_document(path: Path) -> Any:
    if path.suffix==".jsonl":
        result=[]
        for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
            try: result.append(json.loads(line))
            except json.JSONDecodeError: pass
        return result
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_metrics(data: Mapping[str,Any]) -> dict[str,Any]:
    merged=dict(data); merged.update(data.get("metrics",{}) if isinstance(data.get("metrics"),dict) else {})
    metrics={}
    for canonical,aliases in ALIASES.items():
        for alias in aliases:
            value=numeric(merged.get(alias))
            if value is not None: metrics[canonical]=value; break
    workload_source={**merged,**(data.get("workload",{}) if isinstance(data.get("workload"),dict) else {})}
    workload={key:workload_source[key] for key in WORKLOAD if workload_source.get(key) is not None}
    target_met=merged.get("target_met"); target_met=target_met if isinstance(target_met,bool) else None
    throughput=any("throughput" in key for key in metrics); latency=any(key.endswith("_ms") for key in metrics)
    return {"metrics":metrics,"workload":workload,"target_met":target_met,"completeness":"complete" if throughput and latency else "partial","raw_status":merged.get("status")}


def benchmark_result(runtime_dir: Path) -> dict[str,Any]:
    directory=runtime_dir/"benchmark"
    if not directory.is_dir(): return {"status":"missing","searched":str(directory)}
    files=sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix in {".json",".jsonl"})
    best=None
    for path in files:
        try: doc=load_document(path)
        except Exception: continue
        for item in dict_candidates(doc):
            normalized=normalize_metrics(item); score=len(normalized["metrics"])
            if best is None or score>best[0]: best=(score,path,normalized)
    if not best or best[0]==0: return {"status":"unrecognized" if files else "missing","candidates":[str(p) for p in files]}
    score,path,normalized=best
    return {"status":"parsed","source":str(path),"recognized_metric_count":score,**normalized,"candidates":[str(p) for p in files]}


def execute(args: argparse.Namespace,root: Path) -> int:
    pre=preconditions(root,args.target,args.stage,args.check,getattr(args,"allow_incomplete",False),getattr(args,"allow_unvalidated",False))
    if pre["status"]!="passed": return emit({"schema_version":1,"stage":args.stage,"target":args.target,"execution_status":"blocked","preconditions":pre},EXIT_BLOCKED)
    try:
        out=stage_dir(root,args.target,args.stage,args.run_dir); inputs=input_fingerprint(root,args.target,args.stage,args.check); runtime=invoke_runtime(root,args.target,args.check,out,args.against_running,args.check_timeout)
    except Exception as exc: return emit({"stage":args.stage,"target":args.target,"execution_status":"blocked","error":f"{type(exc).__name__}: {exc}"},EXIT_BLOCKED)
    payload=runtime["payload"]; runtime_status=str(payload.get("status") or "failed")
    result={"schema_version":1,"stage":args.stage,"target":args.target,"check":args.check,"against_running":args.against_running,"run_dir":str(out),"runtime_dir":runtime["runtime_dir"],"started_at":now_iso(),"finished_at":now_iso(),"preconditions":pre,"inputs":inputs,"execution_status":runtime_status,"runtime":payload,"runtime_cli":{"returncode":runtime["returncode"],"stdout":runtime["stdout"],"stderr":runtime["stderr"]}}
    runtime_dir=Path(runtime["runtime_dir"])
    if args.stage=="validate":
        structured=validation_result(runtime_dir); result["validation_result"]=structured
        hint=runtime_status if runtime_status in {"failed","blocked"} else (structured.get("derived_status") if structured.get("status")=="parsed" else "needs_review")
    else:
        structured=benchmark_result(runtime_dir); result["benchmark_result"]=structured
        if structured.get("status")=="parsed": atomic_json(out/"metrics.json",structured)
        hint=runtime_status if runtime_status in {"failed","blocked"} else ("passed" if structured.get("completeness")=="complete" else "partial")
    result["status_hint"]=hint; atomic_json(out/"stage-result.json",result)
    if hint=="blocked": return emit(result,EXIT_BLOCKED)
    if hint=="failed": return emit(result,EXIT_FAILED)
    return emit(result,EXIT_OK)


def emit(data: Mapping[str,Any],code:int)->int: print(json.dumps(dict(data),ensure_ascii=False,indent=2)); return code


def add_common(p:argparse.ArgumentParser,default:str)->None:
    p.add_argument("target"); p.add_argument("--check",default=default); p.add_argument("--run-dir"); p.add_argument("--against-running",action="store_true"); p.add_argument("--check-timeout",type=float)


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Execute validation or benchmark stage"); p.add_argument("--repo-root"); sub=p.add_subparsers(dest="stage",required=True)
    v=sub.add_parser("validate"); add_common(v,"validate"); v.add_argument("--allow-incomplete",action="store_true")
    b=sub.add_parser("benchmark"); add_common(b,"benchmark"); b.add_argument("--allow-unvalidated",action="store_true")
    return p


def main(argv:Sequence[str]|None=None)->int:
    args=parser().parse_args(argv)
    try: root=discover_root(args.repo_root); parse_target(args.target); return execute(args,root)
    except Exception as exc: return emit({"execution_status":"blocked","error":f"{type(exc).__name__}: {exc}"},EXIT_BLOCKED)

if __name__=="__main__": raise SystemExit(main())
