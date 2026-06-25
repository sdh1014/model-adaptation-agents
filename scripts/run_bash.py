#!/usr/bin/env python3
"""Run a command with persisted argv, cwd, stdout, stderr, timeout and exit code."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def stop(proc: subprocess.Popen[bytes]) -> list[str]:
    actions=[]
    if proc.poll() is not None:return actions
    try: os.killpg(proc.pid,signal.SIGTERM); actions.append("SIGTERM")
    except ProcessLookupError:return actions
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL);actions.append("SIGKILL")
        except ProcessLookupError:pass
        proc.wait()
    return actions


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);p.add_argument("--cwd");p.add_argument("--timeout",type=float);p.add_argument("command",nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command and a.command[0]=="--" else a.command
    if not cmd:p.error("missing command after --")
    out=Path(a.run_dir).resolve();out.mkdir(parents=True,exist_ok=True);cwd=Path(a.cwd).resolve() if a.cwd else Path.cwd();started=time.time()
    (out/"command.json").write_text(json.dumps({"argv":cmd,"cwd":str(cwd),"timeout_seconds":a.timeout},ensure_ascii=False,indent=2),encoding="utf-8")
    timed=False;cleanup=[]
    with (out/"stdout.log").open("wb") as stdout,(out/"stderr.log").open("wb") as stderr:
        try:proc=subprocess.Popen(cmd,cwd=str(cwd),stdout=stdout,stderr=stderr,start_new_session=True)
        except OSError as exc:
            stderr.write((f"{type(exc).__name__}: {exc}\n").encode());rc=127
        else:
            try:rc=proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:timed=True;cleanup=stop(proc);rc=124
            except KeyboardInterrupt:cleanup=stop(proc);rc=130
    result={"exit_code":rc,"passed":rc==0,"timed_out":timed,"cleanup":cleanup,"duration_seconds":round(time.time()-started,3)}
    (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return rc
if __name__=="__main__":raise SystemExit(main())
