"""Paper-aligned independent-stage CPA for random-address PRF traces.

This adapter reuses exactly the shared SHA-256 CPA hypothesis/scoring primitives.
It preserves the paper distinction: true *preceding* words for each separate
stage evaluation; no claim of end-to-end full-key recovery from that protocol.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import signature_cpa_core as paper
from signature_cpa_core import cpa

PARAMS = ("128f","128s","192f","192s","256f","256s")

def _csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    if not keys:
        keys = ["parameter_set","compiler_opt","kind","seed_id","status"]
    with path.open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=keys,extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k:(json.dumps(v) if isinstance(v,(dict,list,tuple)) else v) for k,v in row.items()})

def _split(value):
    return [x.strip() for x in str(value).split(",") if x.strip()]

def _window(raw: str) -> Optional[slice]:
    if not raw:
        return None
    a,b = [int(x) for x in raw.split(":")]
    if a < 0 or b <= a:
        raise ValueError("--window must be START:STOP with START < STOP")
    return slice(a,b)

def argument_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",required=True,help="Root of existing random-address captures; read only")
    p.add_argument("--output",default="./results/random_address")
    p.add_argument("--params",default=",".join(PARAMS))
    p.add_argument("--opts",default="Os")
    p.add_argument("--kinds",default="fors,wots")
    p.add_argument("--seed-ids",default="0,1,2")
    p.add_argument("--attacks",choices=("compression","schedule","both"),default="both")
    p.add_argument("--max-traces",type=int,default=5000)
    p.add_argument("--window",default="")
    p.add_argument("--pk-seed-hex",default="",help="Explicit PK.seed for ONE security level; omit for public demo capture")
    p.add_argument("--backend",choices=("auto","numpy","tensorflow"),default="auto")
    p.add_argument("--point-batch",type=int,default=1024)
    p.add_argument("--candidate-batch",type=int,default=256)
    p.add_argument("--beam",type=int,default=4)
    p.add_argument("--joint-classes",type=int,default=4)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--strict",action="store_true")
    return p

def main(argv: Optional[Sequence[str]]=None):
    args=argument_parser().parse_args(argv)
    paper.configure_paper_cpa()   # cumulative HW8, byte splits, beam-compatible CPA
    if not cpa.self_test_layouts() or not cpa.self_test_joint_w5_w13_split():
        raise RuntimeError("SHA-256 CPA layout tests failed")
    if args.max_traces < 3 or args.beam < 1 or args.joint_classes < 1:
        raise ValueError("At least 3 traces and positive beam/class widths are required")
    params=[cpa.normalize_param_set(x) for x in _split(args.params)]
    opts=[cpa.normalize_compiler_opt(x) for x in _split(args.opts)]
    kinds=[cpa.normalize_address_kind(x) for x in _split(args.kinds)]
    seeds=[int(x) for x in _split(args.seed_ids)]
    base=Path(cpa.windows_to_wsl_path(args.base)).expanduser().resolve()
    output=Path(args.output).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    if output==base or output in base.parents or base in output.parents:
        raise ValueError("Analysis output must be separate from source captures")
    window=_window(args.window)
    if args.pk_seed_hex and len({cpa.PARAM_N[p] for p in params}) != 1:
        raise ValueError("Explicit --pk-seed-hex with multiple seed lengths is ambiguous; run one level at a time")
    if args.pk_seed_hex:
        explicit_pk=bytes.fromhex(args.pk_seed_hex.replace(" ",""))
    else:
        explicit_pk=None
    output.mkdir(parents=True,exist_ok=True)
    inventory=[]; results=[]; stages=[]
    for p in params:
      for opt in opts:
       for kind in kinds:
        for sid in seeds:
            case={"parameter_set":p,"compiler_opt":opt,"kind":kind,"seed_id":sid}
            try:
                capture=cpa.discover_capture_prefix(base,param_set=p,address_kind=kind,
                          compiler_opt=opt,seed_id=sid,min_random_rows=args.max_traces,verbose=False)
                if capture.random_rows < args.max_traces:
                    raise ValueError("Insufficient traces for requested limit")
                truth=cpa.load_seed_from_shared_manifest(base,param_set=p,seed_id=sid)
                if truth is None and not args.dry_run:
                    raise FileNotFoundError("Validation SK.seed manifest missing")
                pk=explicit_pk if explicit_pk is not None else cpa.get_default_pk_seed(p)
                if len(pk)!=cpa.PARAM_N[p]:
                    raise ValueError("PK.seed width mismatch")
                inventory.append({**case,"status":"ready","capture_prefix":capture.prefix,
                                  "available_rows":capture.random_rows,"selected_rows":args.max_traces,
                                  "trace_points":capture.trace_points,
                                  "has_ground_truth":truth is not None})
                if args.dry_run:
                    continue
                cap=cpa.load_capture_legacy(capture.prefix,param_set=p,
                                            group=kind.upper()+"_RANDOM", max_traces=args.max_traces)
                if cap.traces.shape[0]<3:
                    raise ValueError("Insufficient selected traces")
                if window and window.stop > cap.traces.shape[1]:
                    raise ValueError("Window exceeds capture length")
                engine={"backend":args.backend,"point_batch":args.point_batch,
                        "cache_traces":False,"fallback_to_numpy":True}
                assert truth is not None
                attacks=("compression","schedule") if args.attacks=="both" else (args.attacks,)
                for attack in attacks:
                    if attack == "schedule" and kind == "wots" and cpa.security_bits(p)>128:
                        results.append({**case,"attack":attack,"status":"not_evaluated_in_paper"})
                        continue
                    if attack=="compression":
                        result=paper.compression_cpa_paper_conditional(
                             cap,target="a_update",pk_seed=pk,validation_seed=truth,
                             trace_slice=window,engine_options=engine,beam_width=args.beam,
                             candidate_batch=args.candidate_batch,plot_top=0,plot_config=None,
                             verbose=False)
                        stage_rows=paper.stage_rows_from_compression(result,cap,truth)
                    else:
                        method="exhaustive_w18" if kind=="wots" else "direct_w16"
                        with paper.correct_joint_256_chaining_for_paper(True):
                            result=cpa.run_expansion_cpa(
                              cap,assumed_recovered=None,validation_seed=truth,
                              trace_slices={f"w{i}":window for i in range(5,14)} |
                                      {"w9_w16":window,"w9_w18":window,"w5_w13":window},
                              engine_options=engine,candidate_batch=args.candidate_batch,
                              beam_width=args.beam,plot_top=0,
                              stage_primary_method="byte",
                              stage_chaining="assume_previous_success",
                              w9_128_primary_source=method,
                              joint_stage1_top_classes=args.joint_classes,
                              joint_final_top=16,verbose=False,plot_config=None)
                        stage_rows=paper.stage_rows_from_expansion(result)
                    primary=[r for r in stage_rows if r.get("role","primary")=="primary"
                             and r.get("selected_match") is not None]
                    results.append({**case,"attack":attack,"status":"completed",
                      "evaluation":"conditional_true_priors","selected_rows":cap.traces.shape[0],
                      "correct_stages":sum(bool(x["selected_match"]) for x in primary),
                      "evaluated_stages":len(primary),
                      "conditional_assembled_seed_matches":bool(result.recovered_seed==truth)})
                    for stage in stage_rows:
                        stages.append({**case,"attack":attack,**stage})
            except Exception as exc:
                inventory.append({**case,"status":"missing_or_invalid","reason":f"{type(exc).__name__}: {exc}"})
                if args.strict:
                    raise
    _csv(output/"input_inventory.csv",inventory)
    _csv(output/"trial_results.csv",results)
    _csv(output/"stage_results.csv",stages)
    print(f"Inspected {len(inventory)} cases; completed {sum(r.get('status')=='completed' for r in results)} CPA runs")
    print("Conditional stage matches are NOT an end-to-end chained recovery rate.")
    return {"inspected":len(inventory),"completed":sum(r.get('status')=='completed' for r in results),
            "output":str(output),"dry_run":bool(args.dry_run)}

if __name__=="__main__":
    main()
