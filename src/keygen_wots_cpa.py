#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keygen WOTS_PRF compression CPA across six SHA2 SLH-DSA parameter sets.

Each CPA trial is one fixed SK.seed's *key-generation* PRF-address sequence.
No mixing traces across secret seeds; no FORS PRFs; no signer messages.

paper   : independent-stage success (TRUE preceding secret words for each stage)
chained : full seed recovery using only previous CPA-recovered words
both    : evaluate both, with their metrics kept separate

Preserves the exact user's supplied compression model: A'_t, cumulative HW8,
low-byte-first carry-aware beam width 4, SHA-256 final-block words W5..Wlast.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from numpy.lib.format import open_memmap

import signature_cpa_core as paper
from signature_cpa_core import cpa
import keygen_dataset as capture

VERSION = "2026-11-13-keygen-wots-compression-cpa-v1"
ALL_PARAMS = capture.ALL_PARAMS


def native_path(value) -> Path:
    return capture.native_path(value).resolve()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _file_tag(path: Path) -> list:
    st = path.stat()
    return [str(path.resolve()), int(st.st_size), int(st.st_mtime_ns)]


def _code_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys([*preferred, *(k for row in rows for k in row)]))
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v)
                             for k, v in r.items()})
    os.replace(tmp, path)


def _ids(raw: str, available: Sequence[int]) -> list[int]:
    if str(raw).strip().lower() in ("all", "auto", "*"):
        return sorted(set(int(x) for x in available))
    out = sorted(set(int(x.strip()) for x in str(raw).split(",") if x.strip()))
    if any(x < 0 for x in out):
        raise ValueError("seed IDs must be >= 0")
    return out


def _window(raw: str, points: int) -> Optional[slice]:
    if not raw:
        return None
    match = re.fullmatch(r"(\d+):(\d+)", str(raw).strip())
    if not match:
        raise ValueError("--window must be START:STOP")
    left, right = map(int, match.groups())
    if not 0 <= left < right <= int(points):
        raise ValueError(f"Invalid sample window {raw} for {points} points")
    return slice(left, right)


def _load_truth(param_root: Path, p: str) -> dict[int, bytes]:
    manifest = param_root / "shared_secret_seeds.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing experimental seed ground truth: {manifest}")
    meta = capture.read_json(manifest)
    if meta.get("parameter_set") != p or int(meta.get("n", -1)) != cpa.PARAM_N[p]:
        raise ValueError(f"Secret manifest parameter/n mismatch: {manifest}")
    out: dict[int, bytes] = {}
    for i, item in enumerate(meta.get("seeds", [])):
        sid = int(item.get("seed_id", i))
        sk = bytes.fromhex(item["sk_seed_hex"])
        if len(sk) != cpa.PARAM_N[p]:
            raise ValueError(f"Invalid SK.seed length at seed_{sid:03d}")
        if item.get("sha256") and item["sha256"].lower() != hashlib.sha256(sk).hexdigest():
            raise ValueError(f"Seed hash mismatch at seed_{sid:03d}")
        if sid in out:
            raise ValueError(f"Duplicate seed ID in {manifest}")
        out[sid] = sk
    return out


def _inspect_input(seed_dir: Path, p: str, opt: str, sid: int, truth: bytes) -> dict:
    row = capture.read_json(seed_dir / "keygen_capture_manifest.json")
    if row.get("format") != capture.VERSION or row.get("status") != "completed":
        raise ValueError(f"Keygen capture not complete: {seed_dir}")
    spec = row["capture_spec"]
    if (spec.get("parameter_set") != p or spec.get("compiler_opt") != opt
            or int(spec.get("seed_id", -1)) != sid
            or spec.get("sk_seed_sha256") != hashlib.sha256(truth).hexdigest()):
        raise ValueError(f"Capture/secret mismatch at {seed_dir}")
    pub = bytes.fromhex(spec["pk_seed_hex"])
    if len(pub) != cpa.PARAM_N[p]:
        raise ValueError(f"PK.seed length mismatch: {seed_dir}")
    # WOTS keygen address shape, count and exact deterministic sequence are validated.
    count = capture.params_info(p)["wots_prf_calls"]
    points = int(spec["samples"])
    prefix = seed_dir / "capture"
    if not capture._validate_completed(prefix, count, points, p):
        raise ValueError(f"Missing/misaligned keygen PRF capture: {seed_dir}")
    return {"prefix": prefix, "pub": pub, "num_rows": count, "points": points,
            "spec": spec, "seed_dir": seed_dir, "source_stat": [
                _file_tag(Path(str(prefix) + suf)) for suf in (
                    "_traces_random.npy", "_addrs_random.npy", "_keygen_sequence.json",
                    "_capture_config.json")],
            }


def scan_inputs(base: Path, params: Sequence[str], opts: Sequence[str], seed_ids: str):
    """Find *completed* keygen sequences and validate mapping against truth pool."""
    inputs = []
    issues = []
    for p in params:
        param_root = base / f"SLH_DSA_{p}_sha2_keygen_prf_sequence"
        if not param_root.is_dir():
            issues.append({"parameter_set":p, "status":"missing_parameter", "reason":str(param_root)})
            continue
        try:
            truth_pool = _load_truth(param_root, p)
        except Exception as exc:
            issues.append({"parameter_set":p, "status":"invalid_secret_manifest",
                           "reason":f"{type(exc).__name__}: {exc}"})
            continue
        for opt in opts:
            condition = capture.capture_condition_dir(base, p, opt)
            found = [int(m.group(1)) for x in condition.iterdir()
                     if x.is_dir() and (m := re.fullmatch(r"seed_(\d+)", x.name))] if condition.is_dir() else []
            for sid in _ids(seed_ids, found):
                if sid not in truth_pool:
                    issues.append({"parameter_set":p, "compiler_opt":opt, "seed_id":sid,
                                   "status":"missing_truth", "reason":"no SK.seed for this seed ID"})
                    continue
                try:
                    row = _inspect_input(condition / f"seed_{sid:03d}", p, opt, sid, truth_pool[sid])
                    inputs.append({"parameter_set":p, "compiler_opt":opt, "seed_id":sid,
                                   "truth":truth_pool[sid], **row})
                except Exception as exc:
                    issues.append({"parameter_set":p, "compiler_opt":opt, "seed_id":sid,
                                   "status":"invalid_or_incomplete_capture",
                                   "reason":f"{type(exc).__name__}: {exc}"})
    return inputs, issues


def _selection_indices(total: int, limit: int, policy: str, rng_seed: int, p: str, sid: int):
    if limit <= 0 or limit >= total:
        return np.arange(total, dtype=np.int64)
    if limit < 3:
        raise ValueError("At least 3 PRF traces needed for CPA")
    if policy == "uniform":
        return np.linspace(0, total-1, num=limit, dtype=np.int64)
    if policy == "first":
        return np.arange(limit, dtype=np.int64)
    if policy == "random":
        digest = hashlib.sha256(f"{p}|{sid}|{rng_seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return np.sort(rng.choice(total, size=limit, replace=False)).astype(np.int64)
    raise ValueError("Selection must be uniform, random, or first")


def select_capture(item: Mapping[str, Any], output: Path, limit: int,
                   policy: str, rng_seed: int, dry_run: bool):
    """Read-only source; materialize only noncontiguous sampling into *analysis* root."""
    indices = _selection_indices(int(item["num_rows"]), int(limit), policy,
                                 rng_seed, item["parameter_set"], item["seed_id"])
    prefix: Path = item["prefix"]
    first = int(indices[0]); last = int(indices[-1])
    contiguous = bool(np.array_equal(indices, np.arange(first, last+1)))
    meta = {
        "selected_rows":int(indices.size), "all_rows":int(item["num_rows"]),
        "contiguous":contiguous,
        "selection_sha256":hashlib.sha256(indices.tobytes()).hexdigest(),
        "source_stat":item["source_stat"],
    }
    if dry_run:
        return None, meta
    trace_path = str(prefix) + "_traces_random.npy"
    addr_path = str(prefix) + "_addrs_random.npy"
    src_t = np.load(trace_path, mmap_mode="r")
    src_a = np.load(addr_path, mmap_mode="r")
    if contiguous:
        wave = src_t[first:last+1]
        adrs = src_a[first:last+1]
    else:
        tag = _json_hash({"meta":meta, "version":VERSION})[:20]
        dest = (output / "_selected_prf_cache" / item["parameter_set"] /
                item["compiler_opt"] / f"seed_{item['seed_id']:03d}" / tag)
        dest.mkdir(parents=True, exist_ok=True)
        tm_path, am_path = dest / "traces.npy", dest / "addrs.npy"
        cached = False
        if tm_path.is_file() and am_path.is_file():
            try:
                cached = (np.load(tm_path, mmap_mode="r").shape == (len(indices), int(item["points"]))
                          and np.load(am_path, mmap_mode="r").shape == (len(indices), 32))
            except Exception:
                cached = False
        if not cached:
            import shutil
            estimated = int(len(indices)) * (int(item["points"]) * 4 + 32)
            if estimated > .8 * shutil.disk_usage(dest).free:
                raise OSError("Not enough free space for analysis-only noncontiguous subset cache")
            # Cache is never written to the source capture directory.
            t = open_memmap(tm_path, mode="w+", dtype=np.float32,
                            shape=(len(indices), int(item["points"])))
            a = open_memmap(am_path, mode="w+", dtype=np.uint8, shape=(len(indices), 32))
            for start in range(0, len(indices), 32):
                ids = indices[start:start+32]
                t[start:start+len(ids)] = src_t[ids]
                a[start:start+len(ids)] = src_a[ids]
            t.flush(); a.flush()
            del t, a
        wave = np.load(tm_path, mmap_mode="r")
        adrs = np.load(am_path, mmap_mode="r")
        meta["cache_dir"] = str(dest)
    # An additional integrity check before ever invoking the original CPA model.
    if not np.all(adrs[:,9] == 5):
        raise ValueError("Capture contains non-WOTS_PRF compressed addresses")
    cap = cpa.CaptureData(traces=wave, addrs=adrs, meta=None, meta_columns=[],
                          source=str(prefix), group="WOTS_KEYGEN", param_set=item["parameter_set"])
    return cap, meta


def public_variation_diagnostics(cap: "cpa.CaptureData", pk_seed: bytes) -> dict:
    """Public-input diversity before W5. Does not use/peek at any SK.seed."""
    words = cpa.build_finalize_words(cap.addrs, param_set=cap.param_set, sk_seed=None)
    n = words.shape[0]
    state = cpa._broadcast_state(cpa.seeded_state_from_pk_seed(cap.param_set, pk_seed), n)
    for t in range(5):
        state, _, _, _ = cpa._round_arrays(state, words[:, t], int(cpa.SHA256_K[t]))
    base = (state["h"] + cpa.Sigma1_32(state["e"])
            + cpa.Ch(state["e"], state["f"], state["g"])
            + np.uint32(cpa.SHA256_K[5])
            + cpa.Sigma0_32(state["a"])
            + cpa.Maj(state["a"], state["b"], state["c"])).astype(np.uint32)
    return {
        "unique_public_W0_W4":int(np.unique(words[:, :5], axis=0).shape[0]),
        "unique_a_update_W5_base_low8":int(np.unique(base & np.uint32(0xff)).size),
        "unique_a_update_W5_base_low16":int(np.unique(base & np.uint32(0xffff)).size),
        "unique_public_W5_high16":int(np.unique(words[:,5] >> np.uint32(16)).size),
    }


def run_compression_attack(cap: "cpa.CaptureData", *,
                           protocol: str, truth: bytes, pk_seed: bytes,
                           window: Optional[slice], engine: dict,
                           beam: int, candidate_batch: int, verbose: bool = False):
    """Exact original (unmodified) cumulative-HW compression CPA primitives."""
    if protocol == "paper":
        result = paper.compression_cpa_paper_conditional(
            cap, target="a_update", pk_seed=pk_seed, validation_seed=truth,
            trace_slice=window, engine_options=engine,
            beam_width=beam, candidate_batch=candidate_batch, plot_top=0,
            plot_config=None, verbose=verbose)
    elif protocol == "chained":
        slices = {t:window for t in range(5, cpa._compression_last_secret_word(cap.param_set)+1)}
        result = cpa.run_compression_cpa(
            cap, target="a_update", pk_seed=pk_seed, validation_seed=truth,
            trace_slices=slices, engine_options=engine,
            beam_width=beam, candidate_batch=candidate_batch, plot_top=0,
            plot_config=None, verbose=verbose)
    else:
        raise ValueError("Protocol must be paper/chained")
    rows = paper.stage_rows_from_compression(result, cap, truth)
    return result, rows


def aggregate(trials: Sequence[Mapping[str, Any]], stages: Sequence[Mapping[str, Any]],
              bootstrap_reps: int = 1000) -> dict:
    """Independent SK.seed units; one keygen PRF sequence per SK.seed."""
    completed = [r for r in trials if r.get("status") == "completed" and r.get("full_seed_match") is not None]
    cond_keys = ("parameter_set", "compiler_opt", "protocol")
    by_key = defaultdict(list)
    for row in completed:
        by_key[tuple(row[k] for k in cond_keys) + (int(row["seed_id"]),)].append(row)
    by_seed = []
    for key, group in sorted(by_key.items()):
        by_seed.append({
            **dict(zip((*cond_keys,"seed_id"),key)),
            "evaluated_keygen_trials":len(group),
            "successful_keygen_trials":sum(bool(x["full_seed_match"]) for x in group),
            "full_seed_success_rate":sum(bool(x["full_seed_match"]) for x in group) / len(group),
            "mean_correct_stage_fraction":float(np.mean([float(x["stage_success_fraction"])
                                                            for x in group])),
        })
    by_condition = defaultdict(list)
    for r in by_seed:
        by_condition[tuple(r[k] for k in cond_keys)].append(r)
    summary = []
    for key, group in sorted(by_condition.items()):
        rates = np.asarray([r["full_seed_success_rate"] for r in group], dtype=np.float64)
        n_trials = sum(r["evaluated_keygen_trials"] for r in group)
        n_success = sum(r["successful_keygen_trials"] for r in group)
        ci_low, ci_high = None, None
        if len(rates) >= 2 and int(bootstrap_reps) > 0:
            seed_rng = int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8],"little")
            rng = np.random.default_rng(seed_rng)
            idx = rng.integers(0,len(rates),size=(int(bootstrap_reps),len(rates)))
            bootstrap = rates[idx].mean(axis=1)
            ci_low, ci_high = map(float,np.quantile(bootstrap,[.025,.975]))
        summary.append({
            **dict(zip(cond_keys,key)),
            "independent_secret_seeds":len(group),
            "evaluated_keygen_trials":n_trials,
            "successful_keygen_trials":n_success,
            "seed_macro_full_seed_success_rate":float(rates.mean()),
            "trial_weighted_full_seed_success_rate":n_success/n_trials,
            "mean_stage_success_fraction":float(np.mean([r["mean_correct_stage_fraction"] for r in group])),
            "seed_cluster_bootstrap95_low":ci_low,
            "seed_cluster_bootstrap95_high":ci_high,
            "evaluation_note":"paper=conditional assembled; chained=true end-to-end recovery",
        })
    stage_group = defaultdict(list)
    for r in stages:
        if r.get("selected_match") is not None:
            stage_group[tuple(r[k] for k in cond_keys) + (r["stage_key"],)].append(r)
    stage_summary = []
    for key, group in sorted(stage_group.items()):
        per_seed = defaultdict(list)
        for row in group:
            per_seed[int(row["seed_id"])].append(float(bool(row["selected_match"])))
        seed_values = [float(np.mean(v)) for v in per_seed.values()]
        stage_summary.append({
            **dict(zip((*cond_keys,"stage_key"),key)),
            "target_word":group[0]["target_word"],
            "evaluated_seed_count":len(per_seed),
            "evaluated_trials":len(group),
            "successful_trials":sum(bool(x["selected_match"]) for x in group),
            "seed_macro_stage_success_rate":float(np.mean(seed_values)),
            "trial_weighted_stage_success_rate":float(np.mean([bool(x["selected_match"]) for x in group])),
        })
    return {"summary_by_seed":by_seed, "summary_by_condition":summary,
            "stage_success_rates":stage_summary}


def _plots(rows: list[dict], output: Path):
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for opt in sorted({r["compiler_opt"] for r in rows}):
        selected = [r for r in rows if r["compiler_opt"] == opt]
        ps = [p for p in ALL_PARAMS if any(x["parameter_set"] == p for x in selected)]
        x = np.arange(len(ps))
        fig, ax = plt.subplots(figsize=(max(7,len(ps)*1.1),4))
        for i, protocol in enumerate(("paper","chained")):
            vals = [next((100*r["seed_macro_full_seed_success_rate"] for r in selected
                         if r["parameter_set"] == p and r["protocol"] == protocol), math.nan)
                    for p in ps]
            ax.bar(x+(i-.5)*.38, vals, .36, label=protocol)
        ax.set_xticks(x,ps)
        ax.set_ylim(0,105)
        ax.set_ylabel("Full SK.seed recovery success rate (%), seed macro")
        ax.set_xlabel("SLH-DSA SHA2 parameter set")
        ax.set_title(f"WOTS_PRF during keygen / {opt}")
        ax.grid(axis="y",alpha=.2)
        ax.legend()
        fig.tight_layout()
        dest = output / "figures" / f"keygen_wots_full_seed_{opt}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest,dpi=180)
        plt.close(fig)


def parser_create():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--base", required=True, help="Read-only root of keygen PRF capture data")
    a.add_argument("--output", default="./SLH_DSA_Keygen_CPA_Results")
    a.add_argument("--params", default=",".join(ALL_PARAMS))
    a.add_argument("--opts", default="Os")
    a.add_argument("--seed-ids", default="auto")
    a.add_argument("--protocol", choices=("paper","chained","both"), default="both")
    a.add_argument("--max-traces", type=int, default=0, help="0: all keygen PRFs; optional subset")
    a.add_argument("--selection", choices=("uniform","random","first"), default="uniform")
    a.add_argument("--selection-seed", type=int, default=20261113)
    a.add_argument("--window", default="", help="Optional sample indices START:STOP")
    a.add_argument("--backend", choices=("auto","numpy","tensorflow"), default="auto")
    a.add_argument("--point-batch", type=int, default=1024)
    a.add_argument("--candidate-batch", type=int, default=256)
    a.add_argument("--beam", type=int, default=4)
    a.add_argument("--bootstrap-reps", type=int, default=1000)
    a.add_argument("--dry-run",action="store_true")
    a.add_argument("--rerun",action="store_true")
    a.add_argument("--no-plots",action="store_true")
    a.add_argument("--verbose",action="store_true")
    a.add_argument("--strict",action="store_true")
    return a


def main(argv: Optional[Sequence[str]] = None):
    args = parser_create().parse_args(argv)
    paper.configure_paper_cpa()
    if not cpa.self_test_layouts():
        raise AssertionError("Original CPA core layout self-test failed")
    if args.beam < 1 or args.candidate_batch < 1 or args.max_traces < 0:
        raise ValueError("Invalid CPA beam/candidate batch/max traces")
    params = [cpa.normalize_param_set(p.strip()) for p in args.params.split(",") if p.strip()]
    opts = [cpa.normalize_compiler_opt(s.strip()) for s in args.opts.split(",") if s.strip()]
    base = native_path(args.base)
    output = native_path(args.output)
    if not base.is_dir():
        raise FileNotFoundError(f"Keygen capture base folder not found: {base}")
    if output == base or base in output.parents or output in base.parents:
        raise ValueError("Analysis output must be a separate folder outside source captures")
    if args.dry_run and any((output/x).is_file() for x in ("trial_results.csv","summary_by_condition.csv")):
        print("[NOTICE] Dry-run results go to an isolated _preview output folder")
        output = output / "_preview"
    inputs, issues = scan_inputs(base,params,opts,args.seed_ids)
    output.mkdir(parents=True,exist_ok=True)
    _write_csv(output/"input_inventory.csv", [
        {k:r[k] for k in ("parameter_set","compiler_opt","seed_id","num_rows","points")}
        | {"capture_prefix":str(r["prefix"]),"sk_seed_sha256":hashlib.sha256(r["truth"]).hexdigest()}
        for r in inputs], ["parameter_set","compiler_opt","seed_id","num_rows","points","capture_prefix"])
    print(f"Validated {len(inputs)} full keygen datasets, across "
          f"{len(set((r['parameter_set'],r['seed_id']) for r in inputs))} (parameter, seed) pairs.")
    print("Each CPA uses only WOTS_PRF traces from one SK.seed's one keygen sequence.")
    if args.dry_run:
        _write_csv(output/"missing_or_skipped.csv",issues,
                   ["parameter_set","compiler_opt","seed_id","status","reason"])
        result = {"version":VERSION, "mode":"dry-run", "completed_input_keygen_trials":len(inputs),
                  "output_root":str(output),"issues":len(issues),"settings":vars(args)}
        capture.write_json_atomic(output/"run_manifest.json",result)
        return result

    codes = {"core":_code_hash(Path(cpa.__file__)),
             "paper_adapter":_code_hash(Path(paper.__file__)),
             "keygen_runner":_code_hash(Path(__file__))}
    protocols = ("paper","chained") if args.protocol == "both" else (args.protocol,)
    trial_rows: list[dict] = []
    stage_rows: list[dict] = []
    for i, item in enumerate(inputs, 1):
        p,opt,sid = item["parameter_set"], item["compiler_opt"], item["seed_id"]
        print(f"[{i}/{len(inputs)}] {p}/{opt}/seed_{sid:03d}, "
              f"keygen WOTS_PRFs={item['num_rows']:,}, points={item['points']:,}")
        try:
            cap, selection = select_capture(item,output,args.max_traces,args.selection,
                                            args.selection_seed,False)
            assert cap is not None
            window = _window(args.window, cap.traces.shape[1])
            variety = public_variation_diagnostics(cap,item["pub"])
            if variety["unique_a_update_W5_base_low8"] < 2:
                print("  [WARN] W5 low8 public base has no variance; CPA cannot distinguish initial byte")
            engine = {"backend":args.backend,"point_batch":args.point_batch,
                      "cache_traces":False,"fallback_to_numpy":True,"tf_device":None}
            for protocol in protocols:
                dest = output / "analysis" / p / opt / f"seed_{sid:03d}" / protocol
                result_file, stages_file = dest / "result.json", dest / "stages.json"
                base_meta = {
                    "version":VERSION, "parameter_set":p, "compiler_opt":opt,
                    "seed_id":sid, "kind":"wots", "operation":"compression",
                    "protocol":protocol, "source_prefix":str(item["prefix"]),
                    "selected_traces":int(selection["selected_rows"]),
                    "available_traces":int(item["num_rows"]),
                    "trace_window":args.window or "full",
                    "compression_target":"a_update", "cum_hw":True,"part_bits":8,
                    "beam_width":int(args.beam), "paper_true_prior_used":protocol == "paper",
                    "ground_truth_sha256":hashlib.sha256(item["truth"]).hexdigest(),
                    "pk_seed_hex":item["pub"].hex(),
                    **variety,
                }
                fingerprint = _json_hash({
                    **base_meta, "selection":selection,
                    "candidate_batch":args.candidate_batch,
                    "backend":args.backend,"point_batch":args.point_batch,
                    "codes":codes,
                })
                if not args.rerun and result_file.is_file() and stages_file.is_file():
                    old = capture.read_json(result_file)
                    if old.get("status") == "completed" and old.get("fingerprint") == fingerprint:
                        trial_rows.append(old)
                        recs = capture.read_json(stages_file).get("stages",[])
                        stage_rows.extend({**{k:old[k] for k in (
                            "parameter_set","compiler_opt","seed_id","protocol")},**rec} for rec in recs)
                        print(f"  [REUSE] {protocol} match={old.get('full_seed_match')}")
                        continue
                dest.mkdir(parents=True,exist_ok=True)
                started = time.perf_counter()
                metadata = {**base_meta,"status":"running","fingerprint":fingerprint}
                capture.write_json_atomic(dest/"status.json",metadata)
                try:
                    res, records = run_compression_attack(
                        cap,protocol=protocol,truth=item["truth"],pk_seed=item["pub"],
                        window=window,engine=engine,beam=args.beam,
                        candidate_batch=args.candidate_batch,verbose=args.verbose)
                    marks = [bool(x["selected_match"]) for x in records
                             if x.get("role") == "primary" and x.get("selected_match") is not None]
                    metadata.update({
                        "status":"completed", "seconds":round(time.perf_counter()-started,3),
                        "recovered_sk_seed_hex":res.recovered_seed.hex(),
                        "full_seed_match":res.recovered_seed==item["truth"],
                        "conditional_assembled_seed_match":(
                            res.recovered_seed==item["truth"] if protocol=="paper" else None),
                        "strict_chained_recovery_match":(
                            res.recovered_seed==item["truth"] if protocol=="chained" else None),
                        "correct_stages":sum(marks), "evaluated_stages":len(marks),
                        "stage_success_fraction":sum(marks)/len(marks) if marks else None,
                    })
                    capture.write_json_atomic(result_file,metadata)
                    capture.write_json_atomic(stages_file,{"stages":records})
                    trial_rows.append(metadata)
                    stage_rows.extend({**{k:metadata[k] for k in (
                        "parameter_set","compiler_opt","seed_id","protocol")},**x} for x in records)
                    print(f"  [DONE] {protocol}: full_seed_match={metadata['full_seed_match']}, "
                          f"stages={sum(marks)}/{len(marks)}")
                except Exception as exc:
                    metadata.update({"status":"failed","seconds":round(time.perf_counter()-started,3),
                                     "error":f"{type(exc).__name__}: {exc}",
                                     "traceback":traceback.format_exc(limit=10)})
                    capture.write_json_atomic(result_file,metadata)
                    trial_rows.append(metadata)
                    issues.append({"parameter_set":p,"compiler_opt":opt,"seed_id":sid,
                                   "protocol":protocol,"status":"cpa_failed", "reason":metadata["error"]})
                    print(f"  [FAIL] {protocol}: {metadata['error']}")
                    if args.strict:
                        raise
        except Exception as exc:
            issues.append({"parameter_set":p,"compiler_opt":opt,"seed_id":sid,
                           "status":"input_or_precheck_failed", "reason":f"{type(exc).__name__}: {exc}"})
            if args.strict:
                raise
        # Save interim status so a long multi-seed run is auditable.
        _write_csv(output/"trial_results.csv",trial_rows,[
            "parameter_set","compiler_opt","seed_id","protocol","status",
            "selected_traces","full_seed_match","correct_stages","evaluated_stages"])
        _write_csv(output/"stage_results.csv",stage_rows,[
            "parameter_set","compiler_opt","seed_id","protocol","stage_key","selected_match"])

    stats = aggregate(trial_rows,stage_rows,args.bootstrap_reps)
    _write_csv(output/"summary_by_seed.csv",stats["summary_by_seed"],[
        "parameter_set","compiler_opt","seed_id","protocol","full_seed_success_rate"])
    _write_csv(output/"summary_by_condition.csv",stats["summary_by_condition"],[
        "parameter_set","compiler_opt","protocol","independent_secret_seeds",
        "seed_macro_full_seed_success_rate","trial_weighted_full_seed_success_rate"])
    _write_csv(output/"stage_success_rates.csv",stats["stage_success_rates"],[
        "parameter_set","compiler_opt","protocol","stage_key","seed_macro_stage_success_rate"])
    _write_csv(output/"missing_or_skipped.csv",issues,[
        "parameter_set","compiler_opt","seed_id","protocol","status","reason"])
    if not args.no_plots:
        _plots(stats["summary_by_condition"],output)
    summary = {"version":VERSION,"status":"completed", "input_root":str(base),
               "output_root":str(output), "completed_input_keygen_trials":len(inputs),
               "completed_cpa_runs":sum(x.get("status")=="completed" for x in trial_rows),
               "failed_cpa_runs":sum(x.get("status")=="failed" for x in trial_rows),
               "issues":len(issues),"settings":vars(args),
               "paper_semantics":"independent stages with true prior values; not strict full-key recovery",
               "chained_semantics":"previous CPA word(s) only, no true-secret prior",}
    capture.write_json_atomic(output/"run_manifest.json",summary)
    print("\n=== KEYGEN WOTS_PRF CPA ===")
    for row in stats["summary_by_condition"]:
        print(f"  {row['parameter_set']}/{row['compiler_opt']}/{row['protocol']}: "
              f"seed-macro={100*row['seed_macro_full_seed_success_rate']:.1f}% "
              f"({row['successful_keygen_trials']}/{row['evaluated_keygen_trials']} keygen trials)")
    print("Outputs:",output)
    return summary


if __name__ == "__main__":
    main()
