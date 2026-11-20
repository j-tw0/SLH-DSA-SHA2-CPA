#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-signature / multi-key SLH-DSA SHA2 CPA evaluation.

The experimental unit is (parameter, compiler, SK.seed ID, signature ID,
address kind, CPA protocol, leakage path). Every CPA uses ONLY the PRF traces
from ONE complete FIPS205 signing sequence, NEVER combined with another
signature or another key.

The supplied paper CPA core and its prior one-signature/prefix adapter provide
all hypotheses, scoring, beam ranking, true-prior stage handling, etc. This
module changes *only* dataset enumeration, per-signature selection and summary
statistics. Capture files and the secret-seed manifests are opened READ ONLY.

The mean across seed-specific success rates is the primary result; trial-weighted
mean is also shown for transparency. Several signatures under one secret seed
are repeated measurements, not independent secret keys.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.format import open_memmap

import signature_cpa_core as paper
from signature_cpa_core import cpa

VERSION = "2026-11-09-one-signature-multiseed-paper-v1"
ALL_PARAMS = ("128f", "128s", "192f", "192s", "256f", "256s")
REQUIRED_SUFFIXES = (
    "_traces_random.npy", "_addrs_random.npy", "_params_random.npy",
    "_meta_columns.npy", "_signature_sequence_meta.npy",
    "_signature_sequence_meta_columns.npy", "_signature_sequence.json",
    "_capture_config.json",
)


def path_native(value: str | os.PathLike) -> Path:
    return paper.native_path(value).expanduser().resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def file_signature(path: Path) -> Tuple[str, int, int]:
    st = path.stat()
    return str(path.resolve()), st.st_size, st.st_mtime_ns


def csv_save(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys([*columns, *(k for row in rows for k in row)]))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for item in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple))
                             else v for k, v in item.items()})
    os.replace(tmp, path)


def json_save(path: Path, obj: Mapping[str, Any]) -> None:
    paper.json_dump(path, obj)


def json_read(path: Path) -> Dict[str, Any]:
    return paper.read_json(path)


def comma_values(raw: str) -> List[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def selected_ids(raw: str, available: Sequence[int]) -> List[int]:
    if raw.strip().lower() in ("all", "auto", "*"):
        return sorted(set(int(v) for v in available))
    ids = sorted({int(v) for v in comma_values(raw)})
    if any(v < 0 for v in ids):
        raise ValueError("IDs must be >= 0")
    return ids


def locate_condition(base: Path, p: str, opt: str, source: str) -> Optional[Tuple[Path, str]]:
    kinds = ("message", "path") if source == "auto" else (source,)
    for typ in kinds:
        name = f"SLH_DSA_{p}_sha2_{paper.SOURCE_SUBDIR[typ]}"
        candidates = [base / name / opt]
        if base.name.lower() == name.lower():
            candidates.append(base / opt)
        if base.name.lower() == opt.lower() and base.parent.name.lower() == name.lower():
            candidates.append(base)
        if base.name.startswith("seed_") and base.parent.name.lower() == opt.lower():
            candidates.append(base.parent)
        for folder in dict.fromkeys(candidates):
            if folder.is_dir():
                return folder, typ
    return None


def load_truth_pool(param_root: Path, p: str) -> Tuple[Dict[int, bytes], Dict[int, List[str]], Dict[str, Any]]:
    """Reconcile original and extended seed pools, without ever writing to them."""
    manifests = ([param_root / "shared_secret_seeds.json"]
                 + sorted(param_root.glob("shared_secret_seeds_extra_to_*.json")))
    results: Dict[int, bytes] = {}
    provenance: Dict[int, List[str]] = defaultdict(list)
    master_seeds = set()
    fingerprints = []
    valid_files = 0
    for mf in dict.fromkeys(manifests):
        if not mf.is_file():
            continue
        m = json_read(mf)
        if m.get("parameter_set") != p or int(m.get("n", -1)) != cpa.PARAM_N[p]:
            raise ValueError(f"Secret seed manifest parameter/n mismatch: {mf}")
        master_seeds.add(int(m["master_seed"]))
        rows = m.get("seeds", [])
        if len(rows) != int(m.get("num_seeds", -1)):
            raise ValueError(f"Malformed secret seed manifest: {mf}")
        for i, row in enumerate(rows):
            sid = int(row.get("seed_id", i))
            seed = bytes.fromhex(str(row["sk_seed_hex"]))
            if len(seed) != cpa.PARAM_N[p]:
                raise ValueError(f"Wrong secret seed length: {mf}, seed_{sid:03d}")
            digest = hashlib.sha256(seed).hexdigest()
            if row.get("sha256") is not None and str(row["sha256"]).lower() != digest:
                raise ValueError(f"Secret seed SHA256 mismatch: {mf}, seed_{sid:03d}")
            if sid in results and results[sid] != seed:
                raise ValueError(f"Conflicting secret seed ID {sid} among manifests in {param_root}")
            results[sid] = seed
            provenance[sid].append(str(mf))
        fingerprints.append(file_signature(mf))
        valid_files += 1
    if len(master_seeds) > 1:
        raise ValueError(f"Conflicting master seeds in manifests: {param_root}")
    return results, provenance, {
        "manifest_files": valid_files, "master_seeds": sorted(master_seeds),
        "manifest_signatures": fingerprints,
    }


@dataclass(frozen=True)
class CaptureTrial:
    p: str
    opt: str
    seed_id: int
    signature_id: int
    source: str
    seed_dir: Path
    prefix: Path
    rows: int
    points: int
    info: Mapping[str, Any]

    @property
    def entry(self) -> "paper.SignatureEntry":
        return paper.SignatureEntry(self.signature_id, self.prefix, self.rows, self.points, self.source)


def inspect_signature(p: str, opt: str, seed_id: int, sig_id: int, typ: str,
                      seed_dir: Path, manifest: Mapping[str, Any],
                      truth: Optional[bytes]) -> CaptureTrial:
    pref = seed_dir / f"signature_{sig_id:03d}" / "capture"
    missing = [suffix for suffix in REQUIRED_SUFFIXES if not Path(str(pref) + suffix).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing capture files {missing}: {pref}")

    seq = json_read(Path(str(pref) + "_signature_sequence.json"))
    cfg = json_read(Path(str(pref) + "_capture_config.json"))
    if seq.get("format") != paper.EXPECTED_SEQUENCE_FORMAT[typ]:
        raise ValueError(f"Wrong signing sequence format: {pref}")
    if int(seq.get("signature_id", -1)) != sig_id or seq.get("parameter_set") != p:
        raise ValueError(f"Signing metadata/parameter mismatch: {pref}")
    if cfg.get("parameter_set") != p or int(cfg.get("n", -1)) != cpa.PARAM_N[p]:
        raise ValueError(f"Capture parameter/n mismatch: {pref}")
    if str(manifest.get("mode")) != "sha2" or cfg.get("build", {}).get("hash_mode", "sha2") != "sha2":
        raise ValueError(f"Dataset is not SHA2: {pref}")
    captured_seed_id = (cfg.get("build") or {}).get("secret_seed_id")
    if captured_seed_id is not None and int(captured_seed_id) != seed_id:
        raise ValueError(f"Capture metadata seed ID mismatch: {pref}")
    secret_hash = (cfg.get("build") or {}).get("sk_seed_sha256")
    if secret_hash and truth is not None and secret_hash != hashlib.sha256(truth).hexdigest():
        raise ValueError(f"Capture seed SHA256 does not match ground truth: {pref}")
    if seq.get("pk_root_hex") and manifest.get("pk_root_hex") != seq.get("pk_root_hex"):
        raise ValueError(f"Signing root differs from seed manifest: {pref}")
    if seq.get("capture_phases") != manifest.get("capture_phases"):
        raise ValueError(f"Signing capture phase mismatch: {pref}")
    waves = np.load(str(pref) + "_traces_random.npy", mmap_mode="r")
    addr = np.load(str(pref) + "_addrs_random.npy", mmap_mode="r")
    meta = np.load(str(pref) + "_signature_sequence_meta.npy", mmap_mode="r")
    if (waves.ndim != 2 or waves.shape[0] < 3 or waves.shape[1] < 2
            or addr.shape != (waves.shape[0], 32)
            or meta.ndim != 2 or meta.shape[0] != waves.shape[0]
            or int(seq.get("captured_total_calls", -1)) != waves.shape[0]):
        raise ValueError(f"Wave/address/metadata row mismatch: {pref}")
    # Verify that this signature is genuinely complete, not a partially written .npy.
    if not np.issubdtype(waves.dtype, np.floating):
        raise ValueError(f"Unexpected trace dtype: {waves.dtype}")
    cols = [str(v) for v in np.load(str(pref) + "_signature_sequence_meta_columns.npy",
                                        allow_pickle=True).tolist()]
    if any(k not in cols for k in ("signature_id", "kind", "phase_code", "algorithm_call_index")):
        raise ValueError(f"Required signature metadata columns absent: {pref}")
    if not np.all(meta[:, cols.index("signature_id")] == sig_id):
        raise ValueError(f"Signature metadata ID mismatch: {pref}")
    return CaptureTrial(p, opt, seed_id, sig_id, typ, seed_dir, pref,
                        int(waves.shape[0]), int(waves.shape[1]), seq)


def scan_trials(base: Path, params: Sequence[str], opts: Sequence[str],
                seed_raw: str, signature_raw: str, source: str):
    trials: List[CaptureTrial] = []
    issues: List[dict] = []
    truth_by_key: Dict[Tuple[str, int], bytes] = {}
    inventory: List[dict] = []
    manifests_by_param: Dict[str, Any] = {}
    for p in params:
        param_loaded = False
        for opt in opts:
            condition = locate_condition(base, p, opt, source)
            if condition is None:
                issues.append(dict(parameter_set=p, compiler_opt=opt, status="missing_condition",
                                   reason="Capture condition folder not found"))
                continue
            condition_dir, typ = condition
            if not param_loaded:
                secrets, provenance, manifest_meta = load_truth_pool(condition_dir.parent, p)
                manifests_by_param[p] = (secrets, provenance, manifest_meta)
                param_loaded = True
            secrets, provenance, manifest_meta = manifests_by_param[p]
            found_seeds = [int(m.group(1)) for d in condition_dir.iterdir()
                           if d.is_dir() and (m := re.fullmatch(r"seed_(\d+)", d.name))]
            for sid in selected_ids(seed_raw, found_seeds):
                sd = condition_dir / f"seed_{sid:03d}"
                manifest_path = sd / "signature_sequence_capture_manifest.json"
                if not manifest_path.is_file():
                    issues.append(dict(parameter_set=p, compiler_opt=opt, seed_id=sid,
                                       status="missing_seed", reason=str(manifest_path)))
                    continue
                manifest = json_read(manifest_path)
                if manifest.get("format") != paper.EXPECTED_MANIFEST_FORMAT[typ]:
                    issues.append(dict(parameter_set=p, compiler_opt=opt, seed_id=sid,
                                       status="invalid_manifest", reason=str(manifest_path)))
                    continue
                if manifest.get("parameter_set") != p or str(manifest.get("mode")) != "sha2":
                    raise ValueError(f"Invalid parameter/mode in {manifest_path}")
                truth = secrets.get(sid)
                if truth is None:
                    issues.append(dict(parameter_set=p, compiler_opt=opt, seed_id=sid,
                                       status="missing_truth", reason="No truth in original/extra secret seed pools"))
                    continue
                truth_by_key[(p, sid)] = truth
                done = {int(row["signature_id"]): row for row in manifest.get("completed", [])
                        if str(row.get("status", "")).startswith("completed")}
                known_ids = set(range(int(manifest.get("num_signatures", 0))))
                known_ids.update(int(m.group(1)) for d in sd.iterdir() if d.is_dir()
                                 if (m := re.fullmatch(r"signature_(\d+)", d.name)))
                for sig in selected_ids(signature_raw, sorted(known_ids)):
                    if sig not in done:
                        issues.append(dict(parameter_set=p, compiler_opt=opt, seed_id=sid,
                                           signature_id=sig, status="incomplete_signature",
                                           reason="Not marked completed in signature capture manifest"))
                        continue
                    try:
                        trial = inspect_signature(p, opt, sid, sig, typ, sd, manifest, truth)
                        trials.append(trial)
                        inventory.append(dict(parameter_set=p, compiler_opt=opt,
                                              seed_id=sid, signature_id=sig,
                                              capture_type=typ, capture_rows=trial.rows,
                                              sample_points=trial.points,
                                              secret_manifest_files="; ".join(provenance[sid]),
                                              source_prefix=str(trial.prefix)))
                    except Exception as exc:
                        issues.append(dict(parameter_set=p, compiler_opt=opt, seed_id=sid,
                                           signature_id=sig, status="invalid_signature",
                                           reason=f"{type(exc).__name__}: {exc}"))
    return sorted(trials, key=lambda x: (x.p, x.opt, x.seed_id, x.signature_id)), \
        truth_by_key, issues, inventory


def select_capture(trial: CaptureTrial, kind: str, phases: Optional[Sequence[str]],
                   per_signature: int, selection: str, selection_seed: int,
                   output: Path, dry_run: bool):
    """Use zero-copy read-only slices for complete FORS/WOTS ranges.

    Only non-contiguous selection creates a new copy, inside the *analysis*
    directory (never the user's acquisition folder).
    """
    indices, total, phase_counts = paper._select_rows(
        trial.entry, kind, phases, per_signature, selection, selection_seed)
    if indices.size < 3:
        raise ValueError(f"{kind}: fewer than three selected PRF traces")
    selection_hash = hashlib.sha256(indices.tobytes()).hexdigest()[:16]
    selection_meta = {
        "source_prefix": str(trial.prefix), "signature_id": trial.signature_id,
        "seed_id": trial.seed_id, "kind": kind, "selection": selection,
        "selected_rows": len(indices), "available_rows": int(total),
        "phase_counts": phase_counts, "selected_indices_sha256_prefix": selection_hash,
    }
    for suffix in ("_traces_random.npy", "_addrs_random.npy", "_signature_sequence_meta.npy"):
        selection_meta[suffix + "_stat"] = file_signature(Path(str(trial.prefix) + suffix))
    continuous = bool(indices.size and np.all(indices == np.arange(indices[0], indices[0] + indices.size)))
    selection_meta["zero_copy_contiguous_slice"] = continuous
    selection_meta["bytes_if_copy"] = (len(indices) * (trial.points * 4 + 32)) if not continuous else 0
    if dry_run:
        return None, selection_meta
    src_t = np.load(str(trial.prefix) + "_traces_random.npy", mmap_mode="r")
    src_a = np.load(str(trial.prefix) + "_addrs_random.npy", mmap_mode="r")
    if continuous:
        cut = slice(int(indices[0]), int(indices[-1])+1)
        t = src_t[cut]
        a = src_a[cut]
    else:
        # Immutable fingerprinted subset cache is needed only for non-contiguous
        # phase selection or per-signature downsampling.
        hash_tag = hashlib.sha256(json.dumps(selection_meta, sort_keys=True).encode()).hexdigest()[:20]
        cache = output / "_selected_trace_cache" / trial.p / trial.opt / f"seed_{trial.seed_id:03d}" / f"signature_{trial.signature_id:03d}" / kind / hash_tag
        trace_path, addr_path = cache / "traces.npy", cache / "addrs.npy"
        cache.mkdir(parents=True, exist_ok=True)
        cached = trace_path.is_file() and addr_path.is_file()
        if cached:
            try:
                cached = (np.load(trace_path, mmap_mode="r").shape == (len(indices), trial.points)
                          and np.load(addr_path, mmap_mode="r").shape == (len(indices), 32))
            except Exception:
                cached = False
        if not cached:
            free = shutil.disk_usage(cache).free
            estimated = int(selection_meta["bytes_if_copy"])
            if estimated > 0.8 * free:
                raise OSError(f"Selected subset needs {estimated/2**30:.2f} GiB but only "
                              f"{free/2**30:.2f} GiB free. Use auto selection/another output disk")
            tm = open_memmap(trace_path, mode="w+", dtype=np.float32, shape=(len(indices), trial.points))
            am = open_memmap(addr_path, mode="w+", dtype=np.uint8, shape=(len(indices), 32))
            for st in range(0, len(indices), 32):
                ids = indices[st:st+32]
                tm[st:st+len(ids)] = src_t[ids]
                am[st:st+len(ids)] = src_a[ids]
            tm.flush(); am.flush()
            del tm, am
        t = np.load(trace_path, mmap_mode="r")
        a = np.load(addr_path, mmap_mode="r")
        selection_meta["subset_cache"] = str(cache)
    if not np.all(a[:, 9] == paper.KIND_CODE[kind]):
        raise ValueError(f"Selected addresses do not match {kind} SHA2 type")
    cap = cpa.CaptureData(traces=t, addrs=a, meta=None, meta_columns=[],
                          source=str(trial.prefix), group=kind.upper(), param_set=trial.p)
    selection_meta["unique_public_addresses"] = int(np.unique(a[:, :22], axis=0).shape[0])
    return cap, selection_meta


def run_attack(cap, *, kind: str, protocol: str, attack: str, truth: bytes,
               pk_seed: bytes, window: Optional[slice], engine: Mapping[str, Any],
               candidate_batch: int, beam: int, joint_classes: int,
               plot_top: int, plot_config, verbose: bool):
    """Dispatch to the ORIGINAL CPA hypotheses and paper protocol adapters."""
    p = cap.param_set
    if attack == "expansion":
        with paper.correct_joint_256_chaining_for_paper(protocol == "paper"):
            result = cpa.run_expansion_cpa(
                cap, assumed_recovered=None, validation_seed=truth,
                trace_slices={f"w{x}": window for x in range(5, 14)}
                             | {"w9_w16": window, "w9_w18": window, "w5_w13": window},
                engine_options=engine, candidate_batch=candidate_batch,
                beam_width=beam, plot_top=plot_top,
                stage_primary_method="byte",
                stage_chaining=("assume_previous_success" if protocol == "paper" else "recovered"),
                w9_128_primary_source=("direct_w16" if kind == "fors" else "exhaustive_w18"),
                joint_stage1_top_classes=joint_classes, joint_final_top=16,
                verbose=verbose, plot_config=plot_config)
        stages = paper.stage_rows_from_expansion(result)
    elif protocol == "paper":
        result = paper.compression_cpa_paper_conditional(
            cap, target="a_update", pk_seed=pk_seed, validation_seed=truth,
            trace_slice=window, engine_options=engine, beam_width=beam,
            candidate_batch=candidate_batch, plot_top=plot_top,
            plot_config=plot_config, verbose=verbose)
        stages = paper.stage_rows_from_compression(result, cap, truth)
    else:
        slices = {t: window for t in range(5, cpa._compression_last_secret_word(p) + 1)}
        result = cpa.run_compression_cpa(
            cap, target="a_update", pk_seed=pk_seed, validation_seed=truth,
            trace_slices=slices, engine_options=engine, beam_width=beam,
            candidate_batch=candidate_batch, plot_top=plot_top,
            verbose=verbose, plot_config=plot_config)
        stages = paper.stage_rows_from_compression(result, cap, truth)
    return result, stages


def aggregate_results(results: Sequence[Mapping[str, Any]],
                      stage_rows: Sequence[Mapping[str, Any]],
                      bootstrap_reps: int = 1000) -> Dict[str, List[dict]]:
    """Aggregate *separate one-signature* results. No mixing of keys in CPA."""
    completed = [r for r in results if r.get("status") == "completed" and r.get("full_seed_match") is not None]
    keys = ("parameter_set", "compiler_opt", "kind", "protocol", "attack")
    by_seed: Dict[Tuple[Any,...], List[Mapping[str,Any]]] = defaultdict(list)
    for r in completed:
        by_seed[tuple(r[k] for k in keys) + (r["seed_id"],)].append(r)
    seed_rows = []
    for key, rows in sorted(by_seed.items()):
        frac = [float(r["stage_success_fraction"]) for r in rows
                if r.get("stage_success_fraction") is not None]
        seed_rows.append({
            **dict(zip((*keys, "seed_id"), key)),
            "evaluated_signatures": len(rows),
            "successful_signatures": sum(bool(r["full_seed_match"]) for r in rows),
            "signature_success_rate": sum(bool(r["full_seed_match"]) for r in rows) / len(rows),
            "mean_correct_stage_fraction": (float(np.mean(frac)) if frac else None),
            "mean_selected_traces": float(np.mean([int(r["selected_traces"]) for r in rows])),
            "signature_ids": [r["signature_id"] for r in rows],
        })

    condition_groups: Dict[Tuple[Any,...], List[dict]] = defaultdict(list)
    for row in seed_rows:
        condition_groups[tuple(row[k] for k in keys)].append(row)
    cond_rows = []
    for key, groups in sorted(condition_groups.items()):
        seed_rates = np.asarray([r["signature_success_rate"] for r in groups], dtype=float)
        total_trials = sum(r["evaluated_signatures"] for r in groups)
        total_successes = sum(r["successful_signatures"] for r in groups)
        stage_rates = [r["mean_correct_stage_fraction"] for r in groups
                       if r["mean_correct_stage_fraction"] is not None]
        low, high = None, None
        if len(seed_rates) >= 2 and bootstrap_reps > 0:
            # Cluster bootstrap over keys; signatures within a key remain paired.
            fingerprint = hashlib.sha256(repr(key).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(fingerprint[:8], "little"))
            indices = rng.integers(0, len(seed_rates), size=(bootstrap_reps, len(seed_rates)))
            draws = seed_rates[indices].mean(axis=1)
            low, high = (float(np.quantile(draws, .025)), float(np.quantile(draws, .975)))
        cond_rows.append({
            **dict(zip(keys, key)),
            "evaluated_seed_count": len(groups),
            "evaluated_signature_trials": total_trials,
            "successful_signature_trials": total_successes,
            "seed_macro_full_seed_success_rate": float(seed_rates.mean()),
            "trial_weighted_full_seed_success_rate": total_successes / total_trials,
            "seed_macro_stage_success_fraction": float(np.mean(stage_rates)) if stage_rates else None,
            "mean_selected_traces_per_signature": float(np.mean([g["mean_selected_traces"] for g in groups])),
            "seed_macro_success_rate_std": float(seed_rates.std(ddof=1)) if len(groups)>1 else None,
            "seed_cluster_bootstrap_ci95_low": low,
            "seed_cluster_bootstrap_ci95_high": high,
            "independent_secret_seeds": len(groups),
            "notes": "Separate one-signature CPA per trial; cluster bootstrap samples secret seeds",
        })

    # Same signature index across *different keys*, without combining any power
    # traces. Useful for seeing which message/sequence ID succeeds most often.
    by_sig_id: Dict[Tuple[Any, ...], List[Mapping[str,Any]]] = defaultdict(list)
    for r in completed:
        by_sig_id[tuple(r[k] for k in keys) + (r["signature_id"],)].append(r)
    sig_rows = []
    for key, group in sorted(by_sig_id.items()):
        ids = [int(r["seed_id"]) for r in group]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate seed/signature trial detected: {key}")
        fractions = [float(r["stage_success_fraction"]) for r in group
                     if r.get("stage_success_fraction") is not None]
        sig_rows.append({
            **dict(zip((*keys, "signature_id"), key)),
            "evaluated_seed_count": len(ids),
            "successful_seed_count": sum(bool(r["full_seed_match"]) for r in group),
            "full_seed_success_rate_across_seeds": sum(bool(r["full_seed_match"]) for r in group) / len(group),
            "mean_correct_stage_fraction": float(np.mean(fractions)) if fractions else None,
            "mean_selected_traces": float(np.mean([int(r["selected_traces"]) for r in group])),
        })

    by_stage_trial: Dict[Tuple[Any,...], List[Mapping[str, Any]]] = defaultdict(list)
    for r in stage_rows:
        if r.get("role") != "primary" or r.get("selected_match") is None:
            continue
        key = tuple(r.get(k) for k in keys) + (str(r.get("stage_key")),)
        by_stage_trial[key].append(r)
    stage_summary = []
    for key, group in sorted(by_stage_trial.items()):
        seed_stages: Dict[int, List[bool]] = defaultdict(list)
        for row in group:
            seed_stages[int(row["seed_id"])].append(bool(row["selected_match"]))
        seed_stage_means = [sum(vals)/len(vals) for vals in seed_stages.values()]
        stage_summary.append({
            **dict(zip((*keys, "stage_key"), key)),
            "target_word": group[0].get("target_word"),
            "intermediate_word": group[0].get("intermediate_word"),
            "evaluated_seed_count": len(seed_stages),
            "evaluated_signature_trials": len(group),
            "successful_signature_trials": sum(bool(r["selected_match"]) for r in group),
            "seed_macro_stage_success_rate": float(np.mean(seed_stage_means)),
            "trial_weighted_stage_success_rate": sum(bool(r["selected_match"]) for r in group)/len(group),
        })
    return {"summary_by_seed": seed_rows, "summary_by_condition": cond_rows,
            "summary_by_signature": sig_rows, "stage_success_rates": stage_summary}


def save_signature_index_plots(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups: Dict[Tuple[str,str,str,str], List[Mapping[str,Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["parameter_set"], row["compiler_opt"], row["protocol"], row["attack"])].append(row)
    for key, data in groups.items():
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        plotted = False
        for kind in ("fors", "wots"):
            subset = sorted((r for r in data if r["kind"] == kind), key=lambda r:int(r["signature_id"]))
            if not subset:
                continue
            x = [int(r["signature_id"]) for r in subset]
            y = [100*float(r["full_seed_success_rate_across_seeds"]) for r in subset]
            ax.plot(x, y, marker="o", label=kind.upper())
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_ylim(-5, 105)
        ax.set_xlabel("Signature ID (each CPA uses ONE signature only)")
        ax.set_ylabel("Full SK.seed success rate across keys (%)")
        ax.set_title(" / ".join(key))
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        figure_path = output / "figures" / ("by_signature_" + "_".join(key) + ".png")
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)


def save_summary_plots(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for protocol in sorted({r["protocol"] for r in rows}):
        for attack in sorted({r["attack"] for r in rows}):
            subset = [r for r in rows if r["attack"] == attack and r["protocol"] == protocol]
            for opt in sorted({r["compiler_opt"] for r in subset}):
                selected = [r for r in subset if r["compiler_opt"] == opt]
                ps = [p for p in ALL_PARAMS if any(r["parameter_set"] == p for r in selected)]
                if not ps:
                    continue
                lookup = {(r["parameter_set"], r["kind"]): r for r in selected}
                x = np.arange(len(ps))
                fig, ax = plt.subplots(figsize=(max(7, len(ps)*1.1), 4.1))
                for j, kind in enumerate(("fors", "wots")):
                    vals = [float(lookup[(p,kind)]["seed_macro_full_seed_success_rate"])*100
                            if (p,kind) in lookup else np.nan for p in ps]
                    ax.bar(x + (j-.5)*.38, vals, .37, label=kind.upper())
                ax.set_xticks(x, ps)
                ax.set_ylim(0,105)
                ax.set_ylabel("Full SK.seed success rate (%) - mean across seeds")
                ax.set_xlabel("SLH-DSA SHA2 parameter set")
                ax.set_title(f"One-signature CPA | {opt} | {protocol} | {attack}")
                ax.legend()
                ax.grid(axis="y", alpha=.2)
                fig.tight_layout()
                output_path = output / "figures" / f"one_signature_{opt}_{protocol}_{attack}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=180)
                plt.close(fig)


def parser_create() -> argparse.ArgumentParser:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--base", required=True, help="Root of existing captured signatures; always read-only")
    a.add_argument("--output", default="./OneSignature_MultiSeed_CPA_Results")
    a.add_argument("--params", default=",".join(ALL_PARAMS))
    a.add_argument("--opts", default="Os")
    a.add_argument("--source", choices=("message", "path", "auto"), default="message")
    a.add_argument("--seed-ids", default="auto", help="auto: all existing seed_XXX or comma-separated IDs")
    a.add_argument("--signature-ids", default="all", help="all: each completed signature independently; or 0,1,2")
    a.add_argument("--kinds", default="fors,wots")
    a.add_argument("--phases", default="", help="Optional comma-separated FIPS 205 phase filters")
    a.add_argument("--attacks", default="expansion,compression")
    a.add_argument("--protocol", choices=("paper", "chained", "both"), default="paper")
    a.add_argument("--per-signature", type=paper.parse_per_signature, default=0,
                   help="auto/0 = all available PRFs of one kind in each ONE signature")
    a.add_argument("--selection", choices=("uniform", "random", "first"), default="uniform")
    a.add_argument("--selection-seed", type=int, default=20261109)
    a.add_argument("--window", default="", help="Original sample range START:STOP; default whole trace")
    a.add_argument("--backend", choices=("auto", "tensorflow", "numpy"), default="auto")
    a.add_argument("--tf-point-batch", type=int, default=1024)
    a.add_argument("--candidate-batch", type=int, default=256)
    a.add_argument("--beam", type=int, default=4)
    a.add_argument("--joint-classes", type=int, default=4)
    a.add_argument("--pk-seed-hex", default="")
    a.add_argument("--save-stage-plots", action="store_true")
    a.add_argument("--plot-top", type=int, default=4)
    a.add_argument("--no-summary-plots", action="store_true")
    a.add_argument("--bootstrap-reps", type=int, default=1000)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--rerun", action="store_true")
    a.add_argument("--verbose", action="store_true")
    a.add_argument("--strict", action="store_true")
    return a


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parser_create().parse_args(argv)
    paper.configure_paper_cpa()
    if args.beam != 4 or args.joint_classes != 4:
        print("WARNING: beam/joint-classes differ from the paper's value 4")
    if not cpa.self_test_layouts() or not cpa.self_test_joint_w5_w13_split():
        raise AssertionError("CPA core self-tests failed")
    params = [cpa.normalize_param_set(p) for p in comma_values(args.params)]
    opts = [cpa.normalize_compiler_opt(p) for p in comma_values(args.opts)]
    kinds = comma_values(args.kinds)
    attacks = comma_values(args.attacks)
    phases = comma_values(args.phases) or None
    protocols = ("paper", "chained") if args.protocol == "both" else (args.protocol,)
    if any(k not in paper.KIND_CODE for k in kinds):
        raise ValueError("--kinds must be fors,wots")
    if any(k not in ("expansion", "compression") for k in attacks):
        raise ValueError("--attacks must be expansion,compression")
    if phases and any(k not in paper.PHASES for k in phases):
        raise ValueError("Unknown phase")
    if args.beam < 1 or args.joint_classes < 1 or args.candidate_batch < 1:
        raise ValueError("Invalid CPA beam/candidate batch")
    base = path_native(args.base)
    output = path_native(args.output)
    if not base.is_dir():
        raise FileNotFoundError(f"Missing input capture root: {base}")
    # Prevent accidentally placing the analysis directory inside a capture seed.
    if output == base:
        raise ValueError("Use a separate output folder; output cannot equal input")
    # An exploratory CLI dry-run must never erase an existing real CPA summary.
    if args.dry_run and any((output / filename).exists() for filename in
                            ("trial_results.csv", "summary_by_condition.csv")):
        print("[NOTICE] Existing CPA results detected; writing dry-run into _preview instead")
        output = output / "_preview"

    trials, truth_by_key, issues, inventory = scan_trials(
        base, params, opts, args.seed_ids, args.signature_ids, args.source)
    for trial in trials:
        if output == trial.seed_dir or trial.seed_dir in output.parents:
            raise ValueError(f"Analysis output may not be inside capture seed folder: {trial.seed_dir}")
    output.mkdir(parents=True, exist_ok=True)
    csv_save(output / "input_inventory.csv", inventory,
             ["parameter_set", "compiler_opt", "seed_id", "signature_id", "capture_rows", "sample_points"])
    print(f"Input inventory: {len(trials)} complete signature trials across "
          f"{len(set((r.p,r.seed_id) for r in trials))} (parameter, seed) pairs")
    print("Unit of evaluation: ONE complete signing sequence / ONE secret seed. NO pooling across signatures.")
    print("Capture source opened read-only; analysis results written under", output)
    if not trials:
        csv_save(output / "missing_or_skipped.csv", issues,
                 ["parameter_set", "compiler_opt", "seed_id", "signature_id", "status", "reason"])
        json_save(output / "run_manifest.json", {"version": VERSION, "status": "no_usable_data",
                        "input_root": str(base), "issues": issues, "settings": vars(args)})
        if args.strict:
            raise ValueError("No usable signature data, inspect missing_or_skipped.csv")
        return {"status": "no_usable_data", "inventory_count": 0, "output": str(output)}

    if not args.dry_run:
        print("CPA model: cumulative HW8, carry-aware, beam 4, A_update, 256-bit joint class 4")
    core_hash = sha256_file(Path(cpa.__file__))
    adapter_hash = sha256_file(Path(paper.__file__))
    stage_rows = []
    trial_results = []
    for t_index, trial in enumerate(trials, 1):
        truth = truth_by_key[(trial.p, trial.seed_id)]
        pk = bytes.fromhex(args.pk_seed_hex.replace(" ", "")) if args.pk_seed_hex else cpa.get_default_pk_seed(trial.p)
        if len(pk) != cpa.PARAM_N[trial.p]:
            raise ValueError(f"PK.seed length mismatch for {trial.p}")
        print(f"\n[{t_index}/{len(trials)}] {trial.p}/{trial.opt} seed_{trial.seed_id:03d} "
              f"signature_{trial.signature_id:03d} total PRFs={trial.rows:,}", flush=True)
        for kind in kinds:
            try:
                cap, selection = select_capture(
                    trial, kind, phases, args.per_signature, args.selection,
                    args.selection_seed, output, args.dry_run)
            except Exception as exc:
                issues.append(dict(parameter_set=trial.p, compiler_opt=trial.opt,
                                   seed_id=trial.seed_id, signature_id=trial.signature_id,
                                   kind=kind, status="unusable_kind",
                                   reason=f"{type(exc).__name__}: {exc}"))
                if args.strict:
                    raise
                print("  [SKIP]", kind, type(exc).__name__, str(exc)[:200])
                continue
            print(f"  [{kind}] selected {selection['selected_rows']:,}/{selection['available_rows']:,} "
                  f"PRFs; zero-copy={selection['zero_copy_contiguous_slice']}")
            if args.dry_run:
                continue
            assert cap is not None
            window = paper.parse_window(args.window, cap.traces.shape[1])
            unique_addr = int(selection["unique_public_addresses"])
            engine = dict(backend=args.backend, tf_device=None,
                          point_batch=args.tf_point_batch, cache_traces=False,
                          fallback_to_numpy=True)
            for protocol in protocols:
                for attack in attacks:
                    if attack == "expansion" and kind == "wots" and trial.p[:3] in ("192", "256"):
                        issues.append(dict(parameter_set=trial.p, compiler_opt=trial.opt,
                                           seed_id=trial.seed_id, signature_id=trial.signature_id,
                                           kind=kind, protocol=protocol, attack=attack,
                                           status="outside_paper_scope",
                                           reason="192/256 WOTS full message-schedule CPA not claimed in paper"))
                        continue
                    dest = (output / "analysis" / trial.p / trial.opt /
                            f"seed_{trial.seed_id:03d}" / f"signature_{trial.signature_id:03d}" /
                            kind / protocol / attack)
                    summary_path, stages_path = dest / "result.json", dest / "stages.json"
                    meta = {
                        "version": VERSION, "parameter_set": trial.p,
                        "compiler_opt": trial.opt, "seed_id": trial.seed_id,
                        "signature_id": trial.signature_id, "capture_type": trial.source,
                        "kind": kind, "protocol": protocol, "attack": attack,
                        "signatures": 1, "selected_traces": int(selection["selected_rows"]),
                        "available_traces": int(selection["available_rows"]),
                        "unique_public_addresses": unique_addr,
                        "capture_prefix": str(trial.prefix),
                        "trace_window": args.window or "full",
                        "hamming_model": "cumulative_HW8_with_carry",
                        "beam_width": args.beam,
                        "joint_stage1_classes": args.joint_classes,
                        "compression_target": "a_update",
                        "paper_true_prior_used": (protocol == "paper"),
                        "ground_truth_sha256": hashlib.sha256(truth).hexdigest(),
                        "pk_seed_hex": pk.hex(),
                    }
                    fingerprint_spec = {
                        **meta, "selection": selection, "backend": args.backend,
                        "tf_point_batch": args.tf_point_batch,
                        "candidate_batch": args.candidate_batch,
                        "save_stage_plots": args.save_stage_plots,
                        "plot_top": args.plot_top,
                        "core_hash": core_hash, "adapter_hash": adapter_hash,
                    }
                    fingerprint = hashlib.sha256(json.dumps(fingerprint_spec, sort_keys=True).encode()).hexdigest()
                    if not args.rerun and summary_path.is_file() and stages_path.is_file():
                        old = json_read(summary_path)
                        if old.get("status") == "completed" and old.get("analysis_fingerprint") == fingerprint:
                            trial_results.append(old)
                            details = json_read(stages_path).get("stages", [])
                            stage_rows.extend({**{k: old[k] for k in (
                                "parameter_set", "compiler_opt", "seed_id", "signature_id",
                                "kind", "protocol", "attack", "selected_traces")}, **r}
                                              for r in details)
                            print(f"    [REUSE] {protocol}/{attack} match={old['full_seed_match']}")
                            continue
                    start = time.perf_counter()
                    dest.mkdir(parents=True, exist_ok=True)
                    status = {**meta, "status": "running", "analysis_fingerprint": fingerprint}
                    json_save(dest / "status.json", status)
                    plot_cfg = cpa.PlotConfig(
                        root=dest if args.save_stage_plots else None, param_set=trial.p,
                        compiler_opt=trial.opt, address_kind=kind,
                        seed_id=trial.seed_id, save_plots=args.save_stage_plots,
                        show_plots=False, dpi=300, print_saved_paths=False)
                    try:
                        result, details = run_attack(
                            cap, kind=kind, protocol=protocol, attack=attack,
                            truth=truth, pk_seed=pk, window=window,
                            engine=engine, candidate_batch=args.candidate_batch,
                            beam=args.beam, joint_classes=args.joint_classes,
                            plot_top=args.plot_top if args.save_stage_plots else 0,
                            plot_config=plot_cfg, verbose=args.verbose)
                        bool_matches = [bool(r["selected_match"]) for r in details
                                        if r.get("role") == "primary" and r.get("selected_match") is not None]
                        status.update(status="completed", seconds=round(time.perf_counter()-start, 3),
                                      recovered_sk_seed_hex=result.recovered_seed.hex(),
                                      full_seed_match=(result.recovered_seed == truth),
                                      correct_stages=sum(bool_matches), evaluated_stages=len(bool_matches),
                                      stage_success_fraction=sum(bool_matches)/len(bool_matches) if bool_matches else None)
                        json_save(summary_path, status)
                        json_save(stages_path, {"stages": details})
                        trial_results.append(status)
                        stage_rows.extend({**{k: status[k] for k in (
                            "parameter_set", "compiler_opt", "seed_id", "signature_id",
                            "kind", "protocol", "attack", "selected_traces")}, **r}
                                          for r in details)
                        print(f"    [DONE] {protocol}/{attack} "
                              f"full_seed_match={status['full_seed_match']} "
                              f"stages={status['correct_stages']}/{status['evaluated_stages']} "
                              f"t={status['seconds']:.1f}s", flush=True)
                    except Exception as exc:
                        status.update(status="failed", seconds=round(time.perf_counter()-start, 3),
                                      error=f"{type(exc).__name__}: {exc}",
                                      traceback=traceback.format_exc(limit=12))
                        json_save(summary_path, status)
                        trial_results.append(status)
                        issues.append(dict(parameter_set=trial.p, compiler_opt=trial.opt,
                                           seed_id=trial.seed_id, signature_id=trial.signature_id,
                                           kind=kind, protocol=protocol, attack=attack,
                                           status="cpa_failed", reason=status["error"]))
                        print("    [FAIL]", status["error"], flush=True)
                        if args.strict:
                            raise
            del cap
        if not args.dry_run:
            # Persist interim results so long runs can be safely interrupted.
            csv_save(output / "trial_results.csv", trial_results,
                     ["parameter_set", "compiler_opt", "seed_id", "signature_id", "kind",
                      "protocol", "attack", "status", "selected_traces", "correct_stages",
                      "evaluated_stages", "stage_success_fraction", "full_seed_match"])
            csv_save(output / "stage_results.csv", stage_rows,
                     ["parameter_set", "compiler_opt", "seed_id", "signature_id", "kind",
                      "protocol", "attack", "stage_key", "selected_match"])

    summaries = aggregate_results(trial_results, stage_rows, args.bootstrap_reps)
    csv_save(output / "summary_by_seed.csv", summaries["summary_by_seed"],
             ["parameter_set", "compiler_opt", "kind", "protocol", "attack", "seed_id",
              "evaluated_signatures", "successful_signatures", "signature_success_rate"])
    csv_save(output / "summary_by_condition.csv", summaries["summary_by_condition"],
             ["parameter_set", "compiler_opt", "kind", "protocol", "attack",
              "evaluated_seed_count", "evaluated_signature_trials", "successful_signature_trials",
              "seed_macro_full_seed_success_rate", "trial_weighted_full_seed_success_rate",
              "seed_macro_stage_success_fraction"])
    csv_save(output / "summary_by_signature.csv", summaries["summary_by_signature"],
             ["parameter_set", "compiler_opt", "kind", "protocol", "attack", "signature_id",
              "evaluated_seed_count", "successful_seed_count", "full_seed_success_rate_across_seeds"])
    csv_save(output / "stage_success_rates.csv", summaries["stage_success_rates"],
             ["parameter_set", "compiler_opt", "kind", "protocol", "attack", "stage_key",
              "evaluated_seed_count", "evaluated_signature_trials",
              "seed_macro_stage_success_rate", "trial_weighted_stage_success_rate"])
    csv_save(output / "missing_or_skipped.csv", issues,
             ["parameter_set", "compiler_opt", "seed_id", "signature_id", "kind", "protocol",
              "attack", "status", "reason"])
    if not args.no_summary_plots and not args.dry_run:
        save_summary_plots(summaries["summary_by_condition"], output)
        save_signature_index_plots(summaries["summary_by_signature"], output)
    run_summary = {
        "version": VERSION, "input_root": str(base), "output_root": str(output),
        "mode": "dry-run" if args.dry_run else "CPA",
        "completed_input_signature_trials": len(trials),
        "completed_cpa_runs": sum(r.get("status") == "completed" for r in trial_results),
        "failed_cpa_runs": sum(r.get("status") == "failed" for r in trial_results),
        "issues": len(issues), "settings": vars(args),
        "paper_semantics": "paper: true previous word supplied; chained: recovered word propagated",
        "success_metric": "The primary rate is the arithmetic mean of per-secret-seed one-signature success rates",
        "independence_note": "Signatures from the same SK.seed are repeated trials, not independent keys",
    }
    json_save(output / "run_manifest.json", run_summary)
    print("\n=== One-signature CPA summary ===")
    for row in summaries["summary_by_condition"]:
        print(f"{row['parameter_set']}/{row['compiler_opt']}/{row['kind']} "
              f"{row['protocol']}/{row['attack']}  "
              f"seed_mean_SR={100*row['seed_macro_full_seed_success_rate']:.1f}%  "
              f"{row['evaluated_seed_count']} keys, {row['evaluated_signature_trials']} signatures")
    print("Outputs:", output)
    return run_summary


if __name__ == "__main__":
    main()
