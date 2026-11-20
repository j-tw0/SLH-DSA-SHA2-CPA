#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-faithful CPA on FIPS 205 SLH-DSA signing-order PRF trace collections.

This is an analysis adapter, NOT a new leakage/CPA model. It loads the
`signature_XXX/capture_*` files saved by the supplied FIPS 205 (v2) and
random-message FIPS 205 (v3) capture helpers, pools selected PRF invocations
by acquisition order, and calls the user's existing CPA primitives.

Paper baseline ("Correlation Power Analysis on SHA-2-Based SLH-DSA"):
  * max absolute Pearson correlation with cumulative Hamming-weight leakage;
  * 8-bit low-to-high carry-aware splits and beam width 4;
  * SHA-256 compression: updated A (a_update), rounds W5..W_last;
  * SHA-256 message schedule: FIPS 205 W16..W23 dependencies, joint
    W5/W13 procedure for 256, exhaustive W18 alternative for 128 WOTS;
  * independent-stage/paper evaluation can supply *true prior* secret words,
    while strict chained recovery never propagates truth;
  * compression independent-stage behavior is implemented as a narrow adapter
    around exactly the same candidate/scoring primitives as the user's core.

Requires next to this file:
    sha256_prf_cpa_core.py

Example:
  python signature_cpa_core.py \
     --base "/path/to/SLH_DSA/signature_captures" \
     --params 128f,128s,192f,192s,256f,256s --opts Os \
     --signatures 1,2,3,5,10 --per-signature auto --max-total 0 \
     --protocol both --kinds fors,wots --attacks expansion,compression \
     --output ./Signature_CPA_Results

Use --dry-run to inspect captures without executing CPA. A *single* seed's
first successful prefix is not an empirical success rate across secret keys.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.format import open_memmap

try:
    import sha256_prf_cpa_core as cpa
except ImportError as exc:
    raise SystemExit(
        "CPA module not found. Place the original supplied Python core, named\n"
        "sha256_prf_cpa_core.py, alongside this script."
    ) from exc

VERSION = "2026-11-04-fips205-signature-prefix-paper-cpa-v1"
PARAMS = ("128f", "128s", "192f", "192s", "256f", "256s")
PHASES = {
    "fors_signature": (1, 6), "fors_treehash": (2, 6),
    "wots_signature": (3, 5), "xmss_treehash": (4, 5),
}
KIND_CODE = {"fors": 6, "wots": 5}
SOURCE_SUBDIR = {
    "message": "message_signature_sequence", "path": "signature_sequence",
}
EXPECTED_MANIFEST_FORMAT = {
    "message": "slh-dsa-fips205-random-message-signing-prf-capture-v3",
    "path": "slh-dsa-fips205-signing-prf-capture-v2",
}
EXPECTED_SEQUENCE_FORMAT = {
    "message": "slh-dsa-fips205-random-message-signing-prf-address-sequence-v3",
    "path": "slh-dsa-fips205-signing-prf-address-sequence-v2",
}

# FIPS 205 parameter constants used to report theoretical, full-signature
# address-PRF call counts. The captured dataset remains the authoritative source
# of AVAILABLE traces (a capture may intentionally omit some signing phases).
FIPS205_PARAMS = {
    "128f": (22, 3, 6, 33),
    "128s": (7, 9, 12, 14),
    "192f": (22, 3, 8, 33),
    "192s": (7, 9, 14, 17),
    "256f": (17, 4, 9, 35),
    "256s": (8, 8, 14, 22),
}  # param_set -> (d, h_prime, a, k)


def expected_prf_calls_per_signature(param_set: str) -> Dict[str, int]:
    """FIPS 205 reference signing order, no intermediate PRF caching.

    FORS: k * 2**a; WOTS+: d * 2**h_prime * len.
    These are full-signature maxima for the attached capture helpers; they are
    NOT used to fabricate traces that were not saved by a partial capture.
    """
    p = cpa.normalize_param_set(param_set)
    d, h_prime, a, k = FIPS205_PARAMS[p]
    n = cpa.PARAM_N[p]
    len1 = (8 * n) // 4  # log2(w) = 4, w = 16
    v = len1 * 15
    len2 = 0
    while v:
        v //= 16
        len2 += 1
    length = len1 + len2
    out = {
        "fors_signature": k,
        "fors_treehash": k * ((1 << a) - 1),
        "wots_signature": d * length,
        "xmss_treehash": d * ((1 << h_prime) - 1) * length,
        "wots_len": length,
    }
    out["fors_total"] = out["fors_signature"] + out["fors_treehash"]
    out["wots_total"] = out["wots_signature"] + out["xmss_treehash"]
    out["total"] = out["fors_total"] + out["wots_total"]
    return out


def parse_per_signature(value: str | int) -> int:
    """Convert user-friendly `auto` / `all` to the existing 0=all mode."""
    s = str(value).strip().lower()
    if s in ("auto", "all", "full"):
        return 0
    try:
        number = int(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--per-signature must be auto/all/full, 0, or a positive integer"
        ) from exc
    if number < 0:
        raise argparse.ArgumentTypeError("--per-signature cannot be negative")
    return number


def native_path(value: str | os.PathLike) -> Path:
    return Path(cpa.windows_to_wsl_path(value)).expanduser()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(value), f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def split_csv(value: str) -> List[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


@dataclass(frozen=True)
class SignatureEntry:
    signature_id: int
    prefix: Path
    rows: int
    points: int
    capture_type: str


@dataclass(frozen=True)
class Collection:
    param_set: str
    compiler_opt: str
    seed_id: int
    capture_type: str
    seed_dir: Path
    manifest: dict
    entries: Tuple[SignatureEntry, ...]

    @property
    def parameter_dir(self) -> Path:
        return self.seed_dir.parent.parent


def _candidate_seed_dirs(base: Path, p: str, opt: str, seed_id: int,
                         source: str) -> List[Tuple[str, Path]]:
    types = ("message", "path") if source == "auto" else (source,)
    out: List[Tuple[str, Path]] = []
    for typ in types:
        folder = f"SLH_DSA_{p}_sha2_{SOURCE_SUBDIR[typ]}"
        expected = base / folder / opt / f"seed_{seed_id:03d}"
        alternatives = (
            expected,
            base / opt / f"seed_{seed_id:03d}" if base.name.lower() == folder.lower() else expected,
            base / f"seed_{seed_id:03d}" if base.name == opt else expected,
            base if base.name == f"seed_{seed_id:03d}" else expected,
        )
        for candidate in dict.fromkeys(alternatives):
            if (candidate / "signature_sequence_capture_manifest.json").is_file():
                out.append((typ, candidate))
    return out


def discover_collection(base: str | os.PathLike, p: str, opt: str, seed_id: int,
                        source: str = "auto") -> Optional[Collection]:
    p = cpa.normalize_param_set(p)
    opt = cpa.normalize_compiler_opt(opt)
    candidates = _candidate_seed_dirs(native_path(base), p, opt, int(seed_id), source)
    if not candidates:
        return None
    errors: List[str] = []
    for typ, seed_dir in candidates:
        try:
            manifest = read_json(seed_dir / "signature_sequence_capture_manifest.json")
            if manifest.get("format") != EXPECTED_MANIFEST_FORMAT[typ]:
                raise ValueError("unsupported capture manifest version / stale v1 dataset")
            if manifest.get("parameter_set") != p or str(manifest.get("mode")) != "sha2":
                raise ValueError("capture parameter set / hash mode mismatch")
            done = {int(e["signature_id"]): e for e in manifest.get("completed", [])
                    if str(e.get("status", "")).startswith("completed")}
            entries: List[SignatureEntry] = []
            # Require complete, uninterrupted prefixes; an incomplete signature
            # must never silently count as a finished signing sequence.
            for signature_id in range(int(manifest.get("num_signatures", 0))):
                if signature_id not in done:
                    break
                prefix = seed_dir / f"signature_{signature_id:03d}" / "capture"
                required = [
                    Path(str(prefix) + suf) for suf in (
                        "_traces_random.npy", "_addrs_random.npy",
                        "_params_random.npy", "_meta_columns.npy",
                        "_signature_sequence_meta.npy",
                        "_signature_sequence_meta_columns.npy",
                        "_signature_sequence.json", "_capture_config.json",
                    )
                ]
                if not all(v.is_file() for v in required):
                    break
                seq = read_json(Path(str(prefix) + "_signature_sequence.json"))
                if seq.get("format") != EXPECTED_SEQUENCE_FORMAT[typ]:
                    raise ValueError(f"incorrect sequence format at signature {signature_id}")
                if seq.get("signature_id") != signature_id or seq.get("parameter_set") != p:
                    raise ValueError(f"sequence index/parameter mismatch at {prefix}")
                cfg = read_json(Path(str(prefix) + "_capture_config.json"))
                if cfg.get("parameter_set") != p or int(cfg.get("n", 0)) != cpa.PARAM_N[p]:
                    raise ValueError(f"capture config mismatch at {prefix}")
                t = np.load(str(prefix) + "_traces_random.npy", mmap_mode="r")
                a = np.load(str(prefix) + "_addrs_random.npy", mmap_mode="r")
                m = np.load(str(prefix) + "_signature_sequence_meta.npy", mmap_mode="r")
                if (t.ndim != 2 or a.shape != (t.shape[0], 32)
                        or m.ndim != 2 or m.shape[0] != t.shape[0]
                        or int(seq.get("captured_total_calls", -1)) != t.shape[0]
                        or int(done[signature_id].get("captured_calls", -1)) != t.shape[0]):
                    raise ValueError(f"incomplete/misaligned capture: {prefix}")
                if entries and t.shape[1] != entries[0].points:
                    raise ValueError("different trace lengths among signatures of one build")
                entries.append(SignatureEntry(signature_id, prefix, int(t.shape[0]),
                                              int(t.shape[1]), typ))
            if not entries:
                raise ValueError("no complete signature data")
            return Collection(p, opt, seed_id, typ, seed_dir, manifest, tuple(entries))
        except Exception as exc:
            errors.append(f"{seed_dir}: {type(exc).__name__}: {exc}")
    raise ValueError("Found signature manifest(s) but cannot use them:\n" + "\n".join(errors))


def load_ground_truth(collection: Collection, override: Optional[str] = None) -> Optional[bytes]:
    if override:
        seed = bytes.fromhex(override.replace(" ", ""))
    else:
        manifest_path = collection.parameter_dir / "shared_secret_seeds.json"
        if not manifest_path.is_file():
            return None
        manifest = read_json(manifest_path)
        if manifest.get("parameter_set") != collection.param_set:
            raise ValueError("ground-truth manifest parameter mismatch")
        matches = [row for index, row in enumerate(manifest.get("seeds", []))
                   if int(row.get("seed_id", index)) == collection.seed_id]
        if not matches:
            return None
        seed = bytes.fromhex(str(matches[0]["sk_seed_hex"]))
    if len(seed) != cpa.PARAM_N[collection.param_set]:
        raise ValueError("ground-truth SK.seed length mismatch")
    return seed


def _meta_columns(prefix: Path) -> List[str]:
    return [str(v) for v in np.load(str(prefix) + "_signature_sequence_meta_columns.npy",
                                   allow_pickle=True).tolist()]


def _select_rows(entry: SignatureEntry, kind: str, phases: Optional[Sequence[str]],
                 per_signature: int, selection: str, selection_seed: int) -> Tuple[np.ndarray, int, Dict[str, int]]:
    meta = np.load(str(entry.prefix) + "_signature_sequence_meta.npy", mmap_mode="r")
    cols = {v: i for i, v in enumerate(_meta_columns(entry.prefix))}
    expected = ("signature_id", "algorithm_call_index", "phase_code", "kind")
    if any(c not in cols for c in expected):
        raise ValueError(f"missing FIPS 205 metadata columns in {entry.prefix}")
    if np.any(meta[:, cols["signature_id"]] != int(entry.signature_id)):
        raise ValueError("signature ID not constant within per-signature capture")
    order = np.asarray(meta[:, cols["algorithm_call_index"]], dtype=np.int64)
    if np.any(order[1:] <= order[:-1]):
        raise ValueError("non-monotonic FIPS 205 algorithm call indices")
    phase_nums = np.asarray(meta[:, cols["phase_code"]], dtype=np.int64)
    kinds = np.asarray(meta[:, cols["kind"]], dtype=np.int64)
    if np.any((kinds == KIND_CODE["fors"]) & ~np.isin(phase_nums, [1, 2])):
        raise ValueError("FORS record occurs outside FORS phases")
    if np.any((kinds == KIND_CODE["wots"]) & ~np.isin(phase_nums, [3, 4])):
        raise ValueError("WOTS record occurs outside WOTS phases")
    mask = kinds == KIND_CODE[kind]
    if phases:
        codes = {PHASES[n][0] for n in phases}
        mask &= np.isin(phase_nums, list(codes))
    rows = np.flatnonzero(mask)
    n_available = len(rows)
    counts = {phase: int(np.count_nonzero(mask & (phase_nums == code)))
              for phase, (code, _) in PHASES.items()}
    if per_signature > 0 and len(rows) > per_signature:
        if selection == "uniform":
            locations = np.linspace(0, len(rows)-1, per_signature, dtype=np.int64)
        elif selection == "random":
            rng = np.random.default_rng(np.random.SeedSequence([
                int(selection_seed), int(entry.signature_id), KIND_CODE[kind]]))
            locations = np.sort(rng.choice(len(rows), size=per_signature, replace=False))
        elif selection == "first":
            locations = np.arange(per_signature, dtype=np.int64)
        else:
            raise ValueError(f"unknown selection policy: {selection}")
        rows = rows[locations]
    # Preserve chronological acquisition order after subsampling.
    return rows.astype(np.int64, copy=False), n_available, counts


def build_pooled_capture(collection: Collection, kind: str, upto: int,
                         output_root: Path, per_signature: int = 500,
                         max_total: int = 10000, selection: str = "uniform",
                         selection_seed: int = 20261104,
                         phases: Optional[Sequence[str]] = None,
                         copy_batch: int = 16,
                         dry_run: bool = False) -> Tuple[Optional["cpa.CaptureData"], dict]:
    """Materialize ONE chronological .npy memmap reused at all signature checkpoints.

    A fixed subsample is picked independently *inside each signature*, making
    the 1,2,3,... prefix truly nested. No trace is synthesized or averaged.
    """
    if kind not in KIND_CODE:
        raise ValueError(f"unsupported kind: {kind}")
    if upto < 1 or upto > len(collection.entries):
        raise ValueError("requested signature count exceeds completed captures")
    selected = list(collection.entries[:upto])
    keys = []
    rows_per_sig = []
    selected_rows = []
    overall = 0
    points = selected[0].points
    for entry in selected:
        idx, available, phase_counts = _select_rows(
            entry, kind, phases, per_signature, selection, selection_seed)
        if max_total > 0:
            idx = idx[:max(0, max_total-overall)]
        rows_per_sig.append({
            "signature_id": entry.signature_id,
            "available_kind_rows": int(available),
            "selected_rows": int(len(idx)),
            "cumulative_selected_rows": int(overall + len(idx)),
            "available_phase_rows": phase_counts,
            "prefix": str(entry.prefix),
        })
        selected_rows.append(idx)
        overall += len(idx)
        for suffix in ("_traces_random.npy", "_addrs_random.npy",
                       "_signature_sequence_meta.npy", "_signature_sequence.json"):
            f = Path(str(entry.prefix) + suffix)
            st = f.stat()
            keys.append([str(f.resolve()), st.st_size, st.st_mtime_ns])
    if overall < 3:
        raise ValueError(f"{kind} has fewer than 3 selected PRF traces")

    spec = {
        "version": VERSION, "parameter_set": collection.param_set,
        "compiler_opt": collection.compiler_opt, "seed_id": collection.seed_id,
        "capture_type": collection.capture_type, "kind": kind,
        "max_signatures": upto, "per_signature": int(per_signature),
        "max_total": int(max_total), "selection": selection,
        "selection_seed": int(selection_seed), "phases": sorted(phases or []),
        "source_stat": keys,
    }
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = output_root / "_pooled_trace_cache" / collection.param_set / collection.compiler_opt / f"seed_{collection.seed_id:03d}" / kind / fingerprint
    theoretical = expected_prf_calls_per_signature(collection.param_set)
    kind_full_max = theoretical[f"{kind}_total"]
    # Captured phases may be a subset of a full signature. The report records
    # both the full theoretical bound and the phase-filtered expectation.
    available_phases = collection.manifest.get("capture_phases")
    if isinstance(available_phases, list):
        requested_phases = set(phases) if phases else set(PHASES)
        expected_for_captured_phases = sum(
            theoretical[name] for name in PHASES
            if PHASES[name][1] == KIND_CODE[kind]
            and name in available_phases and name in requested_phases
        )
    else:
        expected_for_captured_phases = None
    report = {
        "fips205_max_calls_per_full_signature": theoretical,
        "fips205_max_kind_calls_per_full_signature": int(kind_full_max),
        "expected_selected_kind_calls_for_captured_phases": expected_for_captured_phases,
        "capture_row_counts_match_expected": (
            None if expected_for_captured_phases is None
            else all(row["available_kind_rows"] == expected_for_captured_phases
                     for row in rows_per_sig)
        ),
        "collection": str(collection.seed_dir), "capture_type": collection.capture_type,
        "parameter_set": collection.param_set, "compiler_opt": collection.compiler_opt,
        "seed_id": collection.seed_id, "kind": kind, "signature_count": upto,
        "selected_rows": overall, "raw_points": points,
        "data_bytes_estimate": overall * (points * 4 + 32 + 8),
        "selection": selection, "per_signature": per_signature, "max_total": max_total,
        "rows_per_signature": rows_per_sig, "cache_dir": str(cache_dir),
    }
    if dry_run:
        return None, report
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace_path = cache_dir / "traces.npy"
    addrs_path = cache_dir / "addrs.npy"
    sid_path = cache_dir / "signature_ids.npy"
    metadata_path = cache_dir / "selection.json"
    reusable = False
    if all(p.is_file() for p in (metadata_path, trace_path, addrs_path, sid_path)):
        try:
            old = read_json(metadata_path)
            reusable = (old.get("fingerprint") == fingerprint and old.get("spec") == spec
                        and tuple(np.load(trace_path, mmap_mode="r").shape) == (overall, points)
                        and tuple(np.load(addrs_path, mmap_mode="r").shape) == (overall, 32))
        except Exception:
            reusable = False
    if not reusable:
        estimated = report["data_bytes_estimate"]
        available_disk = shutil.disk_usage(cache_dir).free
        if estimated > 0.8 * available_disk:
            raise OSError(f"Pooled trace memmap needs {estimated/2**30:.2f} GiB, "
                          f"but only {available_disk/2**30:.2f} GiB are free. "
                          "Lower --per-signature, use fewer signatures, or free disk space.")
        t_mm = open_memmap(trace_path, mode="w+", dtype=np.float32,
                           shape=(overall, points))
        a_mm = open_memmap(addrs_path, mode="w+", dtype=np.uint8,
                           shape=(overall, 32))
        s_mm = open_memmap(sid_path, mode="w+", dtype=np.int32,
                           shape=(overall,))
        cursor = 0
        for entry, indices in zip(selected, selected_rows):
            trace_source = np.load(str(entry.prefix) + "_traces_random.npy", mmap_mode="r")
            addr_source = np.load(str(entry.prefix) + "_addrs_random.npy", mmap_mode="r")
            for off in range(0, len(indices), max(1, copy_batch)):
                batch = indices[off:off+max(1, copy_batch)]
                count = len(batch)
                a_batch = np.asarray(addr_source[batch], dtype=np.uint8)
                # ADRS^c byte 9 encodes WOTS_PRF=5 / FORS_PRF=6 in SHA2.
                if not np.all(a_batch[:, 9] == KIND_CODE[kind]):
                    raise ValueError(f"FIPS 205 address type mismatches metadata in {entry.prefix}")
                t_mm[cursor:cursor+count] = np.asarray(trace_source[batch], dtype=np.float32)
                a_mm[cursor:cursor+count] = a_batch
                s_mm[cursor:cursor+count] = entry.signature_id
                cursor += count
            print(f"  [pool {collection.param_set}/{kind}] signature {entry.signature_id}: "
                  f"+{len(indices)} = {cursor}", flush=True)
        assert cursor == overall
        for arr in (t_mm, a_mm, s_mm):
            arr.flush()
        del t_mm, a_mm, s_mm
        json_dump(metadata_path, {"fingerprint": fingerprint, "spec": spec, "report": report})

    tr = np.load(trace_path, mmap_mode="r")
    addrs = np.load(addrs_path, mmap_mode="r")
    cap = cpa.CaptureData(traces=tr, addrs=addrs, meta=None, meta_columns=[],
                          source=str(cache_dir), group=f"{kind.upper()}_RANDOM",
                          param_set=collection.param_set)
    report["reused_cache"] = reusable
    return cap, report


def cap_prefix(cap: "cpa.CaptureData", n: int) -> "cpa.CaptureData":
    if not 3 <= n <= cap.traces.shape[0]:
        raise ValueError("invalid prefix trace count")
    return cpa.CaptureData(traces=cap.traces[:n], addrs=cap.addrs[:n],
                           meta=None, meta_columns=[], source=cap.source,
                           group=cap.group, param_set=cap.param_set)


@contextlib.contextmanager
def correct_joint_256_chaining_for_paper(enabled: bool):
    """Preserve paper's true-prior evaluation after the joint 256-bit stage.

    The supplied core already implements conditional chaining at ordinary
    schedule-word stages. Its 256-bit W5/W13 joint stage instead propagates
    the recovered pair unconditionally. We change ONLY the state propagated
    *after scoring the joint pair*, not its candidates, ranking, predictions,
    recovered result or reported success. This ensures later W6..W8 stages
    receive true prior words in the paper evaluation protocol.
    """
    if not enabled:
        yield
        return
    original = cpa._recover_w5_w13_joint_via_w20

    def paper_joint(ctx: Any, *args: Any, **kwargs: Any):
        recovered = original(ctx, *args, **kwargs)
        if ctx.chaining == "assume_previous_success" and ctx.true_words is not None:
            true_w5 = ctx.true_words[:, 5].astype(np.uint32)
            true_w13 = ctx.true_words[:, 13].astype(np.uint32)
            ctx.work[5] = true_w5.copy()
            cpa._set_constant_word(ctx, 13, int(true_w13[0]))
            ctx.expanded[20] = (
                cpa._word(ctx, 4) + cpa.sigma0_32(ctx.work[5])
                + cpa._word(ctx, 13) + cpa.sigma1_32(cpa._word(ctx, 18))
            ).astype(np.uint32)
            ctx.stage_info["w5_w13_joint"]["paper_true_prior_propagation"] = True
        return recovered

    cpa._recover_w5_w13_joint_via_w20 = paper_joint
    try:
        yield
    finally:
        cpa._recover_w5_w13_joint_via_w20 = original


def compression_cpa_paper_conditional(
    cap: "cpa.CaptureData", *, target: str, pk_seed: bytes,
    validation_seed: bytes, trace_slice: Optional[slice],
    engine_options: Mapping[str, Any], beam_width: int,
    candidate_batch: int, plot_top: int, plot_config: Optional[Any],
    verbose: bool = False,
) -> "cpa.CompressionResult":
    """Exactly the user's compression hypotheses/search, with true prior states.

    Unlike the supplied `run_compression_cpa` (which always chains recovered
    words), paper's per-stage success experiment holds all *preceding* words
    correct. Only the state update between already scored stages changes.
    """
    p = cap.param_set
    if target not in ("t1", "e_update", "a_update"):
        raise ValueError("unknown compression target")
    true_seed = bytes(validation_seed)
    if len(true_seed) != cpa.PARAM_N[p]:
        raise ValueError("paper conditional compression needs a valid truth seed")
    pk = bytes(pk_seed)
    if len(pk) != len(true_seed):
        raise ValueError("PK.seed has wrong length")
    public_words = cpa.build_finalize_words(cap.addrs, param_set=p, sk_seed=None)
    true_words = cpa.build_finalize_words(cap.addrs, param_set=p, sk_seed=true_seed)
    state = cpa._broadcast_state(cpa.seeded_state_from_pk_seed(p, pk), len(cap.addrs))
    for t in range(5):
        state, _, _, _ = cpa._round_arrays(state, public_words[:, t], int(cpa.SHA256_K[t]))
    last = cpa._compression_last_secret_word(p)
    result_words: Dict[int, Any] = {}
    fn = (cpa.recover_add_word_with_known_parts_bytewise if cpa.COMPRESSION_PART_BITS == 8
          else cpa.recover_add_word_with_known_halfwords)
    for t in range(5, last+1):
        traces, offset = cpa._stage_traces(cap, trace_slice)
        base_t1 = (
            state["h"] + cpa.Sigma1_32(state["e"])
            + cpa.Ch(state["e"], state["f"], state["g"])
            + np.uint32(cpa.SHA256_K[t])
        ).astype(np.uint32)
        t2 = (cpa.Sigma0_32(state["a"])
              + cpa.Maj(state["a"], state["b"], state["c"])).astype(np.uint32)
        if target == "t1":
            add_base = base_t1
        elif target == "e_update":
            add_base = (state["d"] + base_t1).astype(np.uint32)
        else:
            add_base = (base_t1 + t2).astype(np.uint32)
        known = cpa._known_word_halfwords_for_compression(t, last, public_words)
        recovered_array, candidates = fn(
            traces, add_base, known_halfwords=known,
            true_word_array=true_words[:, t].astype(np.uint32),
            engine_options=engine_options,
            target_name=f"F32_{p}_{target}_W{t}",
            beam_width=beam_width, candidate_batch=candidate_batch,
            plot_top=plot_top, x_offset=offset, verbose=verbose,
            plot_config=plot_config,
            plot_attack=f"compression_{target}", plot_stage=f"W{t}",
        )
        result_words[t] = cpa.CompressionWordResult(
            round_index=t, word_array=recovered_array,
            representative_word=int(recovered_array[0]), candidates=candidates)
        state, _, _, _ = cpa._round_arrays(state, true_words[:, t], int(cpa.SHA256_K[t]))
        if verbose:
            succeeded = bool(np.array_equal(recovered_array, true_words[:, t]))
            print(f"[paper conditional] W{t} match={succeeded}")
    recovered_seed = cpa._assemble_compression_seed(p, result_words)
    return cpa.CompressionResult(p, target, recovered_seed, true_seed, result_words)


def configure_paper_cpa() -> None:
    cpa.CUMULATIVE_HW = True
    cpa.DUAL_METHOD_MODE = "selected"
    cpa.COMPRESSION_PART_BITS = 8


def signature_checkpoints(value: str, available: int) -> List[int]:
    if value.strip().lower() == "all":
        return list(range(1, available + 1))
    if value.strip().lower() == "powers":
        result = [1]
        while result[-1] * 2 <= available:
            result.append(result[-1] * 2)
        if available not in result:
            result.append(available)
        return result
    candidates = sorted({int(v) for v in split_csv(value)})
    found = [x for x in candidates if 0 < x <= available]
    if not found:
        raise ValueError(f"No requested checkpoints <= {available}; got {candidates}")
    return found


def stage_rows_from_expansion(res: "cpa.ExpansionResult") -> List[dict]:
    keep = ("stage_key", "target_word", "intermediate_word", "stage_domain",
            "role", "selected_method", "selected_recovered", "true_word",
            "selected_match", "bytewise_match", "two_byte_match",
            "stage1_true_class_retained", "stage1_true_class_rank",
            "final_true_pair_rank")
    return [{k: rec.get(k) for k in keep} for rec in res.stage_comparisons]


def stage_rows_from_compression(res: "cpa.CompressionResult",
                                cap: "cpa.CaptureData", truth: Optional[bytes]) -> List[dict]:
    expected_words = None if truth is None else cpa.build_finalize_words(
        cap.addrs, param_set=cap.param_set, sk_seed=truth)
    output = []
    for t, item in sorted(res.word_results.items()):
        matched = None if expected_words is None else bool(np.array_equal(
            np.asarray(item.word_array, dtype=np.uint32), expected_words[:,t]))
        output.append({"stage_key": f"W{t}@A'_t", "target_word": f"W{t}",
                       "intermediate_word": "A_update", "stage_domain": "compression_word",
                       "role": "primary", "selected_method": "cumulative_hw8",
                       "selected_recovered": f"0x{item.representative_word:08X}",
                       "true_word": (None if expected_words is None
                                     else f"0x{int(expected_words[0,t]):08X}"),
                       "selected_match": matched})
    return output


def plot_signature_summary(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups: Dict[Tuple[str,str,str,str,str], List[Mapping[str,Any]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        key = tuple(str(row[k]) for k in ("parameter_set", "compiler_opt", "kind", "protocol", "attack"))
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        ordered = sorted(group, key=lambda z: int(z["signatures"]))
        x = [int(v["signatures"]) for v in ordered]
        y = [float(v["stage_success_fraction"]) if v.get("stage_success_fraction") is not None else np.nan
             for v in ordered]
        labels = [int(v["selected_traces"]) for v in ordered]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(x, y, marker="o", linewidth=1)
        ax.set_xlabel("Cumulative complete signatures")
        ax.set_ylabel("Correct CPA stages / evaluated stages")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x)
        for xx, yy, n in zip(x, y, labels):
            if np.isfinite(yy):
                ax.annotate(f"{n:,} traces", (xx, yy), xytext=(3, 7),
                            textcoords="offset points", fontsize=7)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        dest = output / "figures" / ("_".join(key) + "_stages.png")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=200)
        plt.close(fig)


def summarize_observed_thresholds(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    """First *observed* success among tested signature prefixes (not an SR)."""
    grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        key = tuple(str(row.get(k)) for k in (
            "parameter_set", "compiler_opt", "seed_id", "kind", "protocol", "attack"))
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda x: int(x["signatures"]))
        matching = [r for r in ordered if r.get("full_seed_match") is True]
        all_stages = [r for r in ordered if r.get("evaluated_stages", 0) > 0
                      and r.get("correct_stages") == r.get("evaluated_stages")]
        first = matching[0] if matching else None
        stage = all_stages[0] if all_stages else None
        stable = next((r for i, r in enumerate(ordered)
                       if all(t.get("full_seed_match") is True for t in ordered[i:])), None)
        output.append({
            **dict(zip(("parameter_set", "compiler_opt", "seed_id", "kind", "protocol", "attack"), key)),
            "evaluated_checkpoints": [int(r["signatures"]) for r in ordered],
            "first_observed_full_seed_match_signatures": first["signatures"] if first else None,
            "first_observed_full_seed_match_selected_traces": first["selected_traces"] if first else None,
            "first_observed_all_stages_match_signatures": stage["signatures"] if stage else None,
            "first_sustained_match_through_last_tested_checkpoint": (
                stable["signatures"] if stable else None),
            "last_evaluated_signature_count": ordered[-1]["signatures"],
            "last_checkpoint_full_seed_match": ordered[-1].get("full_seed_match"),
            "note": "Observed among the tested prefixes of ONE experimental key; not an estimated population success rate.",
        })
    return output


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Signature capture results root, or specific condition root")
    ap.add_argument("--output", default="./Signature_Paper_CPA_Results")
    ap.add_argument("--source", choices=("auto", "message", "path"), default="auto")
    ap.add_argument("--params", default=",".join(PARAMS))
    ap.add_argument("--opts", default="Os")
    ap.add_argument("--seed-id", type=int, default=0)
    ap.add_argument("--signatures", default="1,2,3,5,10", help="Comma-separated checkpoints, all, powers")
    ap.add_argument("--kinds", default="fors,wots")
    ap.add_argument("--phases", default="", help="Optional phase subset, comma-separated")
    ap.add_argument("--protocol", choices=("paper", "chained", "both"), default="both")
    ap.add_argument("--attacks", default="expansion,compression")
    ap.add_argument("--per-signature", type=parse_per_signature, default=500,
                    help="Max PRF traces of EACH kind per complete signature. "
                         "auto/all/0 = all captured traces (parameter-dependent)")
    ap.add_argument("--max-total", type=int, default=10000,
                    help="Max trace rows of each kind across signatures; 0 = unbounded")
    ap.add_argument("--selection", choices=("uniform", "random", "first"), default="uniform")
    ap.add_argument("--selection-seed", type=int, default=20261104)
    ap.add_argument("--window", default="", help="Optional original sample range: START:STOP; default full")
    ap.add_argument("--backend", choices=("auto", "tensorflow", "numpy"), default="auto")
    ap.add_argument("--tf-point-batch", type=int, default=1024)
    ap.add_argument("--candidate-batch", type=int, default=256)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--joint-classes", type=int, default=4,
                    help="Paper B_class=4 for 256-bit joint W5/W13")
    ap.add_argument("--compression-target", choices=("a_update", "e_update", "t1"), default="a_update")
    ap.add_argument("--pk-seed-hex", default="", help="Override captured PK.seed if not the supplied default")
    ap.add_argument("--sk-seed-hex", default="", help="Override experimental truth seed when manifest is absent")
    ap.add_argument("--save-plots", action="store_true", help="Save per-stage CPA curves")
    ap.add_argument("--plot-top", type=int, default=4)
    ap.add_argument("--no-summary-plots", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Check inputs and sample counts only")
    ap.add_argument("--rerun", action="store_true", help="Ignore completed checkpoint result JSON")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Fail instead of skipping missing parameters")
    return ap


def parse_window(raw: str, points: int) -> Optional[slice]:
    if not raw:
        return None
    m = re.fullmatch(r"(\d+):(\d+)", raw.strip())
    if not m:
        raise ValueError("--window must use START:STOP, e.g., 2000:10000")
    start, stop = map(int, m.groups())
    if not 0 <= start < stop <= points:
        raise ValueError(f"window {start}:{stop} invalid for {points} points")
    return slice(start, stop)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_argparser().parse_args(argv)
    configure_paper_cpa()
    if args.beam != 4 or args.joint_classes != 4 or args.compression_target != "a_update":
        print("WARNING: non-paper baseline CPA parameter override was requested")
    if not cpa.self_test_layouts() or not cpa.self_test_joint_w5_w13_split():
        raise AssertionError("original CPA core self-test failed")
    params = [cpa.normalize_param_set(x) for x in split_csv(args.params)]
    opts = [cpa.normalize_compiler_opt(x) for x in split_csv(args.opts)]
    kinds = split_csv(args.kinds)
    attacks = split_csv(args.attacks)
    phases = split_csv(args.phases) or None
    if any(x not in KIND_CODE for x in kinds):
        raise ValueError("--kinds supports fors,wots")
    if phases and any(x not in PHASES for x in phases):
        raise ValueError("unknown phase in --phases")
    if any(x not in ("expansion", "compression") for x in attacks):
        raise ValueError("unknown attack name")
    if args.per_signature < 0 or args.max_total < 0 or args.beam < 1 or args.joint_classes < 1:
        raise ValueError("sampling / beam widths must be nonnegative/positive")
    protocols = ("paper", "chained") if args.protocol == "both" else (args.protocol,)
    output = native_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []
    stage_rows: List[dict] = []
    missing: List[str] = []
    for p in params:
        for opt in opts:
            try:
                collection = discover_collection(args.base, p, opt, args.seed_id, args.source)
                if collection is None:
                    missing.append(f"{p}/{opt}: no completed signature manifest")
                    continue
                checkpoints = signature_checkpoints(args.signatures, len(collection.entries))
                max_signatures = checkpoints[-1]
                truth = load_ground_truth(collection, args.sk_seed_hex or None)
                if truth is None and "paper" in protocols and not args.dry_run:
                    raise ValueError("paper/independent-stage evaluation requires experimental SK.seed "
                                     "from shared_secret_seeds.json or --sk-seed-hex")
                pk = (bytes.fromhex(args.pk_seed_hex.replace(" ", "")) if args.pk_seed_hex
                      else cpa.get_default_pk_seed(p))
                if len(pk) != cpa.PARAM_N[p]:
                    raise ValueError("PK.seed has wrong length")
                print(f"\n=== {p}/{opt} | {collection.capture_type} | "
                      f"{len(collection.entries)} complete signatures | "
                      f"truth={'present' if truth else 'unknown'} ===", flush=True)
                for kind in kinds:
                    try:
                        pooled, report = build_pooled_capture(
                            collection, kind, max_signatures, output,
                            per_signature=args.per_signature, max_total=args.max_total,
                            selection=args.selection, selection_seed=args.selection_seed,
                            phases=phases, dry_run=args.dry_run)
                        print(f"{kind}: {report['selected_rows']:,} selected PRF traces "
                              f"(FIPS 205 full max: "
                              f"{report['fips205_max_kind_calls_per_full_signature']:,}/signature), "
                              f"{report['raw_points']:,} points each; "
                              f"~{report['data_bytes_estimate']/2**30:.2f} GiB pooled memmap")
                        if report["capture_row_counts_match_expected"] is False:
                            print("  [notice] Captured calls differ from the phase-filtered "
                                  "FIPS 205 expected count; verify capture_phases and metadata.")
                        report_dest = output / "discovery" / p / opt / f"seed_{args.seed_id:03d}" / kind
                        json_dump(report_dest / "selection_report.json", report)
                        if args.dry_run:
                            continue
                        assert pooled is not None
                        window = parse_window(args.window, pooled.traces.shape[1])
                        engine = {
                            "backend": args.backend, "tf_device": None,
                            "point_batch": int(args.tf_point_batch),
                            "cache_traces": False, "fallback_to_numpy": True,
                        }
                        for n_sig in checkpoints:
                            n_trace = int(report["rows_per_signature"][n_sig - 1]["cumulative_selected_rows"])
                            if n_trace < 3:
                                print(f"  [skip] {n_sig} signatures: fewer than 3 selected traces")
                                continue
                            cap = cap_prefix(pooled, n_trace)
                            distinct = int(np.unique(cap.addrs[:, :22], axis=0).shape[0])
                            contributing = sum(
                                z["selected_rows"] > 0 for z in report["rows_per_signature"][:n_sig])
                            if contributing < n_sig:
                                print(f"  [budget warning] {n_sig} completed signatures but only "
                                      f"{contributing} supplied selected traces. Increase --max-total.")
                            for protocol in protocols:
                                for attack in attacks:
                                    if attack == "expansion" and kind == "wots" and p[:3] in ("192", "256"):
                                        # Paper explicitly does not claim a complete
                                        # WOTS message-schedule attack for 192/256.
                                        print(f"  [not in paper] {p}/{kind} message-schedule full recovery")
                                        continue
                                    dest = (output / "analysis" / p / opt / f"seed_{args.seed_id:03d}"
                                            / kind / protocol / f"signatures_{n_sig:03d}" / attack)
                                    summary_path = dest / "result.json"
                                    analysis_spec = {
                                        "version": VERSION, "attack": attack, "protocol": protocol,
                                        "kind": kind, "n_signatures": n_sig,
                                        "source_cache": report["cache_dir"],
                                        "n_traces": n_trace, "window": args.window,
                                        "backend": args.backend, "tf_point_batch": args.tf_point_batch,
                                        "candidate_batch": args.candidate_batch,
                                        "beam": args.beam, "joint_classes": args.joint_classes,
                                        "compression_target": args.compression_target,
                                        "save_plots": args.save_plots, "plot_top": args.plot_top,
                                        "truth_digest": (hashlib.sha256(truth).hexdigest() if truth else None),
                                        "pk_seed_hex": pk.hex(),
                                    }
                                    analysis_fingerprint = hashlib.sha256(
                                        json.dumps(analysis_spec, sort_keys=True).encode()).hexdigest()
                                    if summary_path.is_file() and not args.rerun:
                                        old = read_json(summary_path)
                                        if (old.get("status") == "completed"
                                            and old.get("analysis_fingerprint") == analysis_fingerprint
                                            and old.get("version") == VERSION):
                                            print(f"  [reuse] {p}/{kind}/{protocol}/{attack} "
                                                  f"signatures={n_sig}")
                                            results.append(old)
                                            previous_stages = dest / "stages.json"
                                            if previous_stages.is_file():
                                                old_details = read_json(previous_stages).get("stages", [])
                                                for record in old_details:
                                                    stage_rows.append({**{k: old[k] for k in (
                                                        "parameter_set", "compiler_opt", "kind", "protocol",
                                                        "attack", "signatures", "selected_traces")}, **record})
                                            continue
                                    dest.mkdir(parents=True, exist_ok=True)
                                    plotcfg = cpa.PlotConfig(
                                        root=dest if args.save_plots else None,
                                        param_set=p, compiler_opt=opt, address_kind=kind,
                                        seed_id=args.seed_id, save_plots=args.save_plots,
                                        show_plots=False, dpi=300, print_saved_paths=False)
                                    start = time.perf_counter()
                                    status = {"version": VERSION, "status": "running",
                                              "parameter_set": p, "compiler_opt": opt,
                                              "seed_id": args.seed_id, "kind": kind,
                                              "capture_type": collection.capture_type,
                                              "protocol": protocol, "attack": attack,
                                              "signatures": n_sig, "selected_traces": n_trace,
                                              "available_traces_through_signature": int(sum(
                                                  z["available_kind_rows"] for z in
                                                  report["rows_per_signature"][:n_sig])),
                                              "unique_public_addresses": distinct,
                                              "source_cache": report["cache_dir"],
                                              "analysis_fingerprint": analysis_fingerprint,
                                              "trace_window": args.window or "full",
                                              "signatures_contributing_traces": int(sum(
                                                  z["selected_rows"] > 0 for z in
                                                  report["rows_per_signature"][:n_sig])),
                                              "hamming_model": "cumulative_HW8_with_carry",
                                              "compression_target": args.compression_target,
                                              "beam_width": args.beam,
                                              "joint_stage1_classes": args.joint_classes,
                                              "sample_policy": args.selection,
                                              "per_signature": args.per_signature,
                                              "max_total": args.max_total,
                                              "ground_truth_available": truth is not None,
                                              "ground_truth_sha256": (hashlib.sha256(truth).hexdigest()
                                                                      if truth else None),
                                              "pk_seed_hex": pk.hex(),
                                              }
                                    json_dump(dest / "status.json", status)
                                    print(f"  [CPA] {p}/{opt}/{kind} {protocol}/{attack} "
                                          f"Nsign={n_sig} Ntraces={n_trace} Nunique={distinct}",
                                          flush=True)
                                    try:
                                        if attack == "expansion":
                                            with correct_joint_256_chaining_for_paper(protocol == "paper"):
                                                res = cpa.run_expansion_cpa(
                                                    cap, assumed_recovered=None, validation_seed=truth,
                                                    trace_slices={f"w{x}":window for x in range(5, 14)}
                                                    | {"w9_w16": window, "w9_w18": window,
                                                       "w5_w13": window},
                                                    engine_options=engine, candidate_batch=args.candidate_batch,
                                                    beam_width=args.beam, plot_top=(args.plot_top if args.save_plots else 0),
                                                    stage_primary_method="byte",
                                                    stage_chaining=("assume_previous_success" if protocol == "paper"
                                                                    else "recovered"),
                                                    w9_128_primary_source=("direct_w16" if kind == "fors"
                                                                           else "exhaustive_w18"),
                                                    joint_stage1_top_classes=args.joint_classes,
                                                    joint_final_top=16,
                                                    verbose=args.verbose, plot_config=plotcfg)
                                            details = stage_rows_from_expansion(res)
                                        elif protocol == "paper":
                                            assert truth is not None
                                            res = compression_cpa_paper_conditional(
                                                cap, target=args.compression_target, pk_seed=pk,
                                                validation_seed=truth, trace_slice=window,
                                                engine_options=engine, beam_width=args.beam,
                                                candidate_batch=args.candidate_batch,
                                                plot_top=(args.plot_top if args.save_plots else 0),
                                                plot_config=plotcfg, verbose=args.verbose)
                                            details = stage_rows_from_compression(res, cap, truth)
                                        else:
                                            slices = {t: window for t in range(5, cpa._compression_last_secret_word(p)+1)}
                                            res = cpa.run_compression_cpa(
                                                cap, target=args.compression_target, pk_seed=pk,
                                                validation_seed=truth, trace_slices=slices,
                                                engine_options=engine, beam_width=args.beam,
                                                candidate_batch=args.candidate_batch,
                                                plot_top=(args.plot_top if args.save_plots else 0),
                                                verbose=args.verbose, plot_config=plotcfg)
                                            details = stage_rows_from_compression(res, cap, truth)
                                        checked = [r["selected_match"] for r in details
                                                   if r.get("role") == "primary"
                                                   and r.get("selected_match") is not None]
                                        status.update({
                                            "status": "completed", "seconds": round(time.perf_counter()-start, 3),
                                            "recovered_sk_seed_hex": res.recovered_seed.hex(),
                                            "full_seed_match": res.match,
                                            "assembled_seed_match": res.match,
                                            "strict_chained_recovery_match": (
                                                res.match if protocol == "chained" else None),
                                            "correct_stages": int(sum(bool(x) for x in checked)),
                                            "evaluated_stages": int(len(checked)),
                                            "stage_success_fraction": (sum(bool(x) for x in checked) / len(checked)
                                                                       if checked else None),
                                            "paper_stage_prior_truth_used": (protocol == "paper"),
                                        })
                                        json_dump(summary_path, status)
                                        json_dump(dest / "stages.json", {"stages": details})
                                        for record in details:
                                            stage_rows.append({**{k: status[k] for k in (
                                                "parameter_set", "compiler_opt", "kind", "protocol",
                                                "attack", "signatures", "selected_traces")}, **record})
                                        results.append(status)
                                        print(f"  [DONE] seed_match={res.match} "
                                              f"stages={status['correct_stages']}/{status['evaluated_stages']} "
                                              f"elapsed={status['seconds']:.1f}s", flush=True)
                                    except Exception as exc:
                                        status.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                                       "traceback": traceback.format_exc(limit=10)})
                                        json_dump(summary_path, status)
                                        results.append(status)
                                        if args.strict:
                                            raise
                                        print(f"  [FAIL] {status['error']}", flush=True)
                        del pooled
                    except Exception as exc:
                        if args.strict:
                            raise
                        missing.append(f"{p}/{opt}/{kind}: {type(exc).__name__}: {exc}")
                        print(f"  [SKIP] {missing[-1]}", flush=True)
            except Exception as exc:
                if args.strict:
                    raise
                missing.append(f"{p}/{opt}: {type(exc).__name__}: {exc}")
                print(f"  [SKIP] {missing[-1]}", flush=True)
    write_csv(output / "summary.csv", results)
    write_csv(output / "stage_results.csv", stage_rows)
    thresholds = summarize_observed_thresholds(results)
    write_csv(output / "observed_signature_thresholds.csv", thresholds)
    json_dump(output / "observed_signature_thresholds.json", {"thresholds": thresholds})
    if not args.no_summary_plots and not args.dry_run:
        plot_signature_summary(results, output)
    if missing:
        (output / "missing_or_skipped.txt").write_text("\n".join(missing), encoding="utf-8")
    info = {"version": VERSION, "base": str(native_path(args.base)),
            "output": str(output), "mode": "dry-run" if args.dry_run else "CPA",
            "settings": vars(args),
            "completed": sum(r.get("status") == "completed" for r in results),
            "failed": sum(r.get("status") == "failed" for r in results),
            "missing_or_skipped": missing}
    json_dump(output / "run_manifest.json", info)
    print("\nRun manifest:", output / "run_manifest.json")
    return info


if __name__ == "__main__":
    main()
