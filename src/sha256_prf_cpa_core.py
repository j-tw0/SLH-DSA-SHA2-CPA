"""
Unified SLH-DSA-SHA2 CPA runner for 128/192/256 and f/s parameter sets.

A selected parameter set can be executed individually, or multiple parameter sets,
address types, and seed ids can be processed automatically with the batch runner.
The f/s profile changes address ranges/capture data, but not the CPA equations at a
fixed security level.

Message-schedule strategy
-------------------------
Every CPA stage of every parameter set is executed twice: once with
carry-aware byte-wise CPA (four 8-bit sub-stages) and once with carry-aware
two-byte CPA (low16 -> high16).  Both results are scored against ground truth
when a validation seed is available, and a user-selectable primary method
decides which value is reported.

128f/128s:
    W9 is attacked twice and both results are stored:
      (a) the original 2^16 exhaustive search through the downstream W18
          target (no byte/two-byte split), and
      (b) direct byte-wise / two-byte CPA on W9 inside W16, exactly as in
          192/256, with the public low16 padding 0x8000 supplied as known.
    ``w9_128_primary_source`` selects which of the two feeds the recovered
    SK.seed and the downstream schedule words; the other is recorded with
    role='alternative'. Then W5 -> W6 -> W7 -> W8 are recovered through
    sigma0-dependent schedule words.

192f/192s:
    No prior-secret assumption is required.  The full secret chain is
        W9  via W16   (base is entirely public)
        W10 via W17
        W11 via W18   (public low16 = 0x8000 padding is supplied as a known
                       half/byte pair, so only the secret high16 is searched)
        W5  via W20
        W6 -> W7 -> W8 via W21 -> W22 -> W23

256f/256s:
    No prior-secret assumption is used.  The chain is
        W9  via W16
        W10 via W17
        W11 via W18
        W12 via W19
        joint (W5,W13) via W20
        W6 -> W7 -> W8 via W21 -> W22 -> W23
    The joint W20 stage is intentionally NOT byte-split: Stage 1 ranks 2^15
    W5 equivalence classes from HW16(W20[15:0]); Stage 2 evaluates the retained
    classes x unresolved W5 bit2 x all 2^16 W13 secret halves using HW32(W20).

Stage chaining
--------------
stage_chaining='assume_previous_success' (default) feeds the *true* word into
every downstream schedule computation whenever a validation seed is present,
so each stage's byte vs two-byte success rate is measured independently and a
single failing stage does not cascade into all later stages.  The reported
per-stage values and the assembled SK.seed still come from the CPA itself, so
the end-to-end validation match remains "every stage succeeded".
stage_chaining='recovered' restores strict end-to-end chaining, where each
stage consumes the previous stage's recovered value.

Compression strategy
--------------------
For the selected parameter set, recover the secret-bearing F_32 rounds W5 through
its last secret-bearing word using exactly one user-selected target operation:
    't1', 'e_update', or 'a_update'.
Compression CPA is performed in carry-aware two-byte (16-bit) units:
    low16 -> high16.
Known public halfwords are fixed instead of searched.  In particular, W5 has a
public high16 and the final secret-bearing word has the public low16 padding
0x8000.  The leakage model is HW only, matching the latest experiment setup.

Capture discovery
-----------------
The user supplies only a base results directory, a parameter set, compiler
optimization, address kind, and seed id.  The runner discovers captures produced
by capture_functions_fs_three_sets_multiseed_fixedrandom_both.py.  The preferred
layout is

    <BASE>/
      SLH_DSA_<param>_sha2_compiler_compare/
        <compiler>/
          multi_seed_<wots|fors>_fixed_random/
            seed_<NNN>/
              <prefix>_traces_random.npy
              <prefix>_addrs_random.npy
              <prefix>_params_random.npy
              <prefix>_meta_columns.npy
              <prefix>_capture_config.json

The discovery code also accepts BASE pointing at any lower directory in that
tree and falls back to recursive capture_config.json discovery.

Batch resume / progress
-----------------------
The batch runner writes an attack-level checkpoint after every completed CPA
attack. On restart, completed expansion/compression tasks are skipped. Overall
progress, elapsed time, and estimated remaining time are displayed on one
compact status line; detailed per-stage output is disabled by default in batch
mode.

TensorFlow
----------
Pearson correlation uses TensorFlow when backend='tensorflow' or backend='auto'
and TensorFlow is available.  The engine tries to cache the selected stage trace
window on the selected TF device.  If that fails, it uses point blocks; optional
NumPy fallback is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from matplotlib.ticker import MaxNLocator, FuncFormatter

import csv
import json
import os
import re
import time
import shutil
import zipfile
from xml.sax.saxutils import escape as _xml_escape

import numpy as np

MASK32 = 0xFFFFFFFF
ADRS_FULL_BYTES = 32
ADRS_COMPRESSED_BYTES = 22
KIND_WOTS = 5
KIND_FORS = 6
MODE_SHA2 = 0

CPA_HELPER_VARIANT = "2026-09-18-cumhw8-full-recovery-joint-w5w13-v1"

# =============================================================================
# Paper figure style (IEEE single column = 3.5 in; figure* subfloat at
# 0.48\textwidth is ~3.4 in).  The figure is created at its final printed
# width so the font sizes below are the sizes that appear on paper.
# =============================================================================
PLOT_FIG_SIZE = (3.5, 2.7)          # inches (width, height) -- fixed canvas
PLOT_FONT_LABEL = 9                 # axis labels
PLOT_FONT_TICK = 8                  # tick labels
PLOT_FONT_LEGEND = 8                # legend entries
PLOT_XLABEL = "Sample index"
PLOT_YLABEL = r"$|\rho|$"
PLOT_LEGEND_CUMULATIVE = False      # False: legend shows rho only (no cumulative score)
# Legend placement: any matplotlib ``loc`` string (drawn inside the axes) or
# "below" (legend under the x-axis label, so it never hides a peak).
PLOT_LEGEND_LOC = "below"
PLOT_LEGEND_NCOL_BELOW = 1          # one column: long labels stay inside the canvas
# Fixed axes geometry (figure fractions).  Every saved figure therefore has the
# same pixel size and the same axes rectangle, so LaTeX scales them identically
# and (a)/(b) subfloats come out the same size.  The bottom edge is computed
# from the measured legend and x-label heights.
PLOT_AXES_LEFT = 0.16
PLOT_AXES_RIGHT = 0.98
PLOT_AXES_TOP = 0.96
PLOT_LEGEND_BOTTOM_PAD = 0.015      # gap between canvas bottom and legend
PLOT_LEGEND_GAP = 0.025             # gap between legend top and x tick labels
PLOT_SAVE_TIGHT_BBOX = False        # keep False so the canvas size is preserved
# Vertical marker at the POI t* of the correct candidate:
#   True  -> always, False -> never,
#   "auto" -> only when the correct candidate is not the top curve in the plot
PLOT_MARK_TRUE_POI = "auto"
PLOT_TRUE_POI_STYLE = {"color": "0.3", "linestyle": ":", "linewidth": 0.8}
# Number of decimals on the |rho| (y) axis tick labels.  matplotlib's default
# formatter strips trailing zeros, so one figure shows 0.2 and another 0.23;
# the differing label widths make otherwise identical canvases look slightly
# different in size.  Pin the decimals (None -> keep matplotlib's default).
PLOT_YTICK_DECIMALS = 2

PARAM_N = {
    "128f": 16, "128s": 16,
    "192f": 24, "192s": 24,
    "256f": 32, "256s": 32,
}

# Public, reproducible demonstration values only (not any study dataset).
# Matches the demonstration-seed derivation used in SLH-DSA-PRF-Capture.
import hashlib as _public_demo_hashlib

def _public_demo_seed_hex(role: str, width: int) -> str:
    label = f"SLH-DSA-CAPTURE-PUBLIC-DEMO/{role}/{int(width)}".encode("ascii")
    return " ".join(f"{b:02X}" for b in _public_demo_hashlib.shake_256(label).digest(int(width)))

DEFAULT_SK_SEED_HEX = {
    f"{bits}{variant}": _public_demo_seed_hex(f"{bits}/sk_seed", width)
    for bits, width in (("128", 16), ("192", 24), ("256", 32))
    for variant in ("f", "s")
}
DEFAULT_PK_SEED_HEX = {
    f"{bits}{variant}": _public_demo_seed_hex(f"{bits}/pk_seed", width)
    for bits, width in (("128", 16), ("192", 24), ("256", 32))
    for variant in ("f", "s")
}

HW8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.float32)


def hw_u16(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint32) & np.uint32(0xFFFF)
    return (
        HW8[(x & np.uint32(0xFF)).astype(np.uint8)]
        + HW8[((x >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)]
    ).astype(np.float32)

SHA256_IV = np.array([
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
], dtype=np.uint32)

SHA256_K = np.array([
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
], dtype=np.uint32)


# ---------------------------------------------------------------------------
# Parameter/path helpers
# ---------------------------------------------------------------------------
def normalize_param_set(value: str | int) -> str:
    text = str(value).strip().lower()
    text = text.replace("slh-dsa-", "").replace("sha2-", "")
    text = text.replace("_", "").replace("-", "")
    aliases = {
        "128f": "128f", "128s": "128s", "128": "128f",
        "192f": "192f", "192s": "192s", "192": "192f",
        "256f": "256f", "256s": "256s", "256": "256f",
    }
    if text not in aliases:
        raise ValueError("parameter set must be 128f/128s/192f/192s/256f/256s")
    return aliases[text]


def security_bits(param_set: str | int) -> int:
    return int(normalize_param_set(param_set)[:3])


def get_default_sk_seed(param_set: str | int) -> bytes:
    return bytes.fromhex(DEFAULT_SK_SEED_HEX[normalize_param_set(param_set)])


def get_default_pk_seed(param_set: str | int) -> bytes:
    return bytes.fromhex(DEFAULT_PK_SEED_HEX[normalize_param_set(param_set)])


def windows_to_wsl_path(path: str | os.PathLike) -> str:
    value = str(path)
    if os.name != "nt" and len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/"):
        drive = value[0].lower()
        rest = value[3:].replace('\\', '/')
        return f"/mnt/{drive}/{rest}"
    return value


def normalize_capture_prefix(prefix: str | os.PathLike) -> str:
    value = windows_to_wsl_path(prefix)
    for suffix in (
        "_traces_fixed.npy", "_traces_random.npy",
        "_addrs_fixed.npy", "_addrs_random.npy",
        "_params_fixed.npy", "_params_random.npy",
        "_meta_columns.npy", "_capture_config.json",
    ):
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value[:-4] if value.endswith(".npz") else value


@dataclass
class CaptureData:
    traces: np.ndarray
    addrs: np.ndarray
    meta: Optional[np.ndarray]
    meta_columns: List[str]
    source: str
    group: str
    param_set: str


def _read_capture_config(prefix: str) -> Dict[str, Any]:
    path = Path(prefix + "_capture_config.json")
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_capture_legacy(
    prefix: str | os.PathLike,
    *,
    param_set: str | int,
    group: str = "WOTS_RANDOM",
    max_traces: Optional[int] = None,
    mmap_mode: Optional[str] = "r",
) -> CaptureData:
    """Load a random split and optionally filter WOTS/FORS kind."""
    p = normalize_param_set(param_set)
    normalized = normalize_capture_prefix(prefix)
    config = _read_capture_config(normalized)
    if config:
        cfg = config.get("parameter_set")
        if cfg is not None and normalize_param_set(str(cfg)) != p:
            raise ValueError(f"capture parameter_set={cfg!r}, selected={p!r}")
        cfg_n = config.get("n")
        if cfg_n is not None and int(cfg_n) != PARAM_N[p]:
            raise ValueError(f"capture n={cfg_n}, selected {p} requires n={PARAM_N[p]}")
    else:
        print("warning: capture_config.json not found; parameter-set match cannot be verified")

    traces_path = normalized + "_traces_random.npy"
    addrs_path = normalized + "_addrs_random.npy"
    params_path = normalized + "_params_random.npy"
    columns_path = normalized + "_meta_columns.npy"
    if not os.path.exists(traces_path) or not os.path.exists(addrs_path):
        raise FileNotFoundError(f"missing random split for prefix={normalized}")

    traces = np.load(traces_path, mmap_mode=mmap_mode)
    addrs = np.load(addrs_path, mmap_mode=mmap_mode)
    meta = np.load(params_path, mmap_mode=mmap_mode) if os.path.exists(params_path) else None
    columns = (
        [str(v) for v in np.load(columns_path, allow_pickle=True).tolist()]
        if os.path.exists(columns_path)
        else []
    )
    if traces.shape[0] != addrs.shape[0]:
        raise ValueError("trace/address row mismatch")
    if addrs.ndim != 2 or addrs.shape[1] != ADRS_FULL_BYTES:
        raise ValueError(f"addresses must have shape (N,32), got {addrs.shape}")

    name = str(group).upper()
    mask = np.ones(addrs.shape[0], dtype=bool)
    if meta is not None and columns:
        col = {value: index for index, value in enumerate(columns)}
        if "RANDOM" in name and "is_fixed" in col:
            mask &= np.asarray(meta[:, col["is_fixed"]] == 0)
        if "WOTS" in name:
            if "kind" not in col:
                raise ValueError("WOTS group requires metadata kind column")
            mask &= np.asarray(meta[:, col["kind"]] == KIND_WOTS)
        if "FORS" in name:
            if "kind" not in col:
                raise ValueError("FORS group requires metadata kind column")
            mask &= np.asarray(meta[:, col["kind"]] == KIND_FORS)
    elif "WOTS" in name or "FORS" in name:
        print("warning: kind metadata missing; all random rows are used")

    indices = np.flatnonzero(mask)
    if max_traces is not None:
        indices = indices[:max(0, int(max_traces))]
    if indices.size == 0:
        raise ValueError(f"no rows matched group={group}")

    contiguous = np.array_equal(indices, np.arange(indices[0], indices[0] + indices.size))
    if contiguous:
        sl = slice(int(indices[0]), int(indices[-1]) + 1)
        selected_traces = traces[sl]
        selected_addrs = addrs[sl]
        selected_meta = None if meta is None else meta[sl]
    else:
        selected_traces = traces[indices]
        selected_addrs = addrs[indices]
        selected_meta = None if meta is None else meta[indices]

    return CaptureData(
        traces=selected_traces,
        addrs=np.asarray(selected_addrs, dtype=np.uint8),
        meta=None if selected_meta is None else np.asarray(selected_meta, dtype=np.int64),
        meta_columns=columns,
        source=normalized,
        group=str(group),
        param_set=p,
    )


# ---------------------------------------------------------------------------
# SHA-256 utilities
# ---------------------------------------------------------------------------
def rotr32(x: Union[int, np.ndarray], c: int):
    value = np.asarray(x, dtype=np.uint32)
    return ((value >> np.uint32(c)) | (value << np.uint32(32 - c))).astype(np.uint32)


def sigma0_32(x: Union[int, np.ndarray]):
    value = np.asarray(x, dtype=np.uint32)
    return (rotr32(value, 7) ^ rotr32(value, 18) ^ (value >> np.uint32(3))).astype(np.uint32)


def sigma1_32(x: Union[int, np.ndarray]):
    value = np.asarray(x, dtype=np.uint32)
    return (rotr32(value, 17) ^ rotr32(value, 19) ^ (value >> np.uint32(10))).astype(np.uint32)


def Sigma0_32(x: Union[int, np.ndarray]):
    value = np.asarray(x, dtype=np.uint32)
    return (rotr32(value, 2) ^ rotr32(value, 13) ^ rotr32(value, 22)).astype(np.uint32)


def Sigma1_32(x: Union[int, np.ndarray]):
    value = np.asarray(x, dtype=np.uint32)
    return (rotr32(value, 6) ^ rotr32(value, 11) ^ rotr32(value, 25)).astype(np.uint32)


def Ch(x, y, z):
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)
    z = np.asarray(z, dtype=np.uint32)
    return ((x & y) ^ ((~x) & z)).astype(np.uint32)


def Maj(x, y, z):
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)
    z = np.asarray(z, dtype=np.uint32)
    return ((x & y) ^ (x & z) ^ (y & z)).astype(np.uint32)


def hw_u32(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint32)
    return (
        HW8[(x & np.uint32(0xFF)).astype(np.uint8)]
        + HW8[((x >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)]
        + HW8[((x >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)]
        + HW8[((x >> np.uint32(24)) & np.uint32(0xFF)).astype(np.uint8)]
    ).astype(np.float32)



# =============================================================================
# Cumulative-HW split CPA (Tunstall-style).  When CUMULATIVE_HW is True, the
# hypothesis for part j is HW(all confirmed lower parts) + HW(current part)
# instead of HW(current part) alone.  The lower parts are the beam path's
# already-selected guesses; their output (hence HW) is a per-trace vector that
# depends on the public base A, so the offset changes the ranking.  Carry
# propagation is unchanged: the current part's base still includes the carry
# from the lower parts.   Set from the notebook:  cpa.CUMULATIVE_HW = True
# =============================================================================
CUMULATIVE_HW = False

# Compute only the split method that is actually propagated.
#   "both"     : original behaviour (byte-wise AND two-byte at every stage)
#   "selected" : only stage_primary_method ("byte" or "2byte") -> ~2x faster
DUAL_METHOD_MODE = "both"

# Split granularity of the COMPRESSION-round attack: 16 (default, the
# paper method) or 8 (byte-wise twin).  The message-schedule granularity is
# chosen per run with stage_primary_method.
COMPRESSION_PART_BITS = 16


def _lower_parts_hw(A: np.ndarray, parent: Mapping[str, Any], part_index: int,
                    part_bits: int, public: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-trace HW of the outputs of every confirmed part below ``part_index``.

    Recomputes each lower part's output from the public base ``A`` and the
    path's confirmed guesses, propagating carries exactly as the stages did.
    ``public`` is the public XOR word for the public_xor variants
    (add_part = public_part ^ guess); pass None for the plain add variants.
    """
    A = np.asarray(A, dtype=np.uint32).reshape(-1)
    mask = np.uint32((1 << part_bits) - 1)
    total = np.zeros(A.size, dtype=np.float32)
    carry = np.zeros(A.size, dtype=np.uint32)
    key = "halfwords" if part_bits == 16 else "bytes"
    parts = parent.get(key, [])
    for jj in range(part_index):
        g = parts[jj] if jj < len(parts) else None
        if g is None:
            break
        base = (((A >> np.uint32(part_bits * jj)) & mask) + carry).astype(np.uint32)
        if public is not None:
            pub = ((np.asarray(public, dtype=np.uint32).reshape(-1) >> np.uint32(part_bits * jj)) & mask)
            add = (pub ^ np.uint32(int(g))).astype(np.uint32)
        else:
            add = np.uint32(int(g))
        out_full = (base + add).astype(np.uint32)
        out = out_full & mask
        total += (hw_u16(out) if part_bits == 16 else HW8[out.astype(np.uint8)]).astype(np.float32)
        carry = (out_full >> np.uint32(part_bits)).astype(np.uint32)
    return total

def _sigma0_scalar(x: int) -> int:
    value = int(x) & MASK32
    return (
        ((value >> 7) | ((value << 25) & MASK32))
        ^ ((value >> 18) | ((value << 14) & MASK32))
        ^ (value >> 3)
    ) & MASK32


def _linear_inverse(function: Callable[[int], int]) -> Callable[[int], int]:
    rows: List[List[int]] = []
    for out_bit in range(32):
        coefficients = 0
        for in_bit in range(32):
            if (function(1 << in_bit) >> out_bit) & 1:
                coefficients |= 1 << in_bit
        rows.append([coefficients, 1 << out_bit])

    for column in range(32):
        pivot = next((r for r in range(column, 32) if (rows[r][0] >> column) & 1), None)
        if pivot is None:
            raise ValueError("linear map is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(32):
            if row != column and ((rows[row][0] >> column) & 1):
                rows[row][0] ^= rows[column][0]
                rows[row][1] ^= rows[column][1]

    inverse_rows = [row[1] for row in rows]

    def inverse(value: int) -> int:
        y = int(value) & MASK32
        output = 0
        for bit, row in enumerate(inverse_rows):
            if (row & y).bit_count() & 1:
                output |= 1 << bit
        return output & MASK32

    return inverse


sigma0_inv_32 = _linear_inverse(_sigma0_scalar)


def build_finalize_words(
    addrs: np.ndarray,
    *,
    param_set: str | int,
    sk_seed: Optional[bytes] = None,
) -> np.ndarray:
    p = normalize_param_set(param_set)
    n = PARAM_N[p]
    addresses = np.asarray(addrs, dtype=np.uint8)
    if addresses.ndim != 2 or addresses.shape[1] != ADRS_FULL_BYTES:
        raise ValueError("addrs must have shape (N,32)")
    seed = bytes(n) if sk_seed is None else bytes(sk_seed)
    if len(seed) != n:
        raise ValueError(f"SK.seed for {p} must contain {n} bytes")

    rows = addresses.shape[0]
    block = np.zeros((rows, 64), dtype=np.uint8)
    block[:, :ADRS_COMPRESSED_BYTES] = addresses[:, :ADRS_COMPRESSED_BYTES]
    block[:, ADRS_COMPRESSED_BYTES:ADRS_COMPRESSED_BYTES + n] = np.frombuffer(seed, dtype=np.uint8)[None, :]
    block[:, ADRS_COMPRESSED_BYTES + n] = np.uint8(0x80)
    total_bits = (64 + ADRS_COMPRESSED_BYTES + n) * 8
    block[:, 56:64] = np.frombuffer(total_bits.to_bytes(8, "big"), dtype=np.uint8)[None, :]

    b = block.reshape(rows, 16, 4).astype(np.uint32)
    return (
        (b[:, :, 0] << np.uint32(24))
        | (b[:, :, 1] << np.uint32(16))
        | (b[:, :, 2] << np.uint32(8))
        | b[:, :, 3]
    ).astype(np.uint32)


def expand_words(words0_15: np.ndarray, *, last_word: int = 23) -> np.ndarray:
    initial = np.asarray(words0_15, dtype=np.uint32)
    words = np.zeros((initial.shape[0], last_word + 1), dtype=np.uint32)
    words[:, :16] = initial
    for t in range(16, last_word + 1):
        words[:, t] = (
            words[:, t - 16]
            + sigma0_32(words[:, t - 15])
            + words[:, t - 7]
            + sigma1_32(words[:, t - 2])
        ).astype(np.uint32)
    return words


def word_to_be_bytes(value: int) -> bytes:
    return (int(value) & MASK32).to_bytes(4, "big")


# ---------------------------------------------------------------------------
# TensorFlow correlation engine
# ---------------------------------------------------------------------------
def tensorflow_status() -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False, "gpus": []}
    try:
        import tensorflow as tf
        result["available"] = True
        result["version"] = tf.__version__
        result["gpus"] = [device.name for device in tf.config.list_physical_devices("GPU")]
    except Exception as exc:
        result["error"] = repr(exc)
    return result


# Once a CUDA OOM occurred, avoid retrying TensorFlow for every later CPA stage.
# Reset with a fresh Python/Jupyter kernel; no correlation/CPA model is altered.
_TF_DISABLED_FOR_SESSION = False


def _cpa_memory_budget(name: str, default_mib: float) -> int:
    """One transient matrix's budget (MiB); input is overridable from notebook."""
    try:
        mib = float(os.environ.get(name, str(default_mib)))
        if not (0 < mib < 8192):
            raise ValueError(name)
    except (TypeError, ValueError):
        mib = float(default_mib)
    return max(1024, int(mib * 1024 * 1024))


def _cpa_max_cols(n_rows: int, item_size: int, max_cols: int, budget: int) -> int:
    """Fit one (rows, columns) array within a requested allocation budget."""
    per_col = max(1, int(n_rows)) * int(item_size)
    return max(1, min(int(max_cols), max(1, int(budget) // per_col)))


class CorrelationEngine:
    """Max-absolute Pearson CPA using memory-bounded TF or NumPy blocks.

    No changes to the hypothesis, Hamming-weight model, candidate ranking,
    carry handling or sample range. Both candidate columns AND trace points
    are bounded before host-to-device transfer to avoid huge EagerConst tensors.

    Defaults (each is a per-matrix budget, not an overall GPU memory limit):
        SLH_CPA_TF_H_MB=24 : GPU hypothesis columns
        SLH_CPA_TF_T_MB=24 : GPU trace sample columns
        SLH_CPA_NP_MB=32   : CPU float64 blocks on NumPy fallback
    """

    def __init__(
        self,
        traces: np.ndarray,
        *,
        backend: str = "tensorflow",
        tf_device: Optional[str] = None,
        point_batch: int = 2048,
        cache_traces: bool = True,
        fallback_to_numpy: bool = True,
    ):
        self.traces = np.asarray(traces, dtype=np.float32)
        if self.traces.ndim != 2:
            raise ValueError(f"traces must be 2-D, got {self.traces.shape}")
        self.point_batch = max(1, int(point_batch))
        self.fallback_to_numpy = bool(fallback_to_numpy)
        self.tf = None
        self.device = tf_device
        self.cached = False
        self._Tc = None
        self._Tn = None
        self.tf_h_budget = _cpa_memory_budget("SLH_CPA_TF_H_MB", 24)
        self.tf_t_budget = _cpa_memory_budget("SLH_CPA_TF_T_MB", 24)
        self.numpy_budget = _cpa_memory_budget("SLH_CPA_NP_MB", 32)

        request = str(backend).lower()
        if request not in ("tensorflow", "tf", "auto", "numpy"):
            raise ValueError("backend must be tensorflow/tf/auto/numpy")

        if request != "numpy" and not _TF_DISABLED_FOR_SESSION:
            try:
                import tensorflow as tf
                for gpu in tf.config.list_physical_devices("GPU"):
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError:
                        pass
                self.tf = tf
                if self.device is None:
                    self.device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
            except Exception:
                if request in ("tensorflow", "tf") and not self.fallback_to_numpy:
                    raise
                self.tf = None

        self.backend = "tensorflow" if self.tf is not None else "numpy"

        if self.tf is not None and cache_traces:
            try:
                tf = self.tf
                with tf.device(self.device):
                    T = tf.convert_to_tensor(self.traces, dtype=tf.float32)
                    Tc = T - tf.reduce_mean(T, axis=0, keepdims=True)
                    Tn = tf.sqrt(tf.reduce_sum(tf.square(Tc), axis=0))
                    self._Tc = Tc
                    self._Tn = tf.maximum(Tn, tf.constant(1e-12, tf.float32))
                    self.cached = True
            except Exception as exc:
                print("TensorFlow trace caching failed; using point-block transfer:", type(exc).__name__)
                self._Tc = None
                self._Tn = None
                self.cached = False

    def _score_numpy(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback: never materialize the whole memmapped trace as float64."""
        Hraw = np.asarray(H)
        if Hraw.ndim == 1:
            Hraw = Hraw[:, None]
        if Hraw.ndim != 2 or Hraw.shape[0] != self.traces.shape[0]:
            raise ValueError("hypothesis/trace row mismatch")
        n_rows, n_guesses = Hraw.shape
        hcols = _cpa_max_cols(n_rows, 8, n_guesses, self.numpy_budget)
        tcols = _cpa_max_cols(n_rows, 8, self.point_batch, self.numpy_budget)

        scores = np.zeros(n_guesses, dtype=np.float32)
        pois = np.zeros(n_guesses, dtype=np.int32)
        for h_start in range(0, n_guesses, hcols):
            h_stop = min(h_start + hcols, n_guesses)
            H64 = np.asarray(Hraw[:, h_start:h_stop], dtype=np.float64)
            Hc = H64 - H64.mean(axis=0, keepdims=True)
            Hn = np.maximum(np.sqrt(np.sum(Hc * Hc, axis=0)), 1e-12)
            block_scores = np.zeros(h_stop - h_start, dtype=np.float32)
            block_pois = np.zeros(h_stop - h_start, dtype=np.int32)

            for start in range(0, self.traces.shape[1], tcols):
                stop = min(start + tcols, self.traces.shape[1])
                # Read one memmap time window rather than all samples at once.
                Tb = np.asarray(self.traces[:, start:stop], dtype=np.float64)
                Tc = Tb - Tb.mean(axis=0, keepdims=True)
                Tn = np.maximum(np.sqrt(np.sum(Tc * Tc, axis=0)), 1e-12)
                corr = (Hc.T @ Tc) / (Hn[:, None] * Tn[None, :])
                ac = np.abs(np.clip(corr, -1.0, 1.0))
                local = ac.argmax(axis=1)
                peak = ac[np.arange(ac.shape[0]), local]
                better = peak > block_scores
                block_scores[better] = peak[better].astype(np.float32)
                block_pois[better] = (start + local[better]).astype(np.int32)
            scores[h_start:h_stop] = block_scores
            pois[h_start:h_stop] = block_pois
        return scores, pois

    def _score_tensorflow(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tf = self.tf
        assert tf is not None
        Hnp = np.asarray(H, dtype=np.float32)
        if Hnp.ndim == 1:
            Hnp = Hnp[:, None]
        if Hnp.ndim != 2 or Hnp.shape[0] != self.traces.shape[0]:
            raise ValueError("hypothesis/trace row mismatch")
        n_rows, n_guesses = Hnp.shape
        hcols = _cpa_max_cols(n_rows, 4, n_guesses, self.tf_h_budget)
        tcols = _cpa_max_cols(n_rows, 4, self.point_batch, self.tf_t_budget)
        scores = np.zeros(n_guesses, dtype=np.float32)
        pois = np.zeros(n_guesses, dtype=np.int32)

        with tf.device(self.device):
            for h_start in range(0, n_guesses, hcols):
                h_stop = min(h_start + hcols, n_guesses)
                # Critical: copy only hcols to the GPU (NOT the complete Hnp).
                Ht = tf.convert_to_tensor(Hnp[:, h_start:h_stop], dtype=tf.float32)
                Hc = Ht - tf.reduce_mean(Ht, axis=0, keepdims=True)
                Hn = tf.sqrt(tf.reduce_sum(tf.square(Hc), axis=0))
                Hn = tf.maximum(Hn, tf.constant(1e-12, tf.float32))
                block_scores = np.zeros(h_stop - h_start, dtype=np.float32)
                block_pois = np.zeros(h_stop - h_start, dtype=np.int32)

                for start in range(0, self.traces.shape[1], tcols):
                    stop = min(start + tcols, self.traces.shape[1])
                    if self.cached:
                        Tc = self._Tc[:, start:stop]
                        Tn = self._Tn[start:stop]
                    else:
                        # Critical: copy only tcols to the GPU (NOT full traces).
                        Tb = tf.convert_to_tensor(self.traces[:, start:stop], dtype=tf.float32)
                        Tc = Tb - tf.reduce_mean(Tb, axis=0, keepdims=True)
                        Tn = tf.sqrt(tf.reduce_sum(tf.square(Tc), axis=0))
                        Tn = tf.maximum(Tn, tf.constant(1e-12, tf.float32))
                    corr = tf.matmul(Hc, Tc, transpose_a=True)
                    corr = corr / (Hn[:, None] * Tn[None, :])
                    ac = tf.abs(corr)
                    peak = tf.reduce_max(ac, axis=1).numpy().astype(np.float32)
                    local = tf.argmax(ac, axis=1, output_type=tf.int32).numpy()
                    better = peak > block_scores
                    block_scores[better] = peak[better]
                    block_pois[better] = start + local[better]
                    del corr, ac, peak, local
                    if not self.cached:
                        del Tb, Tc, Tn
                scores[h_start:h_stop] = block_scores
                pois[h_start:h_stop] = block_pois
                del Ht, Hc, Hn
        return scores, pois

    def score(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        global _TF_DISABLED_FOR_SESSION
        if self.backend == "numpy":
            return self._score_numpy(H)
        try:
            return self._score_tensorflow(H)
        except Exception as exc:
            if not self.fallback_to_numpy:
                raise
            # Repeated OOMs for every CPA candidate/stage are costly and noisy.
            # One warning per kernel is enough; future engines use NumPy.
            if not _TF_DISABLED_FOR_SESSION:
                print("[CPA] TensorFlow failed; switching to memory-bounded NumPy "
                      "for the rest of this kernel:", type(exc).__name__, str(exc)[:300])
            _TF_DISABLED_FOR_SESSION = True
            self.tf = None
            self.backend = "numpy"
            self._Tc = None
            self._Tn = None
            self.cached = False
            return self._score_numpy(H)

    def curve(self, hypothesis: np.ndarray) -> np.ndarray:
        """Compute one Pearson curve without converting full trace memmap."""
        h = np.asarray(hypothesis, dtype=np.float64).reshape(-1)
        if h.size != self.traces.shape[0]:
            raise ValueError("hypothesis/trace row mismatch")
        hc = h - float(h.mean(dtype=np.float64))
        hn = max(float(np.sqrt(np.sum(hc * hc))), 1e-12)
        tcols = _cpa_max_cols(self.traces.shape[0], 8, self.point_batch, self.numpy_budget)
        output = np.zeros(self.traces.shape[1], dtype=np.float32)
        for start in range(0, self.traces.shape[1], tcols):
            stop = min(start + tcols, self.traces.shape[1])
            Tb = np.asarray(self.traces[:, start:stop], dtype=np.float64)
            Tc = Tb - Tb.mean(axis=0, keepdims=True)
            tn = np.sqrt(np.sum(Tc * Tc, axis=0))
            den = hn * np.maximum(tn, 1e-12)
            output[start:stop] = np.abs(np.clip((hc @ Tc) / den, -1.0, 1.0)).astype(np.float32)
        return output


# ---------------------------------------------------------------------------
# Generic CPA candidates/plots
# ---------------------------------------------------------------------------
@dataclass
class ByteCandidate:
    word: int
    score: float
    bytes_lsb: List[int]
    corr_history: List[float]
    poi_history: List[int]
    last_corr: float
    last_poi: int


@dataclass
class HalfwordCandidate:
    word: int
    score: float
    halfwords_lsb: List[int]
    corr_history: List[float]
    poi_history: List[int]
    last_corr: float
    last_poi: int


@dataclass
class ExpansionResult:
    param_set: str
    recovered_seed: bytes
    validation_seed: Optional[bytes]
    recovered: Dict[str, int]
    stage_info: Dict[str, Any]
    stage_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    method_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def match(self) -> Optional[bool]:
        return None if self.validation_seed is None else self.recovered_seed == self.validation_seed


@dataclass
class CompressionWordResult:
    round_index: int
    word_array: np.ndarray
    representative_word: int
    candidates: List[HalfwordCandidate]


@dataclass
class CompressionResult:
    param_set: str
    target: str
    recovered_seed: bytes
    validation_seed: Optional[bytes]
    word_results: Dict[int, CompressionWordResult]

    @property
    def match(self) -> Optional[bool]:
        return None if self.validation_seed is None else self.recovered_seed == self.validation_seed



@dataclass
class PlotConfig:
    """Output/display settings for CPA stage figures."""
    root: Optional[Path]
    param_set: str
    compiler_opt: str
    address_kind: str
    seed_id: int
    save_plots: bool = True
    show_plots: bool = True
    dpi: int = 300
    print_saved_paths: bool = True


def _safe_plot_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value).strip())
    return cleaned.strip("_") or "stage"


def _plot_path(
    config: Optional[PlotConfig],
    *,
    attack: str,
    stage: str,
    filename: str,
) -> Optional[Path]:
    if config is None or not config.save_plots or config.root is None:
        return None
    directory = (
        config.root
        / _safe_plot_component(attack)
        / _safe_plot_component(stage)
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _safe_plot_component(filename)


def _finish_figure(
    fig,
    *,
    config: Optional[PlotConfig],
    save_path: Optional[Path],
) -> None:
    import matplotlib.pyplot as plt

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=(300 if config is None else int(config.dpi)),
            bbox_inches=("tight" if PLOT_SAVE_TIGHT_BBOX else None),
            facecolor="white",
        )
        if config is None or bool(config.print_saved_paths):
            print("saved plot:", save_path)

    show = True if config is None else bool(config.show_plots)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _apply_rho_tick_format(ax) -> None:
    """Force the |rho| (y) axis tick labels to a fixed number of decimals.

    Without this, matplotlib strips trailing zeros (0.2 vs 0.23), so the
    y-axis label width -- and therefore the apparent figure size -- varies
    from plot to plot.  Controlled by ``PLOT_YTICK_DECIMALS``.
    """
    if PLOT_YTICK_DECIMALS is None:
        return
    decimals = int(PLOT_YTICK_DECIMALS)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}")
    )


def _legend_suffix(last_corr: float, cumulative_score: Optional[float] = None) -> str:
    # The paper defines only rho (the maximum absolute correlation).  The
    # cumulative beam score is an internal quantity and is not shown unless
    # PLOT_LEGEND_CUMULATIVE is set.
    if cumulative_score is None or not PLOT_LEGEND_CUMULATIVE:
        return f"ρ={float(last_corr):.4f}"
    return f"ρ={float(last_corr):.4f} Σρ={float(cumulative_score):.4f}"


def _legend_label_with_rank(label: str, rank: Optional[int]) -> str:
    """Append the full-window rank to a legend entry.

    Used for the correct candidate when it is not among the top plotted
    candidates (e.g. the W16-based WOTS+ stage where it ranks 69th).
    """
    if rank is None:
        return label
    return f"{label} (rank {int(rank)})"


def _apply_paper_legend(
    fig,
    ax,
    text_colors: Sequence[str],
    *,
    dpi: Optional[int] = None,
) -> None:
    """Draw the legend and fix the axes geometry in the paper style.

    Call after all curves/labels are set (no ``tight_layout`` needed).  With
    ``PLOT_LEGEND_LOC == "below"`` the legend is placed under the x-axis
    label and the axes bottom is computed from the measured legend and
    label heights; left/right/top come from PLOT_AXES_*.  All figures saved
    by this module therefore share one canvas size and one axes rectangle.
    """
    common = dict(
        fontsize=PLOT_FONT_LEGEND,
        borderpad=0.25,
        labelspacing=0.2,
        handlelength=1.2,
        handletextpad=0.4,
        framealpha=0.9,
    )
    handles, labels = ax.get_legend_handles_labels()
    legend = None
    below = str(PLOT_LEGEND_LOC).lower() == "below"
    if handles:
        if below:
            legend = fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, float(PLOT_LEGEND_BOTTOM_PAD)),
                ncol=max(1, int(PLOT_LEGEND_NCOL_BELOW)),
                columnspacing=1.0,
                borderaxespad=0.0,
                **common,
            )
        else:
            legend = ax.legend(loc=str(PLOT_LEGEND_LOC), borderaxespad=0.25, **common)
        for legend_text, color in zip(legend.get_texts(), text_colors):
            legend_text.set_color(color)

    # Provisional rectangle, then measure what the tick labels / x-label /
    # legend need below the axes spine and set the final rectangle.  Text
    # extents are rounded to whole pixels, so the measurement is done at the
    # DPI the figure will be saved with (``dpi``); otherwise the legend height
    # differs by several percent between screen and file.
    left, right, top = float(PLOT_AXES_LEFT), float(PLOT_AXES_RIGHT), float(PLOT_AXES_TOP)
    ax.set_position([left, 0.5, right - left, top - 0.5])
    original_dpi = fig.get_dpi()
    if dpi is not None and int(dpi) > 0:
        fig.set_dpi(int(dpi))
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_h = float(fig.bbox.height)
        below_px = float(ax.get_window_extent(renderer).y0 - ax.get_tightbbox(renderer).y0)
        legend_top = 0.0
        if legend is not None and below:
            legend_top = float(legend.get_window_extent(renderer).y1)
        bottom = (legend_top + below_px) / fig_h + float(PLOT_LEGEND_GAP)
        bottom = min(bottom, top - 0.2)
        ax.set_position([left, bottom, right - left, top - bottom])
    finally:
        fig.set_dpi(original_dpi)


def _true_path_record(
    records: Sequence[Tuple[float, int, int, float, int]],
    parents: Sequence[Mapping[str, Any]],
    half_index: int,
    true_halves: Mapping[int, int],
    known: Mapping[int, Any],
) -> Optional[Tuple[int, Tuple[float, int, int, float, int]]]:
    """Locate the correct candidate of one 16-bit beam stage.

    ``records`` must already be sorted by descending total score.  Returns
    ``(rank, record)`` where ``rank`` is 1-based, or ``None`` when the
    correct prefix did not survive the beam at an earlier stage.
    """
    expected = true_halves.get(int(half_index))
    if expected is None:
        return None
    expected = int(expected) & 0xFFFF
    for rank, record in enumerate(records, 1):
        _total, parent_index, guess, _corr, _poi = record
        if int(guess) != expected:
            continue
        parent = parents[int(parent_index)]
        prefix_ok = True
        for i in range(int(half_index)):
            if i in known:
                continue
            if int(parent["halfwords"][i]) != (int(true_halves[i]) & 0xFFFF):
                prefix_ok = False
                break
        if prefix_ok:
            return rank, record
    return None


def _legend_label_byte(
    value: int,
    last_corr: float,
    cumulative_score: Optional[float] = None,
) -> str:
    return f"0x{int(value) & 0xFF:02X} {_legend_suffix(last_corr, cumulative_score)}"


def _legend_label_halfword(
    value: int,
    last_corr: float,
    cumulative_score: Optional[float] = None,
) -> str:
    return f"0x{int(value) & 0xFFFF:04X} {_legend_suffix(last_corr, cumulative_score)}"


def _legend_label_word(
    value: int,
    last_corr: float,
    cumulative_score: Optional[float] = None,
) -> str:
    return f"0x{int(value) & 0xFFFFFFFF:08X} {_legend_suffix(last_corr, cumulative_score)}"


def _plot_curves(
    engine: CorrelationEngine,
    hypotheses: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    correct_index: Optional[int],
    x_offset: int,
    title: Optional[str] = None,
    plot_config: Optional[PlotConfig] = None,
    attack: str = "cpa",
    stage: str = "stage",
    filename: str = "result.png",
    mark_true_poi: Optional[Union[bool, str]] = None,
) -> None:
    """Plot candidate correlation curves, display them, and optionally save.

    ``mark_true_poi`` overrides the module-level ``PLOT_MARK_TRUE_POI``
    (True / False / "auto") for this figure only.
    """
    import matplotlib.pyplot as plt

    if not hypotheses:
        return

    curves = [engine.curve(h) for h in hypotheses]
    peaks = [float(np.max(curve)) for curve in curves]
    blue_index = next(
        (i for i in range(len(curves)) if i != correct_index),
        None,
    )
    red_peak = None if correct_index is None else peaks[correct_index]
    blue_peak = None if blue_index is None else peaks[blue_index]

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
    text_colors: List[str] = []

    for i, (curve, label) in enumerate(zip(curves, labels)):
        if i == correct_index:
            color, width, alpha = "red", 0.5, 0.5
            z = 4 if blue_peak is None or blue_peak >= red_peak else 3
        elif i == blue_index:
            color, width, alpha = "blue", 0.5, 0.5
            z = 4 if red_peak is None or red_peak > blue_peak else 3
        else:
            color, width, alpha, z = "gray", 0.5, 0.5, 1

        x = np.arange(curve.size) + int(x_offset)
        ax.plot(
            x,
            curve,
            color=color,
            linewidth=width,
            alpha=alpha,
            zorder=z,
            label=label,
        )
        text_colors.append(
            color if i in (correct_index, blue_index) else "0.35"
        )

    # Mark t*, the POI of the correct candidate (its own argmax).  In "auto"
    # mode the marker is drawn only when a wrong candidate peaks higher, i.e.
    # when the reader needs to see *where* the correct candidate leaks.
    if correct_index is not None and mark_true_poi is None:
        mark_true_poi = PLOT_MARK_TRUE_POI
    if correct_index is not None and mark_true_poi:
        others = [p for i, p in enumerate(peaks) if i != correct_index]
        draw = (
            mark_true_poi is True
            or (others and max(others) > peaks[correct_index])
        )
        if draw:
            t_star = int(np.argmax(curves[correct_index])) + int(x_offset)
            ax.axvline(t_star, zorder=0, **PLOT_TRUE_POI_STYLE)
            ax.annotate(
                r"$t^*$",
                xy=(t_star, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -2),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=PLOT_FONT_LEGEND,
                color=PLOT_TRUE_POI_STYLE.get("color", "0.3"),
            )

    #if title:
    #    ax.set_title(str(title))
    ax.set_xlabel(PLOT_XLABEL, fontsize=PLOT_FONT_LABEL)
    ax.set_ylabel(PLOT_YLABEL, fontsize=PLOT_FONT_LABEL)
    ax.tick_params(axis="both", labelsize=PLOT_FONT_TICK)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    _apply_rho_tick_format(ax)
    _apply_paper_legend(
        fig, ax, text_colors,
        dpi=(300 if plot_config is None else int(plot_config.dpi)),
    )
    _finish_figure(
        fig,
        config=plot_config,
        save_path=_plot_path(
            plot_config,
            attack=attack,
            stage=stage,
            filename=filename,
        ),
    )

def _stage_traces(cap: CaptureData, trace_slice: Optional[slice]) -> Tuple[np.ndarray, int]:
    if trace_slice is None:
        return np.asarray(cap.traces, dtype=np.float32), 0
    start, stop, step = trace_slice.indices(cap.traces.shape[1])
    if step != 1 or stop <= start:
        raise ValueError("trace_slice must be a non-empty step-1 slice")
    return np.asarray(cap.traces[:, start:stop], dtype=np.float32), start


def _true_constant(words: Optional[np.ndarray], index: int) -> Optional[int]:
    if words is None:
        return None
    unique = np.unique(np.asarray(words[:, index], dtype=np.uint32))
    return int(unique[0]) if unique.size == 1 else None


def _path_matches_true(path: Mapping[str, Any], true_key: Optional[int], byte_index: int) -> bool:
    if true_key is None:
        return False
    for index in range(byte_index + 1):
        value = path["bytes"][index]
        if value is None or int(value) != ((int(true_key) >> (8 * index)) & 0xFF):
            return False
    return True



def recover_add_const_bytewise(
    traces: np.ndarray,
    add_base: np.ndarray,
    *,
    engine_options: Mapping[str, Any],
    target_name: str,
    true_key: Optional[int] = None,
    known_key_bytes: Optional[Mapping[int, int]] = None,
    beam_width: int = 4,
    plot_top: int = 4,
    x_offset: int = 0,
    verbose: bool = True,
    plot_config: Optional[PlotConfig] = None,
    plot_attack: str = "message_schedule",
    plot_stage: Optional[str] = None,
) -> Tuple[int, List[ByteCandidate]]:
    """Recover K from output=A+K. Plot order follows cumulative beam score."""
    T = np.asarray(traces, dtype=np.float32)
    A = np.asarray(add_base, dtype=np.uint32).reshape(-1)
    known = {
        int(k): int(v) & 0xFF
        for k, v in dict(known_key_bytes or {}).items()
    }
    engine = CorrelationEngine(T, **dict(engine_options))
    width = max(1, int(beam_width))
    stage_name = str(plot_stage or target_name)

    paths: List[Dict[str, Any]] = [{
        "bytes": [None, None, None, None],
        "word": 0,
        "score": 0.0,
        "carry": np.zeros(A.size, dtype=np.uint32),
        "corr_history": [],
        "poi_history": [],
    }]

    guesses = np.arange(256, dtype=np.uint16)

    for byte_index in range(4):
        if byte_index in known:
            guess = int(known[byte_index])
            next_paths = []
            for parent in paths:
                base_byte = (
                    ((A >> np.uint32(8 * byte_index)) & np.uint32(0xFF))
                    + parent["carry"]
                ).astype(np.uint32)
                new = dict(parent)
                new["bytes"] = list(parent["bytes"])
                new["bytes"][byte_index] = guess
                new["word"] = int(parent["word"]) | (guess << (8 * byte_index))
                new["carry"] = (
                    (base_byte + np.uint32(guess)) >> np.uint32(8)
                ).astype(np.uint32)
                next_paths.append(new)
            paths = next_paths
            continue

        records: List[Tuple[float, int, int, float, int]] = []
        for parent_index, parent in enumerate(paths):
            base_byte = (
                ((A >> np.uint32(8 * byte_index)) & np.uint32(0xFF))
                + parent["carry"]
            ).astype(np.uint32)
            output = (
                base_byte[:, None].astype(np.uint16) + guesses[None, :]
            ) & np.uint16(0xFF)
            H = HW8[output.astype(np.uint8)]
            if CUMULATIVE_HW and byte_index > 0:
                H = H.astype(np.float32) + _lower_parts_hw(A, parent, byte_index, 8)[:, None]
            scores, pois = engine.score(H)
            for guess in range(256):
                records.append((
                    float(parent["score"]) + float(scores[guess]),
                    parent_index,
                    guess,
                    float(scores[guess]),
                    int(pois[guess]),
                ))

        records.sort(key=lambda row: row[0], reverse=True)
        selected = records[:width]
        next_paths = []
        for total, parent_index, guess, last_corr, poi in selected:
            parent = paths[parent_index]
            base_byte = (
                ((A >> np.uint32(8 * byte_index)) & np.uint32(0xFF))
                + parent["carry"]
            ).astype(np.uint32)
            new_bytes = list(parent["bytes"])
            new_bytes[byte_index] = guess
            next_paths.append({
                "bytes": new_bytes,
                "word": int(parent["word"]) | (guess << (8 * byte_index)),
                "score": total,
                "carry": (
                    (base_byte + np.uint32(guess)) >> np.uint32(8)
                ).astype(np.uint32),
                "corr_history": list(parent["corr_history"]) + [last_corr],
                "poi_history": list(parent["poi_history"]) + [x_offset + poi],
                "last_corr": last_corr,
                "last_poi": x_offset + poi,
                "last_guess": guess,
            })
        paths = next_paths

        if verbose:
            print(
                f"[{target_name}] byte {byte_index}, "
                f"backend={engine.backend}, cached={engine.cached}"
            )
            for rank, path in enumerate(paths, 1):
                mark = (
                    " OK"
                    if _path_matches_true(path, true_key, byte_index)
                    else ""
                )
                print(
                    f"word=0x{int(path['word']) & MASK32:08X} "
                    f"corr={float(path['last_corr']):.6f} "
                    f"score={float(path['score']):.6f}{mark}"
                )

        if plot_top > 0:
            displayed = sorted(
                paths,
                key=lambda item: float(item["score"]),
                reverse=True,
            )[:max(1, int(plot_top))]

            hypotheses: List[np.ndarray] = []
            labels: List[str] = []
            correct_index: Optional[int] = None

            for index, path in enumerate(displayed):
                carry = np.zeros(A.size, dtype=np.uint32)
                for lower in range(byte_index):
                    bbase = (
                        ((A >> np.uint32(8 * lower)) & np.uint32(0xFF))
                        + carry
                    ).astype(np.uint32)
                    carry = (
                        (bbase + np.uint32(path["bytes"][lower]))
                        >> np.uint32(8)
                    ).astype(np.uint32)

                bbase = (
                    ((A >> np.uint32(8 * byte_index)) & np.uint32(0xFF))
                    + carry
                ).astype(np.uint32)
                out = (
                    (bbase + np.uint32(path["bytes"][byte_index]))
                    & np.uint32(0xFF)
                ).astype(np.uint8)
                hypotheses.append(HW8[out])
                labels.append(
                    _legend_label_word(
                        int(path["word"]),
                        float(path["last_corr"]),
                        float(path["score"]),
                    )
                )
                if _path_matches_true(path, true_key, byte_index):
                    correct_index = index

            _plot_curves(
                engine,
                hypotheses,
                labels,
                correct_index=correct_index,
                x_offset=x_offset,
                title=f"{target_name} / byte {byte_index}",
                plot_config=plot_config,
                attack=plot_attack,
                stage=stage_name,
                filename=f"byte_{byte_index}.png",
            )

    compact = [
        ByteCandidate(
            word=int(path["word"]) & MASK32,
            score=float(path["score"]),
            bytes_lsb=[int(v) for v in path["bytes"]],
            corr_history=[float(v) for v in path["corr_history"]],
            poi_history=[int(v) for v in path["poi_history"]],
            last_corr=float(path.get("last_corr", 0.0)),
            last_poi=int(path.get("last_poi", -1)),
        )
        for path in paths
    ]
    return compact[0].word, compact


def recover_add_const_halfwordwise(
    traces: np.ndarray,
    add_base: np.ndarray,
    *,
    engine_options: Mapping[str, Any],
    target_name: str,
    true_key: Optional[int] = None,
    known_halfwords: Optional[Mapping[int, int]] = None,
    beam_width: int = 4,
    candidate_batch: int = 256,
    plot_top: int = 4,
    x_offset: int = 0,
    verbose: bool = True,
    plot_config: Optional[PlotConfig] = None,
    plot_attack: str = "message_schedule",
    plot_stage: Optional[str] = None,
) -> Tuple[int, List[HalfwordCandidate]]:
    """Recover K from output=A+K in two carry-aware 16-bit stages.

    Stage 0 guesses K[15:0] and uses HW(output[15:0]). Stage 1 propagates
    the low-half carry, guesses K[31:16], and uses HW(output[31:16]).

    ``known_halfwords`` fixes a public half (for example the 0x8000 SHA-2
    padding in the last secret-bearing word) instead of searching it.
    """
    T = np.asarray(traces, dtype=np.float32)
    A = np.asarray(add_base, dtype=np.uint32).reshape(-1)
    known = {
        int(k): int(v) & 0xFFFF
        for k, v in dict(known_halfwords or {}).items()
    }
    engine = CorrelationEngine(T, **dict(engine_options))
    width = max(1, int(beam_width))
    batch = max(1, int(candidate_batch))
    stage_name = str(plot_stage or target_name)

    paths: List[Dict[str, Any]] = [{
        "halfwords": [None, None],
        "word": 0,
        "score": 0.0,
        "carry": np.zeros(A.size, dtype=np.uint32),
        "corr_history": [],
        "poi_history": [],
        "last_corr": 0.0,
        "last_poi": -1,
    }]

    for half_index in range(2):
        if half_index in known:
            guess = int(known[half_index])
            next_paths = []
            for parent in paths:
                base_half = (
                    ((A >> np.uint32(16 * half_index)) & np.uint32(0xFFFF))
                    + parent["carry"]
                ).astype(np.uint32)
                halfwords = list(parent["halfwords"])
                halfwords[half_index] = guess
                new = dict(parent)
                new["halfwords"] = halfwords
                new["word"] = int(parent["word"]) | (guess << (16 * half_index))
                new["carry"] = (
                    (base_half + np.uint32(guess)) >> np.uint32(16)
                ).astype(np.uint32)
                next_paths.append(new)
            paths = next_paths
            if verbose:
                part = "low16" if half_index == 0 else "high16"
                print(f"[{target_name}] {part} fixed to 0x{guess:04X} (public)")
            continue

        records: List[Tuple[float, int, int, float, int]] = []
        parents = paths  # beam entries that generated ``records``

        for parent_index, parent in enumerate(paths):
            base_half = (
                ((A >> np.uint32(16 * half_index)) & np.uint32(0xFFFF))
                + parent["carry"]
            ).astype(np.uint32)

            for start in range(0, 65536, batch):
                stop = min(start + batch, 65536)
                guesses = np.arange(start, stop, dtype=np.uint32)
                output = (
                    base_half[:, None] + guesses[None, :]
                ) & np.uint32(0xFFFF)
                H = hw_u16(output)
                if CUMULATIVE_HW and half_index > 0:
                    H = H.astype(np.float32) + _lower_parts_hw(A, parent, half_index, 16)[:, None]
                scores, pois = engine.score(H)
                for local, guess in enumerate(range(start, stop)):
                    records.append((
                        float(parent["score"]) + float(scores[local]),
                        parent_index,
                        int(guess),
                        float(scores[local]),
                        int(pois[local]),
                    ))

        records.sort(key=lambda row: row[0], reverse=True)
        next_paths: List[Dict[str, Any]] = []
        for total, parent_index, guess, last_corr, poi in records[:width]:
            parent = paths[parent_index]
            base_half = (
                ((A >> np.uint32(16 * half_index)) & np.uint32(0xFFFF))
                + parent["carry"]
            ).astype(np.uint32)
            halfwords = list(parent["halfwords"])
            halfwords[half_index] = int(guess)
            next_paths.append({
                "halfwords": halfwords,
                "word": int(parent["word"]) | (int(guess) << (16 * half_index)),
                "score": float(total),
                "carry": ((base_half + np.uint32(guess)) >> np.uint32(16)).astype(np.uint32),
                "corr_history": list(parent["corr_history"]) + [float(last_corr)],
                "poi_history": list(parent["poi_history"]) + [int(x_offset + poi)],
                "last_corr": float(last_corr),
                "last_poi": int(x_offset + poi),
            })
        paths = next_paths

        if verbose:
            part = "low16" if half_index == 0 else "high16"
            print(f"[{target_name}] {part}, backend={engine.backend}, cached={engine.cached}")
            for rank, path in enumerate(paths, 1):
                mark = ""
                if true_key is not None:
                    correct = True
                    for i in range(half_index + 1):
                        expected = (int(true_key) >> (16 * i)) & 0xFFFF
                        if int(path["halfwords"][i]) != expected:
                            correct = False
                            break
                    mark = " OK" if correct else ""
                print(
                    f"  [{rank}] 0x{int(path['word']) & MASK32:08X} "
                    f"corr={float(path['last_corr']):.6f} "
                    f"score={float(path['score']):.6f}{mark}"
                )

        if plot_top > 0:
            displayed = sorted(
                paths,
                key=lambda item: float(item["score"]),
                reverse=True,
            )[:max(1, int(plot_top))]
            display_ranks: List[Optional[int]] = [None] * len(displayed)

            def _is_true_path(path: Mapping[str, Any]) -> bool:
                if true_key is None:
                    return False
                for i in range(half_index + 1):
                    expected = (int(true_key) >> (16 * i)) & 0xFFFF
                    if int(path["halfwords"][i]) != expected:
                        return False
                return True

            # Paper figure: always draw the correct candidate, even when it is
            # not among the top-``plot_top`` beam entries (e.g. rank 69 in the
            # W16-based WOTS+ stage), and state its full-window rank.
            if true_key is not None and not any(_is_true_path(p) for p in displayed):
                true_halves = {
                    i: (int(true_key) >> (16 * i)) & 0xFFFF
                    for i in range(2)
                    if i not in known
                }
                hit = _true_path_record(records, parents, half_index, true_halves, known)
                if hit is not None:
                    rank, (total, parent_index, guess, last_corr, poi) = hit
                    parent = parents[int(parent_index)]
                    halfwords = list(parent["halfwords"])
                    halfwords[half_index] = int(guess)
                    displayed.append({
                        "halfwords": halfwords,
                        "word": int(parent["word"]) | (int(guess) << (16 * half_index)),
                        "score": float(total),
                        "last_corr": float(last_corr),
                        "last_poi": int(x_offset + poi),
                    })
                    display_ranks.append(int(rank))

            hypotheses: List[np.ndarray] = []
            labels: List[str] = []
            correct_index: Optional[int] = None

            for display_index, path in enumerate(displayed):
                carry = np.zeros(A.size, dtype=np.uint32)
                if half_index == 1:
                    low_base = (A & np.uint32(0xFFFF)).astype(np.uint32)
                    carry = (
                        (low_base + np.uint32(path["halfwords"][0]))
                        >> np.uint32(16)
                    ).astype(np.uint32)

                base_half = (
                    ((A >> np.uint32(16 * half_index)) & np.uint32(0xFFFF))
                    + carry
                ).astype(np.uint32)
                output = (
                    base_half + np.uint32(path["halfwords"][half_index])
                ) & np.uint32(0xFFFF)
                hypotheses.append(hw_u16(output))

                if half_index == 0:
                    label = _legend_label_halfword(
                        int(path["halfwords"][0]),
                        float(path["last_corr"]),
                        float(path["score"]),
                    )
                else:
                    label = _legend_label_word(
                        int(path["word"]),
                        float(path["last_corr"]),
                        float(path["score"]),
                    )
                labels.append(
                    _legend_label_with_rank(label, display_ranks[display_index])
                )

                if _is_true_path(path):
                    correct_index = display_index

            _plot_curves(
                engine,
                hypotheses,
                labels,
                correct_index=correct_index,
                x_offset=x_offset,
                title=f"{target_name} / {'low16' if half_index == 0 else 'high16'}",
                plot_config=plot_config,
                attack=plot_attack,
                stage=stage_name,
                filename=f"halfword_{half_index}.png",
            )

    compact = [
        HalfwordCandidate(
            word=int(path["word"]) & MASK32,
            score=float(path["score"]),
            halfwords_lsb=[int(v) for v in path["halfwords"]],
            corr_history=[float(v) for v in path["corr_history"]],
            poi_history=[int(v) for v in path["poi_history"]],
            last_corr=float(path["last_corr"]),
            last_poi=int(path["last_poi"]),
        )
        for path in paths
    ]
    return compact[0].word, compact


def recover_add_public_xor_const_bytewise(
    traces: np.ndarray,
    add_base: np.ndarray,
    public_xor: np.ndarray,
    *,
    engine_options: Mapping[str, Any],
    target_name: str,
    true_key: Optional[int] = None,
    beam_width: int = 4,
    plot_top: int = 4,
    x_offset: int = 0,
    verbose: bool = True,
    plot_config: Optional[PlotConfig] = None,
    plot_attack: str = "message_schedule",
    plot_stage: Optional[str] = None,
) -> Tuple[int, List[ByteCandidate]]:
    """Recover K from A+(P XOR K), plotting every CPA-recovered byte."""
    T = np.asarray(traces, dtype=np.float32)
    A = np.asarray(add_base, dtype=np.uint32).reshape(-1)
    P = np.asarray(public_xor, dtype=np.uint32).reshape(-1)
    engine = CorrelationEngine(T, **dict(engine_options))
    stage_name = str(plot_stage or target_name)

    paths: List[Dict[str, Any]] = [{
        "bytes": [None, None, None, None],
        "word": 0,
        "score": 0.0,
        "carry": np.zeros(A.size, dtype=np.uint32),
        "corr_history": [],
        "poi_history": [],
    }]
    guesses = np.arange(256, dtype=np.uint16)

    for byte_index in range(4):
        public_byte = (
            (P >> np.uint32(8 * byte_index)) & np.uint32(0xFF)
        ).astype(np.uint32)
        records: List[Tuple[float, int, int, float, int]] = []

        for parent_index, parent in enumerate(paths):
            base_byte = (
                ((A >> np.uint32(8 * byte_index)) & np.uint32(0xFF))
                + parent["carry"]
            ).astype(np.uint32)
            add_byte = public_byte[:, None].astype(np.uint16) ^ guesses[None, :]
            output = (
                base_byte[:, None].astype(np.uint16) + add_byte
            ) & np.uint16(0xFF)
            H = HW8[output.astype(np.uint8)]
            if CUMULATIVE_HW and byte_index > 0:
                H = H.astype(np.float32) + _lower_parts_hw(A, parent, byte_index, 8, public=P)[:, None]
            scores, pois = engine.score(H)
            for guess in range(256):
                records.append((
                    float(parent["score"]) + float(scores[guess]),
                    parent_index,
                    guess,
                    float(scores[guess]),
                    int(pois[guess]),
                ))

        records.sort(key=lambda row: row[0], reverse=True)
        next_paths = []
        for total, parent_index, guess, last_corr, poi in records[
            :max(1, int(beam_width))
        ]:
            parent = paths[parent_index]
            base_byte = (
                ((A >> np.uint32(8 * byte_index)) & np.uint32(0xFF))
                + parent["carry"]
            ).astype(np.uint32)
            add_byte = public_byte ^ np.uint32(guess)
            new_bytes = list(parent["bytes"])
            new_bytes[byte_index] = guess
            next_paths.append({
                "bytes": new_bytes,
                "word": int(parent["word"]) | (guess << (8 * byte_index)),
                "score": total,
                "carry": (
                    (base_byte + add_byte) >> np.uint32(8)
                ).astype(np.uint32),
                "corr_history": list(parent["corr_history"]) + [last_corr],
                "poi_history": list(parent["poi_history"]) + [x_offset + poi],
                "last_corr": last_corr,
                "last_poi": x_offset + poi,
                "last_guess": guess,
            })
        paths = next_paths

        if verbose:
            print(
                f"[{target_name}] byte {byte_index}, "
                f"backend={engine.backend}, cached={engine.cached}"
            )
            for rank, path in enumerate(paths, 1):
                mark = (
                    " OK"
                    if _path_matches_true(path, true_key, byte_index)
                    else ""
                )
                print(
                    f"corr={path['last_corr']:.6f} "
                    f"score={path['score']:.6f} "
                )

        if plot_top > 0:
            displayed = sorted(
                paths,
                key=lambda item: float(item["score"]),
                reverse=True,
            )[:max(1, int(plot_top))]
            hypotheses: List[np.ndarray] = []
            labels: List[str] = []
            correct_index: Optional[int] = None

            for index, path in enumerate(displayed):
                carry = np.zeros(A.size, dtype=np.uint32)
                for lower in range(byte_index):
                    lower_public = (
                        (P >> np.uint32(8 * lower)) & np.uint32(0xFF)
                    ).astype(np.uint32)
                    lower_base = (
                        ((A >> np.uint32(8 * lower)) & np.uint32(0xFF))
                        + carry
                    ).astype(np.uint32)
                    lower_add = (
                        lower_public ^ np.uint32(path["bytes"][lower])
                    )
                    carry = (
                        (lower_base + lower_add) >> np.uint32(8)
                    ).astype(np.uint32)

                current_base = (
                    ((A >> np.uint32(8 * byte_index)) & np.uint32(0xFF))
                    + carry
                ).astype(np.uint32)
                current_add = public_byte ^ np.uint32(path["bytes"][byte_index])
                out = (
                    (current_base + current_add) & np.uint32(0xFF)
                ).astype(np.uint8)
                hypotheses.append(HW8[out])
                labels.append(
                    _legend_label_word(
                        int(path["word"]),
                        float(path["last_corr"]),
                        float(path["score"]),
                    )
                )
                if _path_matches_true(path, true_key, byte_index):
                    correct_index = index

            _plot_curves(
                engine,
                hypotheses,
                labels,
                correct_index=correct_index,
                x_offset=x_offset,
                title=f"{target_name} / byte {byte_index}",
                plot_config=plot_config,
                attack=plot_attack,
                stage=stage_name,
                filename=f"byte_{byte_index}.png",
            )

    compact = [
        ByteCandidate(
            word=int(path["word"]) & MASK32,
            score=float(path["score"]),
            bytes_lsb=[int(v) for v in path["bytes"]],
            corr_history=[float(v) for v in path["corr_history"]],
            poi_history=[int(v) for v in path["poi_history"]],
            last_corr=float(path["last_corr"]),
            last_poi=int(path["last_poi"]),
        )
        for path in paths
    ]
    return compact[0].word, compact


def recover_add_public_xor_const_halfwordwise(
    traces: np.ndarray,
    add_base: np.ndarray,
    public_xor: np.ndarray,
    *,
    engine_options: Mapping[str, Any],
    target_name: str,
    true_key: Optional[int] = None,
    beam_width: int = 4,
    candidate_batch: int = 256,
    plot_top: int = 4,
    x_offset: int = 0,
    verbose: bool = True,
    plot_config: Optional[PlotConfig] = None,
    plot_attack: str = "message_schedule",
    plot_stage: Optional[str] = None,
) -> Tuple[int, List[HalfwordCandidate]]:
    """Recover K from output = A + (P XOR K) in two carry-aware 16-bit stages.

    This is the two-byte counterpart of
    ``recover_add_public_xor_const_bytewise`` and is used for the W5 stage,
    where P = sigma0(public W5[31:16]) and K = sigma0(secret16).
    """
    T = np.asarray(traces, dtype=np.float32)
    A = np.asarray(add_base, dtype=np.uint32).reshape(-1)
    P = np.asarray(public_xor, dtype=np.uint32).reshape(-1)
    engine = CorrelationEngine(T, **dict(engine_options))
    width = max(1, int(beam_width))
    batch = max(1, int(candidate_batch))
    stage_name = str(plot_stage or target_name)

    paths: List[Dict[str, Any]] = [{
        "halfwords": [None, None],
        "word": 0,
        "score": 0.0,
        "carry": np.zeros(A.size, dtype=np.uint32),
        "corr_history": [],
        "poi_history": [],
        "last_corr": 0.0,
        "last_poi": -1,
    }]

    for half_index in range(2):
        public_half = (
            (P >> np.uint32(16 * half_index)) & np.uint32(0xFFFF)
        ).astype(np.uint32)
        records: List[Tuple[float, int, int, float, int]] = []
        parents = paths  # beam entries that generated ``records``

        for parent_index, parent in enumerate(paths):
            base_half = (
                ((A >> np.uint32(16 * half_index)) & np.uint32(0xFFFF))
                + parent["carry"]
            ).astype(np.uint32)

            for start in range(0, 65536, batch):
                stop = min(start + batch, 65536)
                guesses = np.arange(start, stop, dtype=np.uint32)
                add_half = (
                    public_half[:, None] ^ guesses[None, :]
                ).astype(np.uint32)
                output = (
                    base_half[:, None] + add_half
                ) & np.uint32(0xFFFF)
                H = hw_u16(output)
                if CUMULATIVE_HW and half_index > 0:
                    H = H.astype(np.float32) + _lower_parts_hw(A, parent, half_index, 16, public=P)[:, None]
                scores, pois = engine.score(H)
                for local, guess in enumerate(range(start, stop)):
                    records.append((
                        float(parent["score"]) + float(scores[local]),
                        parent_index,
                        int(guess),
                        float(scores[local]),
                        int(pois[local]),
                    ))

        records.sort(key=lambda row: row[0], reverse=True)
        next_paths: List[Dict[str, Any]] = []
        for total, parent_index, guess, last_corr, poi in records[:width]:
            parent = paths[parent_index]
            base_half = (
                ((A >> np.uint32(16 * half_index)) & np.uint32(0xFFFF))
                + parent["carry"]
            ).astype(np.uint32)
            add_half = (public_half ^ np.uint32(guess)).astype(np.uint32)
            halfwords = list(parent["halfwords"])
            halfwords[half_index] = int(guess)
            next_paths.append({
                "halfwords": halfwords,
                "word": int(parent["word"]) | (int(guess) << (16 * half_index)),
                "score": float(total),
                "carry": (
                    (base_half + add_half) >> np.uint32(16)
                ).astype(np.uint32),
                "corr_history": list(parent["corr_history"]) + [float(last_corr)],
                "poi_history": list(parent["poi_history"]) + [int(x_offset + poi)],
                "last_corr": float(last_corr),
                "last_poi": int(x_offset + poi),
            })
        paths = next_paths

        if verbose:
            part = "low16" if half_index == 0 else "high16"
            print(
                f"[{target_name}] {part}, "
                f"backend={engine.backend}, cached={engine.cached}"
            )
            for rank, path in enumerate(paths, 1):
                mark = ""
                if true_key is not None:
                    correct = True
                    for i in range(half_index + 1):
                        expected = (int(true_key) >> (16 * i)) & 0xFFFF
                        if int(path["halfwords"][i]) != expected:
                            correct = False
                            break
                    mark = " OK" if correct else ""
                print(
                    f"  [{rank}] 0x{int(path['word']) & MASK32:08X} "
                    f"corr={float(path['last_corr']):.6f} "
                    f"score={float(path['score']):.6f}{mark}"
                )

        if plot_top > 0:
            displayed = sorted(
                paths,
                key=lambda item: float(item["score"]),
                reverse=True,
            )[:max(1, int(plot_top))]
            display_ranks: List[Optional[int]] = [None] * len(displayed)

            def _is_true_path(path: Mapping[str, Any]) -> bool:
                if true_key is None:
                    return False
                for i in range(half_index + 1):
                    expected = (int(true_key) >> (16 * i)) & 0xFFFF
                    if int(path["halfwords"][i]) != expected:
                        return False
                return True

            # Always draw the correct candidate and state its rank when it is
            # not among the top-``plot_top`` beam entries.
            if true_key is not None and not any(_is_true_path(p) for p in displayed):
                true_halves = {
                    i: (int(true_key) >> (16 * i)) & 0xFFFF for i in range(2)
                }
                hit = _true_path_record(records, parents, half_index, true_halves, {})
                if hit is not None:
                    rank, (total, parent_index, guess, last_corr, poi) = hit
                    parent = parents[int(parent_index)]
                    halfwords = list(parent["halfwords"])
                    halfwords[half_index] = int(guess)
                    displayed.append({
                        "halfwords": halfwords,
                        "word": int(parent["word"]) | (int(guess) << (16 * half_index)),
                        "score": float(total),
                        "last_corr": float(last_corr),
                        "last_poi": int(x_offset + poi),
                    })
                    display_ranks.append(int(rank))

            hypotheses: List[np.ndarray] = []
            labels: List[str] = []
            correct_index: Optional[int] = None

            for display_index, path in enumerate(displayed):
                carry = np.zeros(A.size, dtype=np.uint32)
                if half_index == 1:
                    low_public = (P & np.uint32(0xFFFF)).astype(np.uint32)
                    low_base = (A & np.uint32(0xFFFF)).astype(np.uint32)
                    low_add = (
                        low_public ^ np.uint32(path["halfwords"][0])
                    ).astype(np.uint32)
                    carry = (
                        (low_base + low_add) >> np.uint32(16)
                    ).astype(np.uint32)

                base_half = (
                    ((A >> np.uint32(16 * half_index)) & np.uint32(0xFFFF))
                    + carry
                ).astype(np.uint32)
                add_half = (
                    public_half ^ np.uint32(path["halfwords"][half_index])
                ).astype(np.uint32)
                output = (base_half + add_half) & np.uint32(0xFFFF)
                hypotheses.append(hw_u16(output))

                if half_index == 0:
                    label = _legend_label_halfword(
                        int(path["halfwords"][0]),
                        float(path["last_corr"]),
                        float(path["score"]),
                    )
                else:
                    label = _legend_label_word(
                        int(path["word"]),
                        float(path["last_corr"]),
                        float(path["score"]),
                    )
                labels.append(
                    _legend_label_with_rank(label, display_ranks[display_index])
                )

                if _is_true_path(path):
                    correct_index = display_index

            _plot_curves(
                engine,
                hypotheses,
                labels,
                correct_index=correct_index,
                x_offset=x_offset,
                title=f"{target_name} / {'low16' if half_index == 0 else 'high16'}",
                plot_config=plot_config,
                attack=plot_attack,
                stage=stage_name,
                filename=f"halfword_{half_index}.png",
            )

    compact = [
        HalfwordCandidate(
            word=int(path["word"]) & MASK32,
            score=float(path["score"]),
            halfwords_lsb=[int(v) for v in path["halfwords"]],
            corr_history=[float(v) for v in path["corr_history"]],
            poi_history=[int(v) for v in path["poi_history"]],
            last_corr=float(path["last_corr"]),
            last_poi=int(path["last_poi"]),
        )
        for path in paths
    ]
    return compact[0].word, compact


@dataclass
class ExpansionContext:
    cap: CaptureData
    param_set: str
    public_words: np.ndarray
    true_words: Optional[np.ndarray]
    work: Dict[int, np.ndarray]
    expanded: Dict[int, np.ndarray]
    recovered: Dict[str, int]
    stage_info: Dict[str, Any]
    validation_seed: Optional[bytes]
    chaining: str = "assume_previous_success"
    primary_method: str = "2byte"     # which split method is propagated (used by DUAL_METHOD_MODE)
    comparisons: List[Dict[str, Any]] = field(default_factory=list)


STAGE_CHAINING_MODES = ("assume_previous_success", "recovered")


def normalize_stage_chaining(value: Optional[str]) -> str:
    """Normalize the stage-to-stage value propagation policy."""
    if value is None:
        return "assume_previous_success"
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "assume_previous_success": "assume_previous_success",
        "assume_success": "assume_previous_success",
        "assume": "assume_previous_success",
        "independent": "assume_previous_success",
        "true": "assume_previous_success",
        "ground_truth": "assume_previous_success",
        "recovered": "recovered",
        "chained": "recovered",
        "end_to_end": "recovered",
        "cascade": "recovered",
    }
    if raw not in aliases:
        raise ValueError(
            "stage_chaining must be 'assume_previous_success' or 'recovered'"
        )
    return aliases[raw]


W9_128_SOURCES = ("auto", "exhaustive_w18", "direct_w16")


def normalize_w9_128_source(value: Optional[str]) -> str:
    """Normalize which 128-bit W9 route feeds the recovered SK.seed.

    Both routes are always executed and stored; this only picks the primary.
      'exhaustive_w18' - the original 2^16 search through W18
      'direct_w16'     - byte-wise / two-byte CPA on W9 inside W16
    """
    if value is None:
        return "auto"
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "context": "auto",
        "address_kind": "auto",
        "exhaustive_w18": "exhaustive_w18",
        "exhaustive": "exhaustive_w18",
        "w18": "exhaustive_w18",
        "legacy": "exhaustive_w18",
        "original": "exhaustive_w18",
        "direct_w16": "direct_w16",
        "direct": "direct_w16",
        "w16": "direct_w16",
        "cpa": "direct_w16",
    }
    if raw not in aliases:
        raise ValueError(
            "w9_128_primary_source must be 'auto', 'exhaustive_w18', or 'direct_w16'"
        )
    return aliases[raw]


def _chain_value(
    ctx: ExpansionContext,
    recovered_value: int,
    true_value: Optional[int],
) -> int:
    """Value handed to downstream schedule words.

    With the default policy the true word is propagated when it is known, so a
    failure in one stage does not invalidate the byte vs two-byte measurement
    of every later stage. ``ctx.recovered`` always keeps the CPA result, so the
    assembled SK.seed still reflects genuine end-to-end success.
    """
    if ctx.chaining == "assume_previous_success" and true_value is not None:
        return int(true_value) & MASK32
    return int(recovered_value) & MASK32


def create_expansion_context(
    cap: CaptureData,
    *,
    validation_seed: Optional[bytes] = None,
    chaining: str = "assume_previous_success",
    primary_method: str = "2byte",
) -> ExpansionContext:
    p = cap.param_set
    public = build_finalize_words(cap.addrs, param_set=p, sk_seed=None)
    true = None
    seed = None
    if validation_seed is not None:
        seed = bytes(validation_seed)
        if len(seed) != PARAM_N[p]:
            raise ValueError(f"validation seed for {p} must contain {PARAM_N[p]} bytes")
        true = build_finalize_words(cap.addrs, param_set=p, sk_seed=seed)
    work = {index: public[:, index].astype(np.uint32).copy() for index in range(16)}
    return ExpansionContext(
        cap,
        p,
        public,
        true,
        work,
        {},
        {},
        {},
        seed,
        normalize_stage_chaining(chaining),
        str(primary_method),
        [],
    )


def _hex_value(value: Optional[int], bits: int) -> Optional[str]:
    if value is None:
        return None
    digits = max(1, int(bits) // 4)
    mask = (1 << int(bits)) - 1
    return f"0x{int(value) & mask:0{digits}X}"


def _record_stage_comparison(
    ctx: ExpansionContext,
    plot_config: Optional[PlotConfig],
    *,
    target_word: int,
    intermediate_word: int,
    bytewise_value: int,
    two_byte_value: int,
    selected_method: str,
    selected_value: int,
    true_value: Optional[int],
    value_bits: int = 32,
    target_label: Optional[str] = None,
    stage_domain: str = "word",
    role: str = "primary",
    raw_values: Optional[Mapping[str, Optional[int]]] = None,
) -> Dict[str, Any]:
    """Record one stage's byte-wise vs two-byte outcome.

    ``stage_domain`` documents what the compared value actually is:
      'word'            - the schedule word itself (W9..W12)
      'sigma0'          - sigma0(W_target); the compared value is the inverted
                          word, so it is directly comparable to ground truth
      'sigma0_secret16' - the 16 secret bits of W5
      'exhaustive16'    - 2^16 search, no byte/two-byte split

    ``role`` is 'primary' for the value that actually feeds the recovered
    SK.seed, and 'alternative' for a second, independently measured route to
    the same word (128-bit W9 is recovered both ways).
    """
    mask = (1 << int(value_bits)) - 1
    byte_v = int(bytewise_value) & mask
    two_v = int(two_byte_value) & mask
    sel_v = int(selected_value) & mask
    true_v = None if true_value is None else int(true_value) & mask
    label = target_label or f"W{int(target_word)}"

    payload: Dict[str, Any] = {
        # Backward-compatible key names.
        "target_word": label,
        "intermediate_word": f"W{int(intermediate_word)}",
        "bytewise_recovered": _hex_value(byte_v, value_bits),
        "two_byte_recovered": _hex_value(two_v, value_bits),
        "methods_agree": bool(byte_v == two_v),
        "selected_method": str(selected_method),
        "selected_recovered": _hex_value(sel_v, value_bits),
        "true_word": _hex_value(true_v, value_bits),
        "bytewise_match": None if true_v is None else bool(byte_v == true_v),
        "two_byte_match": None if true_v is None else bool(two_v == true_v),
        # New descriptive keys.
        "target_index": int(target_word),
        "intermediate_index": int(intermediate_word),
        "stage": f"{label}_via_W{int(intermediate_word)}",
        "stage_key": f"{label}@W{int(intermediate_word)}",
        "stage_domain": str(stage_domain),
        "role": str(role),
        "value_bits": int(value_bits),
        "selected_match": None if true_v is None else bool(sel_v == true_v),
        "parameter_set": ctx.param_set,
        "stage_chaining": ctx.chaining,
    }
    if raw_values:
        payload["raw"] = {
            key: _hex_value(value, 32)
            for key, value in dict(raw_values).items()
        }

    ctx.comparisons.append(payload)

    if plot_config is not None and plot_config.save_plots and plot_config.root is not None:
        directory = (
            plot_config.root
            / "message_schedule"
            / f"{label}_via_W{int(intermediate_word)}_comparison"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "comparison.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        payload["comparison_path"] = str(path)

    return payload


def _dual_recover_additive(
    ctx: ExpansionContext,
    *,
    traces: np.ndarray,
    offset: int,
    add_base: np.ndarray,
    stage: str,
    true_key: Optional[int],
    engine_options: Mapping[str, Any],
    beam_width: int,
    candidate_batch: int,
    plot_top: int,
    verbose: bool,
    plot_config: Optional[PlotConfig],
    public_xor: Optional[np.ndarray] = None,
    known_key_bytes: Optional[Mapping[int, int]] = None,
    known_halfwords: Optional[Mapping[int, int]] = None,
) -> Dict[str, Any]:
    """Run byte-wise and two-byte CPA on the same additive stage.

    Returns the raw 32-bit constant recovered by each method. ``public_xor``
    selects the A + (P XOR K) model used by the W5 stage; otherwise the plain
    A + K model is used.
    """
    want_byte = (DUAL_METHOD_MODE != "selected") or (getattr(ctx, "primary_method", "2byte") == "byte")
    want_two = (DUAL_METHOD_MODE != "selected") or (getattr(ctx, "primary_method", "2byte") != "byte")

    byte_value: Optional[int] = None
    two_value: Optional[int] = None
    byte_candidates: List[Any] = []
    two_candidates: List[Any] = []

    if public_xor is None:
        if want_byte:
            byte_value, byte_candidates = recover_add_const_bytewise(
                traces,
                add_base,
                engine_options=engine_options,
                target_name=f"EXP_{ctx.param_set}_{stage}_bytewise",
                true_key=true_key,
                known_key_bytes=known_key_bytes,
                beam_width=beam_width,
                plot_top=plot_top,
                x_offset=offset,
                verbose=verbose,
                plot_config=plot_config,
                plot_attack="message_schedule",
                plot_stage=f"{stage}_bytewise",
            )
        if want_two:
            two_value, two_candidates = recover_add_const_halfwordwise(
                traces,
                add_base,
                engine_options=engine_options,
                target_name=f"EXP_{ctx.param_set}_{stage}_2byte",
                true_key=true_key,
                known_halfwords=known_halfwords,
                beam_width=beam_width,
                candidate_batch=candidate_batch,
                plot_top=plot_top,
                x_offset=offset,
                verbose=verbose,
                plot_config=plot_config,
                plot_attack="message_schedule",
                plot_stage=f"{stage}_2byte",
            )
    else:
        if want_byte:
            byte_value, byte_candidates = recover_add_public_xor_const_bytewise(
                traces,
                add_base,
                public_xor,
                engine_options=engine_options,
                target_name=f"EXP_{ctx.param_set}_{stage}_bytewise",
                true_key=true_key,
                beam_width=beam_width,
                plot_top=plot_top,
                x_offset=offset,
                verbose=verbose,
                plot_config=plot_config,
                plot_attack="message_schedule",
                plot_stage=f"{stage}_bytewise",
            )
        if want_two:
            two_value, two_candidates = recover_add_public_xor_const_halfwordwise(
                traces,
                add_base,
                public_xor,
                engine_options=engine_options,
                target_name=f"EXP_{ctx.param_set}_{stage}_2byte",
                true_key=true_key,
                beam_width=beam_width,
                candidate_batch=candidate_batch,
                plot_top=plot_top,
                x_offset=offset,
                verbose=verbose,
                plot_config=plot_config,
                plot_attack="message_schedule",
                plot_stage=f"{stage}_2byte",
            )

    # When a method is skipped, mirror the computed one so downstream bookkeeping
    # (selected value, comparison records, CSV columns) stays well-formed.
    if byte_value is None:
        byte_value, byte_candidates = two_value, two_candidates
    if two_value is None:
        two_value, two_candidates = byte_value, byte_candidates

    return {
        "byte_raw": int(byte_value) & MASK32,
        "two_byte_raw": int(two_value) & MASK32,
        "byte_candidates": byte_candidates,
        "two_byte_candidates": two_candidates,
        "computed_methods": ("byte" if want_byte else "") + ("+2byte" if want_two else ""),
    }


def _set_constant_word(ctx: ExpansionContext, index: int, value: int) -> None:
    ctx.work[index] = np.full(ctx.cap.addrs.shape[0], np.uint32(int(value) & MASK32), dtype=np.uint32)


def _word(ctx: ExpansionContext, index: int) -> np.ndarray:
    if index < 16:
        return ctx.work[index]
    if index not in ctx.expanded:
        raise RuntimeError(f"W{index} has not been derived")
    return ctx.expanded[index]


def _derive_schedule_word(ctx: ExpansionContext, t: int) -> np.ndarray:
    value = (
        _word(ctx, t - 16)
        + sigma0_32(_word(ctx, t - 15))
        + _word(ctx, t - 7)
        + sigma1_32(_word(ctx, t - 2))
    ).astype(np.uint32)
    ctx.expanded[t] = value
    return value


def _true_secret16_w5(ctx: ExpansionContext) -> Optional[int]:
    if ctx.validation_seed is None:
        return None
    return int.from_bytes(ctx.validation_seed[:2], "big")



def _recover_sigma_word(
    ctx: ExpansionContext,
    *,
    target_initial_word: int,
    output_t: int,
    trace_slice: Optional[slice],
    engine_options: Mapping[str, Any],
    beam_width: int,
    candidate_batch: int,
    plot_top: int,
    primary_method: str,
    verbose: bool,
    plot_config: Optional[PlotConfig] = None,
) -> int:
    """Recover W_target through sigma0(W_target) inside W_output_t.

    Both carry-aware byte-wise CPA and carry-aware two-byte CPA are run on the
    same additive stage; the comparison is stored at the *word* level so it is
    directly comparable with the direct-word stages.
    """
    traces, offset = _stage_traces(ctx.cap, trace_slice)
    base = (
        _word(ctx, output_t - 16)
        + _word(ctx, output_t - 7)
        + sigma1_32(_word(ctx, output_t - 2))
    ).astype(np.uint32)
    true_word = _true_constant(ctx.true_words, target_initial_word)
    true_sigma = (
        None
        if true_word is None
        else int(sigma0_32(np.uint32(true_word)))
    )
    stage = f"W{target_initial_word}_via_W{output_t}"

    dual = _dual_recover_additive(
        ctx,
        traces=traces,
        offset=offset,
        add_base=base,
        stage=stage,
        true_key=true_sigma,
        engine_options=engine_options,
        beam_width=beam_width,
        candidate_batch=candidate_batch,
        plot_top=plot_top,
        verbose=verbose,
        plot_config=plot_config,
    )

    byte_sigma = dual["byte_raw"]
    two_sigma = dual["two_byte_raw"]
    byte_word = sigma0_inv_32(byte_sigma)
    two_word = sigma0_inv_32(two_sigma)
    selected_sigma = byte_sigma if primary_method == "byte" else two_sigma
    selected_word = byte_word if primary_method == "byte" else two_word

    record = _record_stage_comparison(
        ctx,
        plot_config,
        target_word=target_initial_word,
        intermediate_word=output_t,
        bytewise_value=byte_word,
        two_byte_value=two_word,
        selected_method=primary_method,
        selected_value=selected_word,
        true_value=true_word,
        value_bits=32,
        stage_domain="sigma0",
        raw_values={
            "bytewise_sigma0": byte_sigma,
            "two_byte_sigma0": two_sigma,
            "true_sigma0": true_sigma,
        },
    )

    chain_word = _chain_value(ctx, selected_word, true_word)
    _set_constant_word(ctx, target_initial_word, chain_word)
    ctx.expanded[output_t] = (
        base + sigma0_32(np.uint32(chain_word))
    ).astype(np.uint32)
    ctx.recovered[f"w{target_initial_word}"] = int(selected_word) & MASK32
    ctx.stage_info[f"w{target_initial_word}"] = {
        "intermediate": f"W{output_t}",
        "domain": "sigma0",
        "sigma0": selected_sigma,
        "bytewise_word": byte_word,
        "bytewise_sigma0": byte_sigma,
        "bytewise_candidates": dual["byte_candidates"],
        "two_byte_word": two_word,
        "two_byte_sigma0": two_sigma,
        "two_byte_candidates": dual["two_byte_candidates"],
        "selected_method": primary_method,
        "selected_word": int(selected_word) & MASK32,
        "chained_word": int(chain_word) & MASK32,
        "candidates": (
            dual["byte_candidates"]
            if primary_method == "byte"
            else dual["two_byte_candidates"]
        ),
        "comparison": record,
    }
    if verbose:
        print(
            f"W{target_initial_word} comparison: byte=0x{byte_word:08X}, "
            f"2byte=0x{two_word:08X}, "
            f"selected({primary_method})=0x{selected_word:08X}"
        )
        print(
            f"W{output_t}[first]=0x{int(ctx.expanded[output_t][0]):08X}"
        )
    return int(selected_word) & MASK32


def _recover_w5_unified(
    ctx: ExpansionContext,
    *,
    trace_slice: Optional[slice],
    engine_options: Mapping[str, Any],
    beam_width: int,
    candidate_batch: int,
    plot_top: int,
    primary_method: str,
    verbose: bool,
    plot_config: Optional[PlotConfig] = None,
) -> int:
    """Recover the 16 secret bits of W5 through W20.

    W20 = W4 + sigma0(W5) + W13 + sigma1(W18) and sigma0 is linear over XOR,
    so sigma0(W5) = sigma0(public high16) XOR sigma0(secret16). Both the
    byte-wise and the two-byte carry-aware CPA are executed on that model.
    """
    traces, offset = _stage_traces(ctx.cap, trace_slice)
    public_hi = (
        ctx.public_words[:, 5] & np.uint32(0xFFFF0000)
    ).astype(np.uint32)
    public_sigma = sigma0_32(public_hi)
    base = (
        _word(ctx, 4)
        + _word(ctx, 13)
        + sigma1_32(_word(ctx, 18))
    ).astype(np.uint32)
    true_secret = _true_secret16_w5(ctx)
    true_key = (
        None
        if true_secret is None
        else int(sigma0_32(np.uint32(true_secret)))
    )

    dual = _dual_recover_additive(
        ctx,
        traces=traces,
        offset=offset,
        add_base=base,
        stage="W5_via_W20",
        true_key=true_key,
        engine_options=engine_options,
        beam_width=beam_width,
        candidate_batch=candidate_batch,
        plot_top=plot_top,
        verbose=verbose,
        plot_config=plot_config,
        public_xor=public_sigma,
    )

    byte_sigma = dual["byte_raw"]
    two_sigma = dual["two_byte_raw"]
    byte_secret16 = sigma0_inv_32(byte_sigma) & 0xFFFF
    two_secret16 = sigma0_inv_32(two_sigma) & 0xFFFF
    selected_secret16 = (
        byte_secret16 if primary_method == "byte" else two_secret16
    )

    record = _record_stage_comparison(
        ctx,
        plot_config,
        target_word=5,
        intermediate_word=20,
        bytewise_value=byte_secret16,
        two_byte_value=two_secret16,
        selected_method=primary_method,
        selected_value=selected_secret16,
        true_value=true_secret,
        value_bits=16,
        target_label="W5",
        stage_domain="sigma0_secret16",
        raw_values={
            "bytewise_sigma0_secret": byte_sigma,
            "two_byte_sigma0_secret": two_sigma,
            "true_sigma0_secret": true_key,
        },
    )

    chain_secret16 = (
        int(true_secret) & 0xFFFF
        if ctx.chaining == "assume_previous_success" and true_secret is not None
        else int(selected_secret16) & 0xFFFF
    )
    ctx.work[5] = (
        public_hi | np.uint32(chain_secret16)
    ).astype(np.uint32)
    ctx.expanded[20] = (
        base + sigma0_32(ctx.work[5])
    ).astype(np.uint32)
    ctx.recovered["w5_secret16"] = int(selected_secret16) & 0xFFFF
    ctx.stage_info["w5"] = {
        "intermediate": "W20",
        "domain": "sigma0_secret16",
        "sigma0_secret": byte_sigma if primary_method == "byte" else two_sigma,
        "bytewise_secret16": int(byte_secret16),
        "bytewise_sigma0_secret": byte_sigma,
        "bytewise_candidates": dual["byte_candidates"],
        "two_byte_secret16": int(two_secret16),
        "two_byte_sigma0_secret": two_sigma,
        "two_byte_candidates": dual["two_byte_candidates"],
        "selected_method": primary_method,
        "selected_secret16": int(selected_secret16),
        "chained_secret16": int(chain_secret16),
        "candidates": (
            dual["byte_candidates"]
            if primary_method == "byte"
            else dual["two_byte_candidates"]
        ),
        "comparison": record,
    }
    if verbose:
        print(
            f"W5 secret16 comparison: byte=0x{byte_secret16:04X}, "
            f"2byte=0x{two_secret16:04X}, "
            f"selected({primary_method})=0x{selected_secret16:04X}"
        )
        print(f"W20[first]=0x{int(ctx.expanded[20][0]):08X}")
    return int(selected_secret16) & 0xFFFF


def _recover_direct_word(
    ctx: ExpansionContext,
    *,
    target_word: int,
    output_t: int,
    trace_slice: Optional[slice],
    engine_options: Mapping[str, Any],
    beam_width: int,
    candidate_batch: int,
    plot_top: int,
    primary_method: str,
    verbose: bool,
    plot_config: Optional[PlotConfig] = None,
    commit: bool = True,
    role: str = "primary",
    stage_info_key: Optional[str] = None,
) -> int:
    """Recover a secret schedule word that enters W_output_t additively.

    W_t = W_(t-16) + sigma0(W_(t-15)) + W_(t-7) + sigma1(W_(t-2)) with
    target = W_(t-7), so everything except the target is already known and the
    target is a plain additive constant. For the last secret-bearing word the
    public 0x8000 SHA-2 padding in its low 16 bits is supplied as known, so
    only the secret high half is searched by either method.
    """
    traces, offset = _stage_traces(ctx.cap, trace_slice)
    base = (
        _word(ctx, output_t - 16)
        + sigma0_32(_word(ctx, output_t - 15))
        + sigma1_32(_word(ctx, output_t - 2))
    ).astype(np.uint32)
    true_word = _true_constant(ctx.true_words, target_word)
    stage = f"W{target_word}_via_W{output_t}"

    is_last_secret = target_word == _compression_last_secret_word(ctx.param_set)
    known_key_bytes = {0: 0x00, 1: 0x80} if is_last_secret else None
    known_halfwords = {0: 0x8000} if is_last_secret else None

    dual = _dual_recover_additive(
        ctx,
        traces=traces,
        offset=offset,
        add_base=base,
        stage=stage,
        true_key=true_word,
        engine_options=engine_options,
        beam_width=beam_width,
        candidate_batch=candidate_batch,
        plot_top=plot_top,
        verbose=verbose,
        plot_config=plot_config,
        known_key_bytes=known_key_bytes,
        known_halfwords=known_halfwords,
    )

    byte_word = dual["byte_raw"]
    two_word = dual["two_byte_raw"]
    selected_word = byte_word if primary_method == "byte" else two_word

    record = _record_stage_comparison(
        ctx,
        plot_config,
        target_word=target_word,
        intermediate_word=output_t,
        bytewise_value=byte_word,
        two_byte_value=two_word,
        selected_method=primary_method,
        selected_value=selected_word,
        true_value=true_word,
        value_bits=32,
        stage_domain="word",
        role=role,
    )

    info = {
        "intermediate": f"W{output_t}",
        "domain": "word",
        "source": f"direct_w{output_t}",
        "role": role,
        "public_low16_known": bool(is_last_secret),
        "bytewise_word": byte_word,
        "bytewise_candidates": dual["byte_candidates"],
        "two_byte_word": two_word,
        "two_byte_candidates": dual["two_byte_candidates"],
        "selected_method": primary_method,
        "selected_word": int(selected_word) & MASK32,
        "comparison": record,
    }
    ctx.stage_info[stage_info_key or f"w{target_word}"] = info

    if commit:
        chain_word = _chain_value(ctx, selected_word, true_word)
        _set_constant_word(ctx, target_word, chain_word)
        ctx.expanded[output_t] = (base + np.uint32(chain_word)).astype(np.uint32)
        ctx.recovered[f"w{target_word}"] = int(selected_word) & MASK32
        info["chained_word"] = int(chain_word) & MASK32

    if verbose:
        print(
            f"W{target_word} via W{output_t} comparison: "
            f"byte=0x{byte_word:08X}, 2byte=0x{two_word:08X}, "
            f"selected({primary_method})=0x{selected_word:08X}"
        )
        if commit:
            print(f"W{output_t}[first]=0x{int(ctx.expanded[output_t][0]):08X}")
    return int(selected_word) & MASK32

def _search_128_w9(
    ctx: ExpansionContext,
    *,
    trace_slice: Optional[slice],
    engine_options: Mapping[str, Any],
    candidate_batch: int,
    top_n: int,
    plot_top: int,
    verbose: bool,
    plot_config: Optional[PlotConfig] = None,
    commit: bool = True,
    role: str = "primary",
    stage_info_key: Optional[str] = None,
) -> int:
    """Recover 128-bit W9 through W18 and plot the exhaustive 16-bit stage."""
    traces, offset = _stage_traces(ctx.cap, trace_slice)
    engine = CorrelationEngine(traces, **dict(engine_options))

    B16 = (
        _word(ctx, 0)
        + sigma0_32(_word(ctx, 1))
        + sigma1_32(_word(ctx, 14))
    ).astype(np.uint32)
    A18 = (
        _word(ctx, 2)
        + sigma0_32(_word(ctx, 3))
        + _word(ctx, 11)
    ).astype(np.uint32)

    values = np.arange(65536, dtype=np.uint32)
    scores = np.zeros(65536, dtype=np.float32)
    pois = np.zeros(65536, dtype=np.int32)
    batch = max(1, int(candidate_batch))

    for start in range(0, 65536, batch):
        stop = min(start + batch, 65536)
        hi = values[start:stop]
        w9 = (
            (hi << np.uint32(16)) | np.uint32(0x8000)
        ).astype(np.uint32)
        W16 = (B16[:, None] + w9[None, :]).astype(np.uint32)
        W18 = (A18[:, None] + sigma1_32(W16)).astype(np.uint32)
        H = hw_u32(W18)
        block_scores, block_pois = engine.score(H)
        scores[start:stop] = block_scores
        pois[start:stop] = block_pois + int(offset)
        if verbose and start % (batch * 16) == 0:
            print(f"128 W9 2^16: processed {start:5d}/65536")

    order = np.argsort(scores)[::-1]
    best_hi = int(order[0])
    best_w9 = ((best_hi << 16) | 0x8000) & MASK32
    true_w9 = _true_constant(ctx.true_words, 9)
    true_hi = None if true_w9 is None else (true_w9 >> 16) & 0xFFFF

    if verbose:
        print(
            f"best W9=0x{best_w9:08X}, "
            f"corr={float(scores[best_hi]):.6f}, "
        )
        if true_hi is not None:
            true_rank = int(np.where(order == true_hi)[0][0]) + 1
            print(
                f"true W9=0x{true_w9:08X}, "
                f"corr={float(scores[true_hi]):.6f}, "
            )
        for rank, candidate in enumerate(
            order[:max(1, int(top_n))],
            1,
        ):
            c = int(candidate)
            mark = " OK" if true_hi is not None and c == true_hi else ""
            print(
                f"W9=0x{((c << 16) | 0x8000):08X}, "
                f"corr={float(scores[c]):.6f}, "
            )

    if plot_top > 0:
        top_count = max(1, int(plot_top))
        displayed_values = [int(v) for v in order[:top_count]]
        if true_hi is not None and true_hi not in displayed_values:
            displayed_values.append(int(true_hi))

        hypotheses: List[np.ndarray] = []
        labels: List[str] = []
        correct_index: Optional[int] = None
        for index, hi_value in enumerate(displayed_values):
            w9_value = np.uint32((hi_value << 16) | 0x8000)
            w16 = (B16 + w9_value).astype(np.uint32)
            w18 = (A18 + sigma1_32(w16)).astype(np.uint32)
            hypotheses.append(hw_u32(w18))
            rank_pos = int(np.where(order == hi_value)[0][0]) + 1
            label = _legend_label_word(
                int(w9_value),
                float(scores[hi_value]),
            )
            # Only the appended correct candidate (outside the top-N) needs an
            # explicit rank; the top-N entries are listed in rank order.
            labels.append(
                _legend_label_with_rank(
                    label, rank_pos if index >= top_count else None
                )
            )
            if true_hi is not None and hi_value == true_hi:
                correct_index = index

        _plot_curves(
            engine,
            hypotheses,
            labels,
            correct_index=correct_index,
            x_offset=offset,
            title=f"{ctx.param_set} message-schedule W9 via W18",
            plot_config=plot_config,
            attack="message_schedule",
            stage="W9_via_W18",
            filename="exhaustive_16bit.png",
        )

    record = _record_stage_comparison(
        ctx,
        plot_config,
        target_word=9,
        intermediate_word=18,
        bytewise_value=best_w9,
        two_byte_value=best_w9,
        selected_method="exhaustive16",
        selected_value=best_w9,
        true_value=true_w9,
        value_bits=32,
        stage_domain="exhaustive16",
        role=role,
    )

    info = {
        "intermediate": "W18",
        "domain": "exhaustive16",
        "source": "exhaustive_w18",
        "role": role,
        "selected_method": "exhaustive16",
        "selected_word": best_w9,
        "scores": scores,
        "pois": pois,
        "order": order,
        "comparison": record,
    }
    ctx.stage_info[stage_info_key or "w9"] = info

    if commit:
        chain_w9 = _chain_value(ctx, best_w9, true_w9)
        _set_constant_word(ctx, 9, chain_w9)
        ctx.recovered["w9"] = best_w9
        info["chained_word"] = chain_w9
    return best_w9

def assemble_expansion_seed(ctx: ExpansionContext) -> bytes:
    bits = security_bits(ctx.param_set)
    if bits == 128:
        required = ("w5_secret16", "w6", "w7", "w8", "w9")
        missing = [name for name in required if name not in ctx.recovered]
        if missing: raise RuntimeError("missing recovered values: " + ", ".join(missing))
        return (
            int(ctx.recovered["w5_secret16"]).to_bytes(2, "big")
            + word_to_be_bytes(ctx.recovered["w6"])
            + word_to_be_bytes(ctx.recovered["w7"])
            + word_to_be_bytes(ctx.recovered["w8"])
            + bytes([(ctx.recovered["w9"] >> 24) & 0xFF, (ctx.recovered["w9"] >> 16) & 0xFF])
        )
    if bits == 192:
        required = ("w5_secret16", "w6", "w7", "w8", "w9", "w10", "w11")
        missing = [name for name in required if name not in ctx.recovered]
        if missing: raise RuntimeError("missing recovered values: " + ", ".join(missing))
        return (
            int(ctx.recovered["w5_secret16"]).to_bytes(2, "big")
            + b"".join(word_to_be_bytes(ctx.recovered[name]) for name in ("w6", "w7", "w8", "w9", "w10"))
            + bytes([(ctx.recovered["w11"] >> 24) & 0xFF, (ctx.recovered["w11"] >> 16) & 0xFF])
        )
    required = ("w5_secret16", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13")
    missing = [name for name in required if name not in ctx.recovered]
    if missing: raise RuntimeError("missing recovered values: " + ", ".join(missing))
    return (
        int(ctx.recovered["w5_secret16"]).to_bytes(2, "big")
        + b"".join(word_to_be_bytes(ctx.recovered[name]) for name in ("w6", "w7", "w8", "w9", "w10", "w11", "w12"))
        + bytes([(ctx.recovered["w13"] >> 24) & 0xFF, (ctx.recovered["w13"] >> 16) & 0xFF])
    )



# ---------------------------------------------------------------------------
# Compression F_32 CPA
# ---------------------------------------------------------------------------
def _compress_one_block(initial_state: np.ndarray, block64: bytes) -> np.ndarray:
    if len(block64) != 64:
        raise ValueError("block must be exactly 64 bytes")
    W = [int.from_bytes(block64[4*i:4*i+4], "big") for i in range(16)]
    for t in range(16, 64):
        W.append((W[t-16] + _sigma0_scalar(W[t-15]) + W[t-7] + int(sigma1_32(np.uint32(W[t-2])))) & MASK32)
    a, b, c, d, e, f, g, h = [int(v) for v in initial_state]
    for t in range(64):
        T1 = (h + int(Sigma1_32(np.uint32(e))) + int(Ch(np.uint32(e), np.uint32(f), np.uint32(g))) + int(SHA256_K[t]) + W[t]) & MASK32
        T2 = (int(Sigma0_32(np.uint32(a))) + int(Maj(np.uint32(a), np.uint32(b), np.uint32(c)))) & MASK32
        h, g, f, e, d, c, b, a = g, f, e, (d + T1) & MASK32, c, b, a, (T1 + T2) & MASK32
    return np.array([
        (int(initial_state[0]) + a) & MASK32,
        (int(initial_state[1]) + b) & MASK32,
        (int(initial_state[2]) + c) & MASK32,
        (int(initial_state[3]) + d) & MASK32,
        (int(initial_state[4]) + e) & MASK32,
        (int(initial_state[5]) + f) & MASK32,
        (int(initial_state[6]) + g) & MASK32,
        (int(initial_state[7]) + h) & MASK32,
    ], dtype=np.uint32)


def seeded_state_from_pk_seed(param_set: str | int, pk_seed: bytes) -> np.ndarray:
    p = normalize_param_set(param_set)
    n = PARAM_N[p]
    pk = bytes(pk_seed)
    if len(pk) != n:
        raise ValueError(f"PK.seed for {p} must contain {n} bytes")
    return _compress_one_block(SHA256_IV, pk + bytes(64 - n))


def _broadcast_state(state: np.ndarray, rows: int) -> Dict[str, np.ndarray]:
    return {name: np.full(rows, np.uint32(value), dtype=np.uint32) for name, value in zip("abcdefgh", state)}


def _round_arrays(state: Dict[str, np.ndarray], word: np.ndarray, k: int):
    base_t1 = (state["h"] + Sigma1_32(state["e"]) + Ch(state["e"], state["f"], state["g"]) + np.uint32(k)).astype(np.uint32)
    T1 = (base_t1 + np.asarray(word, dtype=np.uint32)).astype(np.uint32)
    T2 = (Sigma0_32(state["a"]) + Maj(state["a"], state["b"], state["c"])).astype(np.uint32)
    new_state = {
        "h": state["g"].copy(), "g": state["f"].copy(), "f": state["e"].copy(),
        "e": (state["d"] + T1).astype(np.uint32), "d": state["c"].copy(),
        "c": state["b"].copy(), "b": state["a"].copy(), "a": (T1 + T2).astype(np.uint32),
    }
    return new_state, T1, T2, base_t1


def _compression_last_secret_word(param_set: str) -> int:
    return {128: 9, 192: 11, 256: 13}[security_bits(param_set)]


def _known_word_halfwords_for_compression(
    t: int,
    last_secret: int,
    public_words: np.ndarray,
) -> Dict[int, Union[int, np.ndarray]]:
    """Known 16-bit halves for compression-round CPA.

    Halfword index 0 is bits [15:0] and index 1 is bits [31:16].

    * W5: high16 is the final two public ADRS^c bytes, low16 is secret.
    * Final secret-bearing word: low16 is the public SHA-256 padding 0x8000,
      high16 is secret.
    * Intermediate secret-bearing words: both halves are secret.
    """
    if t == 5:
        return {
            1: (
                (public_words[:, 5] >> np.uint32(16))
                & np.uint32(0xFFFF)
            ).astype(np.uint32)
        }
    if t == last_secret:
        return {0: 0x8000}
    return {}


def _spec_halfword_array(
    spec: Union[int, np.ndarray],
    rows: int,
) -> np.ndarray:
    if np.isscalar(spec):
        return np.full(rows, int(spec) & 0xFFFF, dtype=np.uint32)
    array = np.asarray(spec, dtype=np.uint32).reshape(-1)
    if array.size != rows:
        raise ValueError("known-halfword array length mismatch")
    return array & np.uint32(0xFFFF)


def _spec_part_array(spec, rows: int, part_bits: int) -> np.ndarray:
    mask = (1 << part_bits) - 1
    if np.isscalar(spec):
        return np.full(rows, int(spec) & mask, dtype=np.uint32)
    array = np.asarray(spec, dtype=np.uint32).reshape(-1)
    if array.size != rows:
        raise ValueError("known-part array length mismatch")
    return array & np.uint32(mask)


def recover_add_word_with_known_parts_bytewise(
    traces: np.ndarray,
    add_base: np.ndarray,
    *,
    known_halfwords: Mapping[int, Union[int, np.ndarray]],
    true_word_array: Optional[np.ndarray],
    engine_options: Mapping[str, Any],
    target_name: str,
    beam_width: int,
    candidate_batch: int,
    plot_top: int,
    x_offset: int,
    verbose: bool,
    plot_config: Optional[PlotConfig] = None,
    plot_attack: str = "compression",
    plot_stage: Optional[str] = None,
) -> Tuple[np.ndarray, List[HalfwordCandidate]]:
    """8-bit counterpart of ``recover_add_word_with_known_halfwords``.

    Recovers W in A+W with carry-aware BYTE-wise CPA (4 parts, low->high).
    ``known_halfwords`` uses the same 16-bit indexing as the half-word version
    (0 = low16, 1 = high16); a known half-word fixes its two bytes and only
    propagates carry, so the public padding / public address halves behave
    exactly as before.  Honours CUMULATIVE_HW.
    """
    T = np.asarray(traces, dtype=np.float32)
    A = np.asarray(add_base, dtype=np.uint32).reshape(-1)
    rows = A.size
    if T.ndim != 2 or T.shape[0] != rows:
        raise ValueError("trace/add_base row mismatch")

    engine = CorrelationEngine(T, **dict(engine_options))
    known_half = {int(k): v for k, v in dict(known_halfwords).items()}
    # expand 16-bit known specs into byte specs
    known_byte: Dict[int, np.ndarray] = {}
    for h, spec in known_half.items():
        arr = _spec_halfword_array(spec, rows)
        known_byte[2 * h] = (arr & np.uint32(0xFF)).astype(np.uint32)
        known_byte[2 * h + 1] = ((arr >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint32)

    width = max(1, int(beam_width))
    stage_name = str(plot_stage or target_name)

    paths: List[Dict[str, Any]] = [{
        "bytes": [None, None, None, None],
        "score": 0.0,
        "carry": np.zeros(rows, dtype=np.uint32),
        "corr_history": [],
        "poi_history": [],
        "last_corr": 0.0,
        "last_poi": -1,
    }]

    true_first = (
        None
        if true_word_array is None
        else int(np.asarray(true_word_array, dtype=np.uint32).reshape(-1)[0])
    )

    for byte_index in range(4):
        shift = np.uint32(8 * byte_index)

        if byte_index in known_byte:
            byte_array = known_byte[byte_index]
            next_paths = []
            for parent in paths:
                base_byte = (((A >> shift) & np.uint32(0xFF)) + parent["carry"]).astype(np.uint32)
                b = list(parent["bytes"]); b[byte_index] = int(byte_array[0])
                new = dict(parent)
                new["bytes"] = b
                new["carry"] = ((base_byte + byte_array) >> np.uint32(8)).astype(np.uint32)
                next_paths.append(new)
            paths = next_paths
            if verbose:
                detail = (f"0x{int(byte_array[0]):02X}" if np.all(byte_array == byte_array[0])
                          else "per-trace public value")
                print(f"[{target_name}] byte{byte_index} fixed to {detail} (public)")
            continue

        records: List[Tuple[float, int, int, float, int]] = []
        for parent_index, parent in enumerate(paths):
            base_byte = (((A >> shift) & np.uint32(0xFF)) + parent["carry"]).astype(np.uint32)
            guesses = np.arange(256, dtype=np.uint32)
            output = (base_byte[:, None] + guesses[None, :]) & np.uint32(0xFF)
            H = HW8[output.astype(np.uint8)]
            if CUMULATIVE_HW and byte_index > 0:
                H = H.astype(np.float32) + _lower_parts_hw(A, parent, byte_index, 8)[:, None]
            scores, pois = engine.score(H)
            for guess in range(256):
                records.append((
                    float(parent["score"]) + float(scores[guess]),
                    parent_index, int(guess), float(scores[guess]), int(pois[guess]),
                ))

        records.sort(key=lambda row: row[0], reverse=True)
        next_paths = []
        for total, parent_index, guess, last_corr, poi in records[:width]:
            parent = paths[parent_index]
            base_byte = (((A >> shift) & np.uint32(0xFF)) + parent["carry"]).astype(np.uint32)
            b = list(parent["bytes"]); b[byte_index] = int(guess)
            next_paths.append({
                "bytes": b,
                "score": float(total),
                "carry": ((base_byte + np.uint32(guess)) >> np.uint32(8)).astype(np.uint32),
                "corr_history": list(parent["corr_history"]) + [float(last_corr)],
                "poi_history": list(parent["poi_history"]) + [int(x_offset + poi)],
                "last_corr": float(last_corr),
                "last_poi": int(x_offset + poi),
            })
        paths = next_paths

        if verbose:
            print(f"[{target_name}] byte{byte_index}, backend={engine.backend}, cached={engine.cached}")
            for rank, path in enumerate(paths, 1):
                partial = 0
                for idx in range(byte_index + 1):
                    if path["bytes"][idx] is not None:
                        partial |= int(path["bytes"][idx]) << (8 * idx)
                mark = ""
                if true_first is not None:
                    ok = all(
                        (idx in known_byte) or
                        int(path["bytes"][idx]) == ((true_first >> (8 * idx)) & 0xFF)
                        for idx in range(byte_index + 1)
                    )
                    mark = " OK" if ok else ""
                print(f"  [{rank}] first-partial=0x{partial:08X} corr={float(path['last_corr']):.6f} "
                      f"score={float(path['score']):.6f} POI={int(path['last_poi'])}{mark}")

        # Save a per-byte candidate figure, exactly like the message-schedule
        # (expansion) byte-wise attack and the 16-bit compression attack do.
        # Without this block the byte-wise compression path produced no plots,
        # so the compression_* folders were never created.
        if plot_top > 0:
            displayed = sorted(
                paths,
                key=lambda item: float(item["score"]),
                reverse=True,
            )[:max(1, int(plot_top))]

            def _is_true_path_byte(path: Mapping[str, Any]) -> bool:
                if true_first is None:
                    return False
                for idx in range(byte_index + 1):
                    if idx in known_byte:
                        continue
                    if path["bytes"][idx] is None:
                        return False
                    if int(path["bytes"][idx]) != ((true_first >> (8 * idx)) & 0xFF):
                        return False
                return True

            hypotheses: List[np.ndarray] = []
            labels: List[str] = []
            correct_index: Optional[int] = None

            for index, path in enumerate(displayed):
                # Rebuild the carry into byte_index from the already
                # fixed/recovered lower bytes (public halves included).
                carry = np.zeros(rows, dtype=np.uint32)
                for lower in range(byte_index):
                    if lower in known_byte:
                        low_add = known_byte[lower]
                    else:
                        low_add = np.full(
                            rows, np.uint32(path["bytes"][lower]), dtype=np.uint32
                        )
                    bbase = (
                        ((A >> np.uint32(8 * lower)) & np.uint32(0xFF)) + carry
                    ).astype(np.uint32)
                    carry = ((bbase + low_add) >> np.uint32(8)).astype(np.uint32)

                bbase = (
                    ((A >> np.uint32(8 * byte_index)) & np.uint32(0xFF)) + carry
                ).astype(np.uint32)
                out = (
                    (bbase + np.uint32(path["bytes"][byte_index])) & np.uint32(0xFF)
                ).astype(np.uint8)
                hypotheses.append(HW8[out])

                # Representative full word for the legend: known bytes take the
                # first element of their (possibly per-trace) public value;
                # undetermined high bytes show as 0x00.
                representative = 0
                for idx in range(4):
                    if idx in known_byte:
                        value = int(known_byte[idx][0]) & 0xFF
                    elif path["bytes"][idx] is not None:
                        value = int(path["bytes"][idx]) & 0xFF
                    else:
                        value = 0
                    representative |= value << (8 * idx)
                labels.append(
                    _legend_label_word(
                        representative,
                        float(path["last_corr"]),
                        float(path["score"]),
                    )
                )

                if _is_true_path_byte(path):
                    correct_index = index

            _plot_curves(
                engine,
                hypotheses,
                labels,
                correct_index=correct_index,
                x_offset=x_offset,
                title=f"{target_name} / byte {byte_index}",
                plot_config=plot_config,
                attack=plot_attack,
                stage=stage_name,
                filename=f"byte_{byte_index}.png",
            )

    best = max(paths, key=lambda item: float(item["score"]))
    word_scalar = 0
    for idx in range(4):
        word_scalar |= (int(best["bytes"][idx]) & 0xFF) << (8 * idx)

    # per-trace word array: public parts may vary per trace
    recovered = np.full(rows, np.uint32(word_scalar), dtype=np.uint32)
    for idx, arr in known_byte.items():
        recovered = (recovered & np.uint32(~(0xFF << (8 * idx)) & MASK32)) | (arr << np.uint32(8 * idx))

    candidates = [
        HalfwordCandidate(
            word=sum((int(p["bytes"][i]) & 0xFF) << (8 * i) for i in range(4)),
            score=float(p["score"]),
            halfwords_lsb=[
                (int(p["bytes"][0]) & 0xFF) | ((int(p["bytes"][1]) & 0xFF) << 8),
                (int(p["bytes"][2]) & 0xFF) | ((int(p["bytes"][3]) & 0xFF) << 8),
            ],
            corr_history=list(p["corr_history"]),
            poi_history=list(p["poi_history"]),
            last_corr=float(p["last_corr"]),
            last_poi=int(p["last_poi"]),
        )
        for p in sorted(paths, key=lambda item: float(item["score"]), reverse=True)
    ]
    return recovered.astype(np.uint32), candidates


def recover_add_word_with_known_halfwords(
    traces: np.ndarray,
    add_base: np.ndarray,
    *,
    known_halfwords: Mapping[int, Union[int, np.ndarray]],
    true_word_array: Optional[np.ndarray],
    engine_options: Mapping[str, Any],
    target_name: str,
    beam_width: int,
    candidate_batch: int,
    plot_top: int,
    x_offset: int,
    verbose: bool,
    plot_config: Optional[PlotConfig] = None,
    plot_attack: str = "compression",
    plot_stage: Optional[str] = None,
) -> Tuple[np.ndarray, List[HalfwordCandidate]]:
    """Recover W in A+W using carry-aware 16-bit CPA.

    The search proceeds low16 -> high16.  Any public halfword is fixed and
    participates in carry propagation without creating a CPA search stage.

    This is the compression-round counterpart of
    ``recover_add_const_halfwordwise`` used by the message-schedule attack.
    """
    T = np.asarray(traces, dtype=np.float32)
    A = np.asarray(add_base, dtype=np.uint32).reshape(-1)
    rows = A.size
    if T.ndim != 2 or T.shape[0] != rows:
        raise ValueError("trace/add_base row mismatch")

    engine = CorrelationEngine(T, **dict(engine_options))
    known = {int(k): v for k, v in dict(known_halfwords).items()}
    width = max(1, int(beam_width))
    batch = max(1, int(candidate_batch))
    stage_name = str(plot_stage or target_name)

    paths: List[Dict[str, Any]] = [{
        "halfwords": [None, None],
        "known_specs": {},
        "score": 0.0,
        "carry": np.zeros(rows, dtype=np.uint32),
        "corr_history": [],
        "poi_history": [],
        "last_corr": 0.0,
        "last_poi": -1,
    }]

    true_first = (
        None
        if true_word_array is None
        else int(np.asarray(true_word_array, dtype=np.uint32).reshape(-1)[0])
    )

    for half_index in range(2):
        part = "low16" if half_index == 0 else "high16"

        # Public halfword: fix it and only propagate the carry.
        if half_index in known:
            half_array = _spec_halfword_array(known[half_index], rows)
            next_paths: List[Dict[str, Any]] = []
            for parent in paths:
                base_half = (
                    (
                        (A >> np.uint32(16 * half_index))
                        & np.uint32(0xFFFF)
                    )
                    + parent["carry"]
                ).astype(np.uint32)

                halfwords = list(parent["halfwords"])
                halfwords[half_index] = int(half_array[0])
                new = dict(parent)
                new["halfwords"] = halfwords
                new["known_specs"] = dict(parent["known_specs"])
                new["known_specs"][half_index] = half_array
                new["carry"] = (
                    (base_half + half_array) >> np.uint32(16)
                ).astype(np.uint32)
                next_paths.append(new)
            paths = next_paths

            if verbose:
                if np.all(half_array == half_array[0]):
                    detail = f"0x{int(half_array[0]):04X}"
                else:
                    detail = "per-trace public value"
                print(
                    f"[{target_name}] {part} fixed to {detail} (public)"
                )
            continue

        # Secret halfword: exhaustively rank all 2^16 candidates.
        records: List[Tuple[float, int, int, float, int]] = []
        parents = paths  # beam entries that generated ``records``

        for parent_index, parent in enumerate(paths):
            base_half = (
                (
                    (A >> np.uint32(16 * half_index))
                    & np.uint32(0xFFFF)
                )
                + parent["carry"]
            ).astype(np.uint32)

            for guess_start in range(0, 65536, batch):
                guess_stop = min(guess_start + batch, 65536)
                guesses = np.arange(
                    guess_start,
                    guess_stop,
                    dtype=np.uint32,
                )
                output = (
                    base_half[:, None] + guesses[None, :]
                ) & np.uint32(0xFFFF)
                H = hw_u16(output)
                if CUMULATIVE_HW and half_index > 0:
                    H = H.astype(np.float32) + _lower_parts_hw(A, parent, half_index, 16)[:, None]
                scores, pois = engine.score(H)

                for local, guess in enumerate(
                    range(guess_start, guess_stop)
                ):
                    records.append((
                        float(parent["score"]) + float(scores[local]),
                        parent_index,
                        int(guess),
                        float(scores[local]),
                        int(pois[local]),
                    ))

        records.sort(key=lambda row: row[0], reverse=True)

        next_paths = []
        for total, parent_index, guess, last_corr, poi in records[:width]:
            parent = paths[parent_index]
            base_half = (
                (
                    (A >> np.uint32(16 * half_index))
                    & np.uint32(0xFFFF)
                )
                + parent["carry"]
            ).astype(np.uint32)

            halfwords = list(parent["halfwords"])
            halfwords[half_index] = int(guess)
            next_paths.append({
                "halfwords": halfwords,
                "known_specs": dict(parent["known_specs"]),
                "score": float(total),
                "carry": (
                    (base_half + np.uint32(guess)) >> np.uint32(16)
                ).astype(np.uint32),
                "corr_history": (
                    list(parent["corr_history"]) + [float(last_corr)]
                ),
                "poi_history": (
                    list(parent["poi_history"]) + [int(x_offset + poi)]
                ),
                "last_corr": float(last_corr),
                "last_poi": int(x_offset + poi),
            })
        paths = next_paths

        if verbose:
            print(
                f"[{target_name}] {part}, "
                f"backend={engine.backend}, cached={engine.cached}"
            )
            for rank, path in enumerate(paths, 1):
                mark = ""
                if true_first is not None:
                    correct = True
                    for idx in range(half_index + 1):
                        if idx in known:
                            continue
                        expected = (
                            true_first >> (16 * idx)
                        ) & 0xFFFF
                        if int(path["halfwords"][idx]) != expected:
                            correct = False
                            break
                    mark = " OK" if correct else ""

                partial = 0
                for idx in range(half_index + 1):
                    if path["halfwords"][idx] is not None:
                        partial |= (
                            int(path["halfwords"][idx])
                            << (16 * idx)
                        )
                print(
                    f"  [{rank}] first-partial=0x{partial:08X} "
                    f"corr={float(path['last_corr']):.6f} "
                    f"score={float(path['score']):.6f} "
                    f"POI={int(path['last_poi'])}{mark}"
                )

        if plot_top > 0:
            displayed = sorted(
                paths,
                key=lambda item: float(item["score"]),
                reverse=True,
            )[:max(1, int(plot_top))]
            display_ranks: List[Optional[int]] = [None] * len(displayed)

            def _is_true_path(path: Mapping[str, Any]) -> bool:
                if true_first is None:
                    return False
                for idx in range(half_index + 1):
                    if idx in known:
                        continue
                    expected = (true_first >> (16 * idx)) & 0xFFFF
                    if int(path["halfwords"][idx]) != expected:
                        return False
                return True

            # Always draw the correct candidate and state its rank when it is
            # not among the top-``plot_top`` beam entries.
            if true_first is not None and not any(_is_true_path(p) for p in displayed):
                true_halves = {
                    idx: (true_first >> (16 * idx)) & 0xFFFF
                    for idx in range(2)
                    if idx not in known
                }
                hit = _true_path_record(records, parents, half_index, true_halves, known)
                if hit is not None:
                    rank, (total, parent_index, guess, last_corr, poi) = hit
                    parent = parents[int(parent_index)]
                    halfwords = list(parent["halfwords"])
                    halfwords[half_index] = int(guess)
                    displayed.append({
                        "halfwords": halfwords,
                        "known_specs": dict(parent["known_specs"]),
                        "score": float(total),
                        "last_corr": float(last_corr),
                        "last_poi": int(x_offset + poi),
                    })
                    display_ranks.append(int(rank))

            hypotheses: List[np.ndarray] = []
            labels: List[str] = []
            correct_index: Optional[int] = None

            for display_index, path in enumerate(displayed):
                carry = np.zeros(rows, dtype=np.uint32)

                # Reconstruct the carry from the already known/recovered low
                # half when plotting the high16 stage.
                if half_index == 1:
                    low_base = (
                        A & np.uint32(0xFFFF)
                    ).astype(np.uint32)
                    if 0 in known:
                        low_add = _spec_halfword_array(known[0], rows)
                    else:
                        low_add = np.full(
                            rows,
                            np.uint32(path["halfwords"][0]),
                            dtype=np.uint32,
                        )
                    carry = (
                        (low_base + low_add) >> np.uint32(16)
                    ).astype(np.uint32)

                base_half = (
                    (
                        (A >> np.uint32(16 * half_index))
                        & np.uint32(0xFFFF)
                    )
                    + carry
                ).astype(np.uint32)
                output = (
                    base_half
                    + np.uint32(path["halfwords"][half_index])
                ) & np.uint32(0xFFFF)
                hypotheses.append(hw_u16(output))

                if half_index == 0:
                    label = _legend_label_halfword(
                        int(path["halfwords"][0]),
                        float(path["last_corr"]),
                        float(path["score"]),
                    )
                else:
                    representative = 0
                    for idx in range(2):
                        if idx in path["known_specs"]:
                            value = int(
                                path["known_specs"][idx][0]
                            )
                        else:
                            value = int(path["halfwords"][idx])
                        representative |= value << (16 * idx)
                    label = _legend_label_word(
                        representative,
                        float(path["last_corr"]),
                        float(path["score"]),
                    )
                labels.append(
                    _legend_label_with_rank(label, display_ranks[display_index])
                )

                if _is_true_path(path):
                    correct_index = display_index

            _plot_curves(
                engine,
                hypotheses,
                labels,
                correct_index=correct_index,
                x_offset=x_offset,
                title=f"{target_name} / {part}",
                plot_config=plot_config,
                attack=plot_attack,
                stage=stage_name,
                filename=f"halfword_{half_index}.png",
            )

    word_arrays: List[np.ndarray] = []
    candidates: List[HalfwordCandidate] = []

    for path in paths:
        word = np.zeros(rows, dtype=np.uint32)
        representative = 0
        representative_halves: List[int] = []

        for half_index in range(2):
            if half_index in path["known_specs"]:
                half_array = path["known_specs"][half_index]
                word |= (
                    half_array.astype(np.uint32)
                    << np.uint32(16 * half_index)
                )
                rep = int(half_array[0]) & 0xFFFF
            else:
                rep = int(path["halfwords"][half_index]) & 0xFFFF
                word |= np.uint32(rep << (16 * half_index))

            representative |= rep << (16 * half_index)
            representative_halves.append(rep)

        word_arrays.append(word)
        candidates.append(HalfwordCandidate(
            word=representative & MASK32,
            score=float(path["score"]),
            halfwords_lsb=representative_halves,
            corr_history=[float(v) for v in path["corr_history"]],
            poi_history=[int(v) for v in path["poi_history"]],
            last_corr=float(path.get("last_corr", 0.0)),
            last_poi=int(path.get("last_poi", -1)),
        ))

    return word_arrays[0], candidates


def _assemble_compression_seed(param_set: str, word_results: Mapping[int, CompressionWordResult]) -> bytes:
    last = _compression_last_secret_word(param_set)
    output = bytearray()
    w5 = int(word_results[5].word_array[0])
    output.extend(bytes([(w5 >> 8) & 0xFF, w5 & 0xFF]))
    for t in range(6, last):
        output.extend(word_to_be_bytes(int(word_results[t].word_array[0])))
    wlast = int(word_results[last].word_array[0])
    output.extend(bytes([(wlast >> 24) & 0xFF, (wlast >> 16) & 0xFF]))
    expected = PARAM_N[normalize_param_set(param_set)]
    if len(output) != expected:
        raise AssertionError(f"assembled compression seed length={len(output)}, expected={expected}")
    return bytes(output)



def run_compression_cpa(
    cap: CaptureData,
    *,
    target: str,
    pk_seed: bytes,
    validation_seed: Optional[bytes] = None,
    trace_slices: Optional[Mapping[int, Optional[slice]]] = None,
    engine_options: Optional[Mapping[str, Any]] = None,
    beam_width: int = 4,
    candidate_batch: int = 256,
    plot_top: int = 4,
    verbose: bool = True,
    plot_config: Optional[PlotConfig] = None,
) -> CompressionResult:
    """Run compression CPA using carry-aware 16-bit (two-byte) stages."""
    p = cap.param_set
    target_name = str(target).lower()
    if target_name not in ("t1", "e_update", "a_update"):
        raise ValueError("compression target must be 't1', 'e_update', or 'a_update'")

    pk = bytes(pk_seed)
    if len(pk) != PARAM_N[p]:
        raise ValueError(f"PK.seed for {p} must contain {PARAM_N[p]} bytes")
    seed = None if validation_seed is None else bytes(validation_seed)
    if seed is not None and len(seed) != PARAM_N[p]:
        raise ValueError(f"validation seed for {p} must contain {PARAM_N[p]} bytes")

    public_words = build_finalize_words(cap.addrs, param_set=p, sk_seed=None)
    true_words = (
        None
        if seed is None
        else build_finalize_words(cap.addrs, param_set=p, sk_seed=seed)
    )
    rows = cap.addrs.shape[0]
    state = _broadcast_state(seeded_state_from_pk_seed(p, pk), rows)

    for t in range(5):
        state, _, _, _ = _round_arrays(
            state,
            public_words[:, t],
            int(SHA256_K[t]),
        )

    last = _compression_last_secret_word(p)
    slices = dict(trace_slices or {})
    engine_options = dict(engine_options or {})
    results: Dict[int, CompressionWordResult] = {}

    if verbose:
        print(f"\n================ COMPRESSION {p} / {target_name} / HW ================")
    for t in range(5, last + 1):
        traces, offset = _stage_traces(cap, slices.get(t))
        base_t1 = (
            state["h"]
            + Sigma1_32(state["e"])
            + Ch(state["e"], state["f"], state["g"])
            + np.uint32(SHA256_K[t])
        ).astype(np.uint32)
        T2 = (
            Sigma0_32(state["a"])
            + Maj(state["a"], state["b"], state["c"])
        ).astype(np.uint32)

        if target_name == "t1":
            add_base = base_t1
        elif target_name == "e_update":
            add_base = (state["d"] + base_t1).astype(np.uint32)
        else:
            add_base = (base_t1 + T2).astype(np.uint32)

        known = _known_word_halfwords_for_compression(
            t,
            last,
            public_words,
        )
        true_array = (
            None if true_words is None else true_words[:, t].astype(np.uint32)
        )
        _recover_word_fn = (
            recover_add_word_with_known_parts_bytewise
            if int(COMPRESSION_PART_BITS) == 8
            else recover_add_word_with_known_halfwords
        )
        recovered_array, candidates = _recover_word_fn(
            traces,
            add_base,
            known_halfwords=known,
            true_word_array=true_array,
            engine_options=engine_options,
            target_name=f"F32_{p}_{target_name}_W{t}",
            beam_width=beam_width,
            candidate_batch=candidate_batch,
            plot_top=plot_top,
            x_offset=offset,
            verbose=verbose,
            plot_config=plot_config,
            plot_attack=f"compression_{target_name}",
            plot_stage=f"W{t}",
        )
        results[t] = CompressionWordResult(
            round_index=t,
            word_array=recovered_array,
            representative_word=int(recovered_array[0]),
            candidates=candidates,
        )
        state, _, _, _ = _round_arrays(
            state,
            recovered_array,
            int(SHA256_K[t]),
        )
        if verbose:
            print(f"W{t} recovered(first)=0x{int(recovered_array[0]):08X}")

    recovered_seed = _assemble_compression_seed(p, results)
    if verbose:
        print("COMPRESSION recovered SK.seed:", recovered_seed.hex())
        if seed is not None:
            print("COMPRESSION validation SK.seed:", seed.hex())
            print("MATCH" if recovered_seed == seed else "MISMATCH")
    return CompressionResult(p, target_name, recovered_seed, seed, results)

def self_test_layouts() -> bool:
    rng = np.random.default_rng(20260807)
    addrs = rng.integers(0, 256, size=(8, 32), dtype=np.uint8)
    expected = {
        128: {"last": 9, "length": 0x330},
        192: {"last": 11, "length": 0x370},
        256: {"last": 13, "length": 0x3B0},
    }
    for p in PARAM_N:
        seed = get_default_sk_seed(p)
        words = build_finalize_words(addrs, param_set=p, sk_seed=seed)
        bits = security_bits(p)
        last = expected[bits]["last"]
        assert int(words[0, 15]) == expected[bits]["length"]
        assert np.all((words[:, last] & np.uint32(0xFFFF)) == np.uint32(0x8000))
        # Independently check the first row's secret-bearing word layout.
        # Do not use hard-coded words derived from private research test vectors.
        padded_block = bytes(addrs[0, :22]) + seed + b"\x80"
        padded_block = padded_block.ljust(64, b"\x00")
        for word_index in range(5, last + 1):
            start = 4 * word_index
            expected_word = int.from_bytes(padded_block[start:start + 4], "big")
            assert int(words[0, word_index]) == expected_word
        schedule = expand_words(words, last_word=23)
        for t in range(16, 24):
            check = (
                schedule[:, t - 16]
                + sigma0_32(schedule[:, t - 15])
                + schedule[:, t - 7]
                + sigma1_32(schedule[:, t - 2])
            ).astype(np.uint32)
            assert np.array_equal(schedule[:, t], check)
    return True


# ===========================================================================
# Attack-only ePrint runner (128/192/256, f/s)
# ===========================================================================
#
# Every stage below is evaluated twice:
#     (1) byte-wise CPA : four 8-bit guesses with carry propagation
#     (2) two-byte CPA  : low16 -> high16 guesses with carry propagation
# Both results are scored against ground truth; the selected primary method
# only decides which value is reported as the recovered one.
#
# 128:
#   W9 via W18 (2^16 exhaustive) AND W9 via W16 (byte-wise / two-byte CPA);
#   both are measured, w9_128_primary_source picks the primary
#   -> W5 via W20 -> W6/W7/W8 via W21/W22/W23.
#
# 192:  no prior-secret assumption
#   W9  via W16   (base is entirely public)
#   W10 via W17
#   W11 via W18   (public low16 = 0x8000 supplied as a known byte/half pair)
#   -> W5 via W20 -> W6/W7/W8 via W21/W22/W23.
#
# 256 legacy note removed: full-recovery runner below uses joint W5/W13
#   W9 via W16 -> W10 via W17 -> W11 via W18 -> W12 via W19
#   -> W5 via W20 -> W6/W7/W8 via W21/W22/W23.
# ===========================================================================


@dataclass
class DownstreamSigmaCandidate:
    word: int
    low16: int
    high16: int
    stage1_score: float
    stage1_poi: int
    stage2_score: float
    stage2_poi: int
    window_start: int
    window_stop: int


def _normalize_local_window(
    n_points: int,
    window: Optional[Union[slice, Tuple[int, int], Tuple[int, int, int]]],
) -> slice:
    if window is None:
        return slice(0, int(n_points), 1)
    if isinstance(window, tuple):
        window = slice(*window)
    start, stop, step = window.indices(int(n_points))
    if step != 1:
        raise ValueError("window step must be 1")
    if stop <= start:
        raise ValueError(f"empty window: {window}")
    return slice(start, stop, 1)


def _poi_window(
    poi: int,
    n_points: int,
    radius: Union[int, Tuple[int, int]],
) -> Tuple[int, int]:
    if isinstance(radius, tuple):
        if len(radius) != 2:
            raise ValueError("poi_radius tuple must be (left, right)")
        left, right = int(radius[0]), int(radius[1])
    else:
        left = right = int(radius)
    if left < 0 or right < 0:
        raise ValueError("poi_radius must be non-negative")
    start = max(0, int(poi) - left)
    stop = min(int(n_points), int(poi) + right + 1)
    if stop <= start:
        raise ValueError("empty adaptive POI window")
    return start, stop


def _sigma1_low16_partial_value(value_low16: np.ndarray) -> np.ndarray:
    """Exact sigma1(x)[22:28] using only x[15:0].

    For SHA-256 sigma1(x)=ROTR17(x)^ROTR19(x)^SHR10(x), output bits
    22..28 depend only on input bits 7..15. This permits exact low16
    candidate ranking before the upper 16 bits are known.
    """
    x = np.asarray(value_low16, dtype=np.uint32) & np.uint32(0xFFFF)
    return (
        (sigma1_32(x) >> np.uint32(22))
        & np.uint32(0x7F)
    ).astype(np.uint8)


def _downstream_sigma_low16_hypothesis(
    add_base: np.ndarray,
    guesses: np.ndarray,
) -> np.ndarray:
    out_low16 = (
        (np.asarray(add_base, dtype=np.uint32)[:, None] & np.uint32(0xFFFF))
        + np.asarray(guesses, dtype=np.uint32)[None, :]
    ) & np.uint32(0xFFFF)
    partial = _sigma1_low16_partial_value(out_low16)
    return HW8[partial].astype(np.float32)


def _downstream_sigma_full_hypothesis(
    add_base: np.ndarray,
    words: np.ndarray,
) -> np.ndarray:
    expanded = (
        np.asarray(add_base, dtype=np.uint32)[:, None]
        + np.asarray(words, dtype=np.uint32)[None, :]
    ).astype(np.uint32)
    return hw_u32(sigma1_32(expanded)).astype(np.float32)


def recover_known_low16_via_next_sigma1(
    traces: np.ndarray,
    add_base: np.ndarray,
    *,
    known_low16: int,
    engine_options: Optional[Mapping[str, Any]] = None,
    candidate_batch: int = 256,
    top_final: int = 16,
    x_offset: int = 0,
    verbose: bool = True,
    plot_top: int = 4,
    plot_config: Optional[PlotConfig] = None,
    plot_stage: str = "downstream_sigma1",
    true_word: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Recover a word with known low16 using HW(sigma1(A + candidate))."""
    T = np.asarray(traces, dtype=np.float32)
    A = np.asarray(add_base, dtype=np.uint32).reshape(-1)
    if T.ndim != 2 or A.size != T.shape[0]:
        raise ValueError("trace/add_base shape mismatch")

    engine = CorrelationEngine(T, **dict(engine_options or {}))
    batch = max(1, int(candidate_batch))
    scores = np.zeros(65536, dtype=np.float32)
    pois = np.zeros(65536, dtype=np.int32)
    low = int(known_low16) & 0xFFFF

    if verbose:
        print(f"[{plot_stage}] exhaustive upper16 via HW(sigma1(expanded word))")

    for start in range(0, 65536, batch):
        stop = min(start + batch, 65536)
        high = np.arange(start, stop, dtype=np.uint32)
        words = ((high << np.uint32(16)) | np.uint32(low)).astype(np.uint32)
        H = _downstream_sigma_full_hypothesis(A, words)
        batch_scores, batch_pois = engine.score(H)
        scores[start:stop] = batch_scores
        pois[start:stop] = batch_pois
        if verbose and start % (batch * 64) == 0:
            print(f"  processed upper16 {start:5d}/65536")

    order = np.argsort(scores)[::-1]
    best_high = int(order[0])
    best_word = ((best_high << 16) | low) & MASK32

    if verbose:
        print(
            f"  best=0x{best_word:08X} corr={float(scores[best_high]):.6f} "
        )

    if plot_top > 0:
        displayed = [int(v) for v in order[:max(1, int(plot_top))]]
        true_high = None
        if true_word is not None and (int(true_word) & 0xFFFF) == low:
            true_high = (int(true_word) >> 16) & 0xFFFF
            if true_high not in displayed:
                displayed.append(true_high)

        hypotheses: List[np.ndarray] = []
        labels: List[str] = []
        correct_index: Optional[int] = None
        for index, high in enumerate(displayed):
            word = ((high << 16) | low) & MASK32
            H = _downstream_sigma_full_hypothesis(
                A, np.asarray([word], dtype=np.uint32)
            )[:, 0]
            hypotheses.append(H)
            rank_pos = int(np.where(order == high)[0][0]) + 1
            labels.append(
                _legend_label_with_rank(
                    _legend_label_word(word, float(scores[high])),
                    rank_pos if index >= max(1, int(plot_top)) else None,
                )
            )
            if true_high is not None and high == true_high:
                correct_index = index

        _plot_curves(
            engine,
            hypotheses,
            labels,
            correct_index=correct_index,
            x_offset=x_offset,
            plot_config=plot_config,
            attack="message_schedule",
            stage=plot_stage,
            filename="exhaustive_upper16.png",
        )

    keep = min(max(1, int(top_final)), 65536)
    return best_word, {
        "scores": scores,
        "pois": (pois + int(x_offset)).astype(np.int32),
        "order": order,
        "top_words": [
            (((int(high) << 16) | low) & MASK32)
            for high in order[:keep]
        ],
        "target": "sigma1",
    }


def recover_full_word_via_next_sigma1_adaptive_poi(
    traces: np.ndarray,
    add_base: np.ndarray,
    *,
    engine_options: Optional[Mapping[str, Any]] = None,
    low16_window: Optional[Union[slice, Tuple[int, int], Tuple[int, int, int]]] = None,
    poi_radius: Union[int, Tuple[int, int]] = 30,
    top_low16: int = 10,
    top_final: int = 16,
    candidate_batch: int = 256,
    x_offset: int = 0,
    verbose: bool = True,
    plot_top: int = 4,
    plot_config: Optional[PlotConfig] = None,
    plot_stage: str = "downstream_sigma1",
    true_word: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Recover a full 32-bit additive word through the next sigma1 intermediate.

    Stage 1 ranks low16 candidates with the exact partial intermediate
    HW(sigma1(A+candidate)[22:28]). Stage 2 combines retained low16 values
    with all high16 candidates and ranks complete words solely by
    HW(sigma1(A+candidate)) correlation in the parent-specific POI window.
    """
    import matplotlib.pyplot as plt

    T_full = np.asarray(traces, dtype=np.float32)
    A = np.asarray(add_base, dtype=np.uint32).reshape(-1)
    if T_full.ndim != 2 or A.size != T_full.shape[0]:
        raise ValueError("trace/add_base shape mismatch")

    opts = dict(engine_options or {})
    batch = max(1, int(candidate_batch))
    top_low = max(1, int(top_low16))
    top_out = max(1, int(top_final))

    # Stage 1: exact low16-dependent subset of sigma1.
    sl = _normalize_local_window(T_full.shape[1], low16_window)
    T1 = T_full[:, sl]
    stage1_offset = int(sl.start)
    engine1 = CorrelationEngine(T1, **opts)
    low_scores = np.zeros(65536, dtype=np.float32)
    low_pois = np.zeros(65536, dtype=np.int32)

    if verbose:
        print(f"[{plot_stage}/Stage1] low16 via HW(sigma1(expanded)[22:28])")

    for start in range(0, 65536, batch):
        stop = min(start + batch, 65536)
        guesses = np.arange(start, stop, dtype=np.uint32)
        H = _downstream_sigma_low16_hypothesis(A, guesses)
        scores, pois = engine1.score(H)
        low_scores[start:stop] = scores
        low_pois[start:stop] = pois + stage1_offset
        if verbose and start % (batch * 64) == 0:
            print(f"  processed low16 {start:5d}/65536")

    low_order = np.argsort(low_scores)[::-1]
    retained = low_order[:top_low].astype(np.uint32)

    if plot_top > 0:
        displayed_low = [int(v) for v in retained[:max(1, int(plot_top))]]
        true_low = None if true_word is None else (int(true_word) & 0xFFFF)
        if true_low is not None and true_low not in displayed_low:
            displayed_low.append(true_low)
        hypotheses: List[np.ndarray] = []
        labels: List[str] = []
        correct_index: Optional[int] = None
        for index, low in enumerate(displayed_low):
            H = _downstream_sigma_low16_hypothesis(
                A, np.asarray([low], dtype=np.uint32)
            )[:, 0]
            hypotheses.append(H)
            rank_pos = int(np.where(low_order == low)[0][0]) + 1
            labels.append(
                _legend_label_with_rank(
                    _legend_label_halfword(low, float(low_scores[low])),
                    rank_pos if index >= max(1, int(plot_top)) else None,
                )
            )
            if true_low is not None and low == true_low:
                correct_index = index
        _plot_curves(
            engine1,
            hypotheses,
            labels,
            correct_index=correct_index,
            x_offset=x_offset + int(sl.start),
            plot_config=plot_config,
            attack="message_schedule",
            stage=f"{plot_stage}_stage1_low16",
            filename="top_candidates.png",
        )

    # Stage 2: complete word, final score comes only from full sigma1 CPA.
    stage2_scores = np.zeros((retained.size, 65536), dtype=np.float32)
    stage2_pois = np.zeros((retained.size, 65536), dtype=np.int32)
    stage2_windows = np.zeros((retained.size, 2), dtype=np.int32)
    engine_cache: Dict[Tuple[int, int], CorrelationEngine] = {}

    if verbose:
        print(f"[{plot_stage}/Stage2] full32 via HW(sigma1(expanded))")

    for parent, low_u32 in enumerate(retained):
        low = int(low_u32)
        start, stop = _poi_window(int(low_pois[low]), T_full.shape[1], poi_radius)
        stage2_windows[parent] = (start, stop)
        key = (start, stop)
        if key not in engine_cache:
            engine_cache[key] = CorrelationEngine(T_full[:, start:stop], **opts)
        engine2 = engine_cache[key]

        for high_start in range(0, 65536, batch):
            high_stop = min(high_start + batch, 65536)
            high = np.arange(high_start, high_stop, dtype=np.uint32)
            words = (
                np.uint32(low)
                | (high << np.uint32(16))
            ).astype(np.uint32)
            H = _downstream_sigma_full_hypothesis(A, words)
            scores, pois = engine2.score(H)
            stage2_scores[parent, high_start:high_stop] = scores
            stage2_pois[parent, high_start:high_stop] = pois + start

    flat = stage2_scores.reshape(-1)
    order = np.argsort(flat)[::-1]
    keep = min(top_out, order.size)
    candidates: List[DownstreamSigmaCandidate] = []
    for flat_index in order[:keep]:
        flat_i = int(flat_index)
        parent = flat_i // 65536
        high = flat_i % 65536
        low = int(retained[parent])
        word = ((high << 16) | low) & MASK32
        start, stop = map(int, stage2_windows[parent])
        poi = int(stage2_pois[parent, high])
        candidates.append(DownstreamSigmaCandidate(
            word=word,
            low16=low,
            high16=high,
            stage1_score=float(low_scores[low]),
            stage1_poi=x_offset + int(low_pois[low]),
            stage2_score=float(stage2_scores[parent, high]),
            stage2_poi=x_offset + poi,
            window_start=x_offset + start,
            window_stop=x_offset + stop,
        ))

    if not candidates:
        raise RuntimeError("downstream sigma1 search produced no candidates")

    if verbose:
        best = candidates[0]
        print(
            f"  best=0x{best.word:08X} corr={best.stage2_score:.6f} "
        )

    # Candidate-specific Stage-2 curves; preserve the latest paper plot style.
    if plot_top > 0:
        specs: List[Tuple[int, int, int, float, int, int, int]] = []
        for flat_i in order[:max(1, int(plot_top))]:
            flat_i = int(flat_i)
            parent = flat_i // 65536
            high = flat_i % 65536
            low = int(retained[parent])
            start, stop = map(int, stage2_windows[parent])
            specs.append((
                low, high, parent,
                float(stage2_scores[parent, high]),
                int(stage2_pois[parent, high]),
                start, stop,
            ))

        if true_word is not None:
            true_low = int(true_word) & 0xFFFF
            true_high = (int(true_word) >> 16) & 0xFFFF
            parent_matches = np.flatnonzero(retained == np.uint32(true_low))
            if parent_matches.size:
                parent = int(parent_matches[0])
                start, stop = map(int, stage2_windows[parent])
                if not any(low == true_low and high == true_high for low, high, *_ in specs):
                    specs.append((
                        true_low, true_high, parent,
                        float(stage2_scores[parent, true_high]),
                        int(stage2_pois[parent, true_high]),
                        start, stop,
                    ))

        fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
        red_index = None
        if true_word is not None:
            for idx, (low, high, *_rest) in enumerate(specs):
                if (((high << 16) | low) & MASK32) == (int(true_word) & MASK32):
                    red_index = idx
                    break
        blue_index = next((i for i in range(len(specs)) if i != red_index), None)
        text_colors: List[str] = []
        top_count = max(1, int(plot_top))

        for index, (low, high, parent, score, poi, start, stop) in enumerate(specs):
            word = ((high << 16) | low) & MASK32
            engine2 = engine_cache[(start, stop)]
            H = _downstream_sigma_full_hypothesis(
                A, np.asarray([word], dtype=np.uint32)
            )[:, 0]
            curve = engine2.curve(H)
            if index == red_index:
                color, width, alpha, z = "red", 0.5, 0.5, 4
            elif index == blue_index:
                color, width, alpha, z = "blue", 0.5, 0.5, 3
            else:
                color, width, alpha, z = "gray", 0.5, 0.5, 1
            rank_pos = int(np.where(order == (parent * 65536 + high))[0][0]) + 1
            ax.plot(
                np.arange(start, stop) + int(x_offset),
                curve,
                color=color,
                linewidth=width,
                alpha=alpha,
                zorder=z,
                label=_legend_label_with_rank(
                    _legend_label_word(word, score),
                    rank_pos if index >= top_count else None,
                ),
            )
            text_colors.append(color if index in (red_index, blue_index) else "0.35")

        ax.set_xlabel(PLOT_XLABEL, fontsize=PLOT_FONT_LABEL)
        ax.set_ylabel(PLOT_YLABEL, fontsize=PLOT_FONT_LABEL)
        ax.tick_params(axis="both", labelsize=PLOT_FONT_TICK)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        _apply_rho_tick_format(ax)
        _apply_paper_legend(
            fig, ax, text_colors,
            dpi=(300 if plot_config is None else int(plot_config.dpi)),
        )
        _finish_figure(
            fig,
            config=plot_config,
            save_path=_plot_path(
                plot_config,
                attack="message_schedule",
                stage=f"{plot_stage}_stage2_full32",
                filename="top_candidates.png",
            ),
        )

    return int(candidates[0].word), {
        "candidates": candidates,
        "low16_scores": low_scores,
        "low16_pois": (low_pois + int(x_offset)).astype(np.int32),
        "low16_order": low_order,
        "retained_low16": retained,
        "stage2_scores": stage2_scores,
        "stage2_pois": (stage2_pois + int(x_offset)).astype(np.int32),
        "stage2_windows": (stage2_windows + np.int32(x_offset)).astype(np.int32),
        "target": "sigma1",
    }



# ---------------------------------------------------------------------------
# 256-bit full-recovery W5/W13 joint W20 stage
# ---------------------------------------------------------------------------
def _w5_class15_to_secret16(class15: Union[int, np.ndarray]):
    """Map a 15-bit W5 equivalence class to the representative with bit2=0."""
    c = np.asarray(class15, dtype=np.uint32) & np.uint32(0x7FFF)
    return ((c & np.uint32(0x3)) | ((c >> np.uint32(2)) << np.uint32(3))).astype(np.uint32)


def _w5_secret16_to_class15(secret16: int) -> int:
    s5 = int(secret16) & 0xFFFF
    return int((s5 & 0x3) | ((s5 >> 1) & 0x7FFC)) & 0x7FFF


def _w5_class_pair(class15: int) -> Tuple[int, int]:
    base = int(_w5_class15_to_secret16(np.uint32(class15))) & 0xFFFF
    return base, base | 0x0004


def _true_secret16_w13(ctx: ExpansionContext) -> Optional[int]:
    if ctx.true_words is None:
        return None
    return (int(ctx.true_words[0, 13]) >> 16) & 0xFFFF


def _record_joint_w5_w13_comparison(
    ctx: ExpansionContext,
    plot_config: Optional[PlotConfig],
    *,
    recovered_w5: int,
    recovered_w13: int,
    true_w5: Optional[int],
    true_w13: Optional[int],
    stage1_true_class_rank: Optional[int],
    stage1_true_class_retained: Optional[bool],
    final_true_pair_rank: Optional[int],
) -> Dict[str, Any]:
    pair_match = None
    if true_w5 is not None and true_w13 is not None:
        pair_match = bool(
            (int(recovered_w5) & 0xFFFF) == (int(true_w5) & 0xFFFF)
            and (int(recovered_w13) & 0xFFFF) == (int(true_w13) & 0xFFFF)
        )
    payload: Dict[str, Any] = {
        "target_word": "W5/W13",
        "intermediate_word": "W20",
        "bytewise_recovered": None,
        "two_byte_recovered": None,
        "methods_agree": None,
        "selected_method": "joint_w20_two_stage",
        "selected_recovered": (
            f"W5=0x{int(recovered_w5) & 0xFFFF:04X},"
            f"W13=0x{int(recovered_w13) & 0xFFFF:04X}"
        ),
        "true_word": (
            None if true_w5 is None or true_w13 is None else
            f"W5=0x{int(true_w5) & 0xFFFF:04X},W13=0x{int(true_w13) & 0xFFFF:04X}"
        ),
        "bytewise_match": None,
        "two_byte_match": None,
        "target_index": [5, 13],
        "intermediate_index": 20,
        "stage": "W5_W13_via_W20",
        "stage_key": "W5/W13@W20",
        "stage_domain": "joint_w5_w13",
        "role": "primary",
        "value_bits": 32,
        "selected_match": pair_match,
        "parameter_set": ctx.param_set,
        "stage_chaining": ctx.chaining,
        "stage1_true_class_rank": stage1_true_class_rank,
        "stage1_true_class_retained": stage1_true_class_retained,
        "final_true_pair_rank": final_true_pair_rank,
    }
    ctx.comparisons.append(payload)

    if plot_config is not None and plot_config.save_plots and plot_config.root is not None:
        directory = plot_config.root / "message_schedule" / "W5_W13_via_W20_comparison"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "comparison.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        payload["comparison_path"] = str(path)
    return payload


def _recover_w5_w13_joint_via_w20(
    ctx: ExpansionContext,
    *,
    trace_slice: Optional[slice],
    engine_options: Mapping[str, Any],
    candidate_batch: int,
    stage1_top_classes: int,
    final_top: int,
    plot_top: int,
    verbose: bool,
    plot_config: Optional[PlotConfig],
) -> Tuple[int, int]:
    """Recover S5=W5[15:0] and S13=W13[31:16] jointly through W20.

    This is the full-recovery 256-bit special stage.  It deliberately does NOT
    use the global 8-bit split/CUMULATIVE_HW switches:
      Stage 1: 2^15 W5 equivalence classes, HW16(W20[15:0]); retain top K.
      Stage 2: K x 2 (W5 bit2) x 2^16 S13, HW32(W20); select top pair.

    Validation truth is used only to score/rank the known correct class/pair and
    never to construct an attack hypothesis or downstream state.
    """
    if security_bits(ctx.param_set) != 256:
        raise ValueError("joint W5/W13 W20 recovery is only valid for 256f/256s")

    traces, offset = _stage_traces(ctx.cap, trace_slice)
    engine = CorrelationEngine(traces, **dict(engine_options))
    batch = max(1, int(candidate_batch))
    keep_classes = min(32768, max(1, int(stage1_top_classes)))
    keep_final = max(1, int(final_top))

    public_hi = (ctx.public_words[:, 5] & np.uint32(0xFFFF0000)).astype(np.uint32)
    # W18 must be the value derived from the actually recovered W9..W12 chain.
    base20 = (_word(ctx, 4) + sigma1_32(_word(ctx, 18))).astype(np.uint32)

    true_s5 = _true_secret16_w5(ctx)
    true_s13 = _true_secret16_w13(ctx)
    true_class = None if true_s5 is None else _w5_secret16_to_class15(true_s5)

    # Stage 1: W13 low16 is public 0x8000; S13 high16 cannot affect W20 low16.
    stage1_scores = np.zeros(32768, dtype=np.float32)
    stage1_pois = np.zeros(32768, dtype=np.int32)
    if verbose:
        print("[W5/W13@W20 Stage1] 2^15 W5 classes via HW16(W20[15:0])")
    for start in range(0, 32768, batch):
        stop = min(start + batch, 32768)
        classes = np.arange(start, stop, dtype=np.uint32)
        s5_base = _w5_class15_to_secret16(classes)
        w5 = public_hi[:, None] | s5_base[None, :]
        w20_low = (
            base20[:, None] + sigma0_32(w5) + np.uint32(0x8000)
        ) & np.uint32(0xFFFF)
        block_scores, block_pois = engine.score(hw_u16(w20_low))
        stage1_scores[start:stop] = block_scores
        stage1_pois[start:stop] = block_pois + int(offset)

    stage1_order = np.argsort(stage1_scores)[::-1]
    retained = stage1_order[:keep_classes].astype(np.uint32)
    true_class_rank = None
    true_class_retained = None
    if true_class is not None:
        true_class_rank = int(np.where(stage1_order == int(true_class))[0][0]) + 1
        true_class_retained = bool(np.any(retained == np.uint32(true_class)))

    if plot_top > 0:
        shown = [int(v) for v in stage1_order[:max(1, int(plot_top))]]
        if true_class is not None and int(true_class) not in shown:
            shown.append(int(true_class))
        hypotheses: List[np.ndarray] = []
        labels: List[str] = []
        correct_index: Optional[int] = None
        for i, cls in enumerate(shown):
            s5_base = int(_w5_class15_to_secret16(np.uint32(cls))) & 0xFFFF
            w5 = public_hi | np.uint32(s5_base)
            low = (base20 + sigma0_32(w5) + np.uint32(0x8000)) & np.uint32(0xFFFF)
            hypotheses.append(hw_u16(low))
            s0, s1 = _w5_class_pair(cls)
            labels.append(f"S5={s0:04X}/{s1:04X} ρ={float(stage1_scores[cls]):.4f}")
            if true_class is not None and cls == int(true_class):
                correct_index = i
        _plot_curves(
            engine, hypotheses, labels,
            correct_index=correct_index,
            x_offset=offset,
            plot_config=plot_config,
            attack="message_schedule",
            stage="W5_W13_via_W20_stage1_15bit",
            filename="top_classes.png",
        )

    # Stage 2: resolve W5 bit2 and S13 jointly from full W20 HW.
    stage2_scores = np.zeros((retained.size, 2, 65536), dtype=np.float32)
    stage2_pois = np.zeros((retained.size, 2, 65536), dtype=np.int32)
    if verbose:
        total = int(retained.size) * 2 * 65536
        print(f"[W5/W13@W20 Stage2] {total:,} W5/W13 pairs via HW32(W20)")

    for parent, cls_u32 in enumerate(retained):
        cls = int(cls_u32)
        s5_zero = int(_w5_class15_to_secret16(np.uint32(cls))) & 0xFFFF
        for bit2 in (0, 1):
            s5 = s5_zero | (bit2 << 2)
            w5 = public_hi | np.uint32(s5)
            base_s5 = (base20 + sigma0_32(w5) + np.uint32(0x8000)).astype(np.uint32)
            for start in range(0, 65536, batch):
                stop = min(start + batch, 65536)
                s13 = np.arange(start, stop, dtype=np.uint32)
                w20 = (base_s5[:, None] + (s13[None, :] << np.uint32(16))).astype(np.uint32)
                block_scores, block_pois = engine.score(hw_u32(w20))
                stage2_scores[parent, bit2, start:stop] = block_scores
                stage2_pois[parent, bit2, start:stop] = block_pois + int(offset)

    flat = stage2_scores.reshape(-1)
    final_order = np.argsort(flat)[::-1]
    best_flat = int(final_order[0])
    best_s13 = best_flat % 65536
    tmp = best_flat // 65536
    best_bit2 = tmp % 2
    best_parent = tmp // 2
    best_class = int(retained[best_parent])
    best_s5_zero = int(_w5_class15_to_secret16(np.uint32(best_class))) & 0xFFFF
    best_s5 = best_s5_zero | (best_bit2 << 2)

    true_pair_rank = None
    if true_s5 is not None and true_s13 is not None and bool(true_class_retained):
        true_parent = int(np.where(retained == np.uint32(true_class))[0][0])
        true_bit2 = (int(true_s5) >> 2) & 1
        true_flat = ((true_parent * 2 + true_bit2) * 65536) + int(true_s13)
        true_pair_rank = int(np.where(final_order == true_flat)[0][0]) + 1

    # Strict end-to-end propagation: use the top-1 recovered pair only.
    ctx.work[5] = (public_hi | np.uint32(best_s5)).astype(np.uint32)
    recovered_w13 = ((int(best_s13) << 16) | 0x8000) & MASK32
    _set_constant_word(ctx, 13, recovered_w13)
    ctx.expanded[20] = (
        _word(ctx, 4) + sigma0_32(ctx.work[5]) + _word(ctx, 13) + sigma1_32(_word(ctx, 18))
    ).astype(np.uint32)
    ctx.recovered["w5_secret16"] = int(best_s5) & 0xFFFF
    ctx.recovered["w13"] = recovered_w13

    pair_match = None
    if true_s5 is not None and true_s13 is not None:
        pair_match = bool(
            (int(best_s5) & 0xFFFF) == (int(true_s5) & 0xFFFF)
            and (int(best_s13) & 0xFFFF) == (int(true_s13) & 0xFFFF)
        )

    top_candidates: List[Dict[str, Any]] = []
    for rank, flat_idx in enumerate(final_order[:min(keep_final, final_order.size)], 1):
        idx2 = int(flat_idx)
        s13 = idx2 % 65536
        temp = idx2 // 65536
        bit2 = temp % 2
        parent = temp // 2
        cls = int(retained[parent])
        s5_zero = int(_w5_class15_to_secret16(np.uint32(cls))) & 0xFFFF
        s5 = s5_zero | (bit2 << 2)
        top_candidates.append({
            "rank": int(rank),
            "w5_secret16": int(s5),
            "w13_secret16": int(s13),
            "stage1_class": int(cls),
            "stage1_rank": int(np.where(stage1_order == cls)[0][0]) + 1,
            "stage1_score": float(stage1_scores[cls]),
            "stage2_score": float(stage2_scores[parent, bit2, s13]),
            "stage2_poi": int(stage2_pois[parent, bit2, s13]),
        })

    ctx.stage_info["w5_w13_joint"] = {
        "intermediate": "W20",
        "method": "stage1_15bit_stage2_pair_hw32_w20",
        "selected_w5_secret16": int(best_s5),
        "selected_w13_secret16": int(best_s13),
        "selected_w13_word": int(recovered_w13),
        "stage1_top_classes": int(keep_classes),
        "stage1_true_class_rank": true_class_rank,
        "stage1_true_class_retained": true_class_retained,
        "final_true_pair_rank": true_pair_rank,
        "pair_match": pair_match,
        "top_candidates": top_candidates,
    }
    _record_joint_w5_w13_comparison(
        ctx, plot_config,
        recovered_w5=best_s5,
        recovered_w13=best_s13,
        true_w5=true_s5,
        true_w13=true_s13,
        stage1_true_class_rank=true_class_rank,
        stage1_true_class_retained=true_class_retained,
        final_true_pair_rank=true_pair_rank,
    )

    if plot_top > 0:
        specs = list(top_candidates[:max(1, int(plot_top))])
        if true_pair_rank is not None and true_s5 is not None and true_s13 is not None:
            if not any(
                int(row["w5_secret16"]) == int(true_s5)
                and int(row["w13_secret16"]) == int(true_s13)
                for row in specs
            ):
                true_parent = int(np.where(retained == np.uint32(true_class))[0][0])
                true_bit2 = (int(true_s5) >> 2) & 1
                specs.append({
                    "rank": int(true_pair_rank),
                    "w5_secret16": int(true_s5),
                    "w13_secret16": int(true_s13),
                    "stage2_score": float(stage2_scores[true_parent, true_bit2, int(true_s13)]),
                })
        hypotheses: List[np.ndarray] = []
        labels: List[str] = []
        correct_index: Optional[int] = None
        for i, row in enumerate(specs):
            s5 = int(row["w5_secret16"]) & 0xFFFF
            s13 = int(row["w13_secret16"]) & 0xFFFF
            w5 = public_hi | np.uint32(s5)
            w20 = (base20 + sigma0_32(w5) + np.uint32(0x8000) + np.uint32(s13 << 16)).astype(np.uint32)
            hypotheses.append(hw_u32(w20))
            labels.append(f"S5={s5:04X} S13={s13:04X} ρ={float(row['stage2_score']):.4f}")
            if true_s5 is not None and true_s13 is not None and s5 == int(true_s5) and s13 == int(true_s13):
                correct_index = i
        _plot_curves(
            engine, hypotheses, labels,
            correct_index=correct_index,
            x_offset=offset,
            plot_config=plot_config,
            attack="message_schedule",
            stage="W5_W13_via_W20_stage2_full32",
            filename="top_pairs.png",
        )

    if verbose:
        print(
            f"joint W5/W13: W5=0x{best_s5:04X}, W13=0x{best_s13:04X}, "
            f"stage1_true_rank={true_class_rank}, final_true_pair_rank={true_pair_rank}"
        )
    return int(best_s5) & 0xFFFF, int(best_s13) & 0xFFFF


def self_test_joint_w5_w13_split() -> bool:
    """Sanity-check the 15+1-bit W5 equivalence split used by the joint stage."""
    rng = np.random.default_rng(20260918)
    public_hi = rng.integers(0, 2**16, size=512, dtype=np.uint32) << np.uint32(16)
    secret = rng.integers(0, 2**16, size=512, dtype=np.uint32)
    a = sigma0_32(public_hi | secret) & np.uint32(0xFFFF)
    b = sigma0_32(public_hi | (secret ^ np.uint32(0x0004))) & np.uint32(0xFFFF)
    if not np.array_equal(a, b):
        raise AssertionError("W5 bit2 must be invisible in sigma0(W5)[15:0]")
    for s5 in (0, 4, 0x0854, 0xFFFF, 0x1234):
        cls = _w5_secret16_to_class15(s5)
        s0, s1 = _w5_class_pair(cls)
        if (s0 ^ s1) != 0x0004:
            raise AssertionError("W5 equivalence pair must differ only in bit2")
        if (s5 & ~0x0004) != (s0 & ~0x0004):
            raise AssertionError("W5 class mapping mismatch")
    return True

def _normalize_attack_assumptions(
    param_set: str | int,
    assumptions: Optional[Mapping[str, int]],
) -> Dict[str, int]:
    """Full-recovery mode: no secret word/halfword assumptions are accepted."""
    normalize_param_set(param_set)
    supplied = dict(assumptions or {})
    if supplied:
        raise ValueError(
            "This cumulative-HW full-recovery variant does not accept "
            "assumed_recovered secret values. Remove conditional assumptions."
        )
    return {}


def conditional_assumptions_from_seed(
    param_set: str | int,
    sk_seed: bytes,
) -> Dict[str, int]:
    """Compatibility shim: full-recovery mode never derives attack assumptions."""
    p = normalize_param_set(param_set)
    seed = bytes(sk_seed)
    if len(seed) != PARAM_N[p]:
        raise ValueError(f"SK.seed for {p} must contain {PARAM_N[p]} bytes")
    return {}


def default_expansion_assumptions(param_set: str | int) -> Dict[str, int]:
    normalize_param_set(param_set)
    return {}

def run_expansion_cpa(
    cap: CaptureData,
    *,
    assumed_recovered: Optional[Mapping[str, int]] = None,
    validation_seed: Optional[bytes] = None,
    trace_slices: Optional[Mapping[str, Optional[slice]]] = None,
    engine_options: Optional[Mapping[str, Any]] = None,
    candidate_batch: int = 256,
    beam_width: int = 4,
    plot_top: int = 4,
    top_16bit: int = 10,
    downstream_sigma_top_low16: int = 10,
    downstream_sigma_top_final: int = 16,
    downstream_sigma_poi_radius: Union[int, Tuple[int, int]] = 30,
    w11_w12_primary_method: str = "2byte",
    stage_primary_method: Optional[str] = None,
    stage_chaining: str = "assume_previous_success",
    w9_128_primary_source: str = "auto",
    joint_stage1_top_classes: int = 10,
    joint_final_top: int = 16,
    verbose: bool = True,
    plot_config: Optional[PlotConfig] = None,
) -> ExpansionResult:
    """Run message-schedule CPA and plot every CPA stage.

    Every stage is executed twice, byte-wise and two-byte, and the per-stage
    comparison is written to
    ``<plot root>/message_schedule/W<t>_via_W<i>_comparison/comparison.json``
    plus a single roll-up at
    ``<plot root>/message_schedule/stage_method_comparison.json``.
    """
    ctx = create_expansion_context(
        cap,
        validation_seed=validation_seed,
        chaining=stage_chaining,
    )
    slices = dict(trace_slices or {})
    engine_options = dict(engine_options or {})
    bits = security_bits(ctx.param_set)
    _normalize_attack_assumptions(ctx.param_set, assumed_recovered)
    raw_method = stage_primary_method if stage_primary_method is not None else w11_w12_primary_method
    primary_method = str(raw_method).strip().lower().replace("-", "")
    aliases = {"byte": "byte", "bytewise": "byte", "1byte": "byte", "2byte": "2byte", "twobyte": "2byte", "halfword": "2byte"}
    if primary_method not in aliases:
        raise ValueError("primary method must be 'byte' or '2byte'")
    primary_method = aliases[primary_method]
    ctx.primary_method = primary_method      # lets DUAL_METHOD_MODE="selected" skip the other method

    stage_kwargs = dict(
        engine_options=engine_options,
        beam_width=beam_width,
        candidate_batch=candidate_batch,
        plot_top=plot_top,
        primary_method=primary_method,
        verbose=verbose,
        plot_config=plot_config,
    )

    ctx.stage_info["stage_chaining"] = ctx.chaining
    ctx.stage_info["stage_primary_method"] = primary_method

    if verbose:
        print(f"\n================ MESSAGE SCHEDULE {ctx.param_set} ================")
        print("trace shape:", cap.traces.shape)
        print("primary method :", primary_method)
        print("stage chaining :", ctx.chaining)
        if ctx.chaining == "assume_previous_success" and ctx.true_words is None:
            print(
                "NOTE: no validation seed supplied, so stage chaining falls "
                "back to the recovered values."
            )

    if bits == 128:
        # W9 is attacked twice, independently:
        #   (a) the original route: 2^16 exhaustive search through W18
        #       (W18 = W2 + sigma0(W3) + W11 + sigma1(W16), W16 carries W9)
        #   (b) the direct route:   W16 = W0 + sigma0(W1) + W9 + sigma1(W14),
        #       where W9 is a plain additive constant whose public low16
        #       padding 0x8000 is supplied as known, so byte-wise and
        #       two-byte CPA both apply -- the same shape as 192/256.
        # Both results are stored; w9_128_primary_source picks the one that
        # feeds the recovered SK.seed and the downstream schedule words.
        source = normalize_w9_128_source(w9_128_primary_source)
        if source == "auto":
            group_name = str(ctx.cap.group).lower()
            source = "direct_w16" if "fors" in group_name else "exhaustive_w18"
        ctx.stage_info["w9_128_primary_source"] = source

        if verbose:
            print("CPA targets: W9 (via W18 exhaustive AND via W16 CPA)")
            print("             -> W5 -> W6 -> W7 -> W8")
            print("W9 primary source:", source)

        exhaustive_w9 = _search_128_w9(
            ctx,
            trace_slice=slices.get("w9_w18", slices.get("w9")),
            engine_options=engine_options,
            candidate_batch=candidate_batch,
            top_n=top_16bit,
            plot_top=plot_top,
            verbose=verbose,
            plot_config=plot_config,
            commit=False,
            role="primary" if source == "exhaustive_w18" else "alternative",
            stage_info_key="w9_via_w18",
        )

        direct_w9 = _recover_direct_word(
            ctx,
            target_word=9,
            output_t=16,
            trace_slice=slices.get("w9_w16", slices.get("w9")),
            commit=False,
            role="primary" if source == "direct_w16" else "alternative",
            stage_info_key="w9_via_w16",
            **stage_kwargs,
        )

        selected_w9 = (
            exhaustive_w9 if source == "exhaustive_w18" else direct_w9
        )
        true_w9 = _true_constant(ctx.true_words, 9)
        chain_w9 = _chain_value(ctx, selected_w9, true_w9)
        _set_constant_word(ctx, 9, chain_w9)
        ctx.recovered["w9"] = int(selected_w9) & MASK32
        ctx.stage_info["w9"] = {
            "primary_source": source,
            "exhaustive_w18_word": int(exhaustive_w9) & MASK32,
            "direct_w16_word": int(direct_w9) & MASK32,
            "sources_agree": bool(
                (int(exhaustive_w9) & MASK32) == (int(direct_w9) & MASK32)
            ),
            "selected_word": int(selected_w9) & MASK32,
            "chained_word": int(chain_w9) & MASK32,
            "via_w18": ctx.stage_info["w9_via_w18"],
            "via_w16": ctx.stage_info["w9_via_w16"],
        }

        if verbose:
            print(
                f"W9 source comparison: exhaustive_w18=0x{exhaustive_w9:08X}, "
                f"direct_w16=0x{direct_w9:08X}, "
                f"selected({source})=0x{selected_w9:08X}"
            )

        for t in (16, 17, 18, 19):
            _derive_schedule_word(ctx, t)

    elif bits == 192:
        # Fully unconditional 192-bit experiment.
        #   W16 = W0 + sigma0(W1) + W9  + sigma1(W14)   -> W9  is additive
        #   W17 = W1 + sigma0(W2) + W10 + sigma1(W15)   -> W10 is additive
        #   W18 = W2 + sigma0(W3) + W11 + sigma1(W16)   -> W11 is additive
        # W11[15:0] is the public 0x8000 padding, so only its high16 is searched.
        if verbose:
            print("no prior-secret assumption required for 192")
            print(
                "CPA targets: W9 via W16 -> W10 via W17 -> W11 via W18 "
                "-> W5 -> W6 -> W7 -> W8"
            )

        for target, t in ((9, 16), (10, 17), (11, 18)):
            _recover_direct_word(
                ctx,
                target_word=target,
                output_t=t,
                trace_slice=slices.get(f"w{target}"),
                **stage_kwargs,
            )

        # W12 is public zero padding for 192, so W19 is derivable from W17.
        _derive_schedule_word(ctx, 19)

    else:
        # Fully unconditional 256-bit experiment. W13 is NOT supplied.
        if verbose:
            print("no prior-secret assumption required for 256")
            print(
                "CPA targets: W9 via W16 -> W10 via W17 -> W11 via W18 "
                "-> W12 via W19 -> joint(W5,W13) via W20 -> W6 -> W7 -> W8"
            )

        for target, t in ((9, 16), (10, 17), (11, 18), (12, 19)):
            _recover_direct_word(
                ctx,
                target_word=target,
                output_t=t,
                trace_slice=slices.get(f"w{target}"),
                **stage_kwargs,
            )

    # W20 stage. 128/192 keep the original cumulative 8-bit W5 split.
    # 256 uses the dedicated two-stage W5/W13 joint recovery (not byte-split).
    if bits == 256:
        _recover_w5_w13_joint_via_w20(
            ctx,
            trace_slice=slices.get("w5_w13", slices.get("w5")),
            engine_options=engine_options,
            candidate_batch=candidate_batch,
            stage1_top_classes=joint_stage1_top_classes,
            final_top=joint_final_top,
            plot_top=plot_top,
            verbose=verbose,
            plot_config=plot_config,
        )
    else:
        _recover_w5_unified(
            ctx,
            trace_slice=slices.get("w5"),
            **stage_kwargs,
        )
    for target, t in ((6, 21), (7, 22), (8, 23)):
        _recover_sigma_word(
            ctx,
            target_initial_word=target,
            output_t=t,
            trace_slice=slices.get(f"w{target}"),
            **stage_kwargs,
        )

    recovered_seed = assemble_expansion_seed(ctx)
    comparisons = list(ctx.comparisons)
    ctx.stage_info["method_comparisons"] = comparisons
    summary = _summarize_stage_comparisons(comparisons)
    ctx.stage_info["method_summary"] = summary
    _save_stage_comparison_rollup(
        plot_config,
        param_set=ctx.param_set,
        comparisons=comparisons,
        summary=summary,
        stage_chaining=ctx.chaining,
        primary_method=primary_method,
        w9_128_primary_source=ctx.stage_info.get("w9_128_primary_source"),
    )

    if verbose:
        print("MESSAGE-SCHEDULE recovered SK.seed:", recovered_seed.hex())
        if validation_seed is not None:
            validation = bytes(validation_seed)
            print("validation SK.seed             :", validation.hex())
            print("MATCH" if recovered_seed == validation else "MISMATCH")
            print(
                "per-stage byte-wise  : "
                f"{summary['bytewise_success']}/{summary['split_scored_stages']}"
            )
            print(
                "per-stage two-byte   : "
                f"{summary['two_byte_success']}/{summary['split_scored_stages']}"
            )
            print(
                "per-stage selected   : "
                f"{summary['selected_success']}/{summary['scored_stages']}"
            )
            if summary.get("joint_scored"):
                print(
                    "joint W5/W13       : "
                    f"{summary['joint_success']}/{summary['joint_scored']}"
                )
            print("per-stage detail     :", summary["primary_detail"])
            if summary.get("alternative_stages"):
                print("alternative route    :", summary["alternative_detail"])

    return ExpansionResult(
        ctx.param_set,
        recovered_seed,
        None if validation_seed is None else bytes(validation_seed),
        dict(ctx.recovered),
        dict(ctx.stage_info),
        comparisons,
        summary,
    )


def _summarize_stage_comparisons(
    comparisons: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate split/exhaustive/joint stage outcomes for one capture."""

    def _mark(value: Optional[bool]) -> str:
        return "?" if value is None else ("+" if value else "-")

    def _label(row: Mapping[str, Any], disambiguate: bool) -> str:
        base = str(row.get("target_word", "?"))
        return str(row.get("stage_key", base)) if disambiguate else base

    def _detail(rows: Sequence[Mapping[str, Any]]) -> str:
        seen: Dict[str, int] = {}
        for row in rows:
            key = str(row.get("target_word", "?"))
            seen[key] = seen.get(key, 0) + 1
        pieces: List[str] = []
        for row in rows:
            domain = str(row.get("stage_domain", ""))
            dup = seen.get(str(row.get("target_word", "?")), 0) > 1
            label = _label(row, dup)
            if domain == "exhaustive16":
                pieces.append(f"{label}=X{_mark(row.get('selected_match'))}")
            elif domain == "joint_w5_w13":
                pieces.append(f"{label}=J{_mark(row.get('selected_match'))}")
            else:
                pieces.append(
                    f"{label}=B{_mark(row.get('bytewise_match'))}"
                    f"H{_mark(row.get('two_byte_match'))}"
                )
        return ";".join(pieces)

    primary = [row for row in comparisons if str(row.get("role", "primary")) != "alternative"]
    alternative = [row for row in comparisons if str(row.get("role", "primary")) == "alternative"]

    selected_scored = [row for row in primary if row.get("selected_match") is not None]
    split_scored = [row for row in primary if row.get("bytewise_match") is not None]
    dual = [row for row in split_scored if row.get("stage_domain") not in ("exhaustive16", "joint_w5_w13")]
    joint = [row for row in primary if row.get("stage_domain") == "joint_w5_w13" and row.get("selected_match") is not None]

    byte_ok = sum(1 for row in split_scored if row.get("bytewise_match") is True)
    two_ok = sum(1 for row in split_scored if row.get("two_byte_match") is True)
    sel_ok = sum(1 for row in selected_scored if row.get("selected_match") is True)
    agree = sum(1 for row in dual if row.get("methods_agree") is True)
    joint_ok = sum(1 for row in joint if row.get("selected_match") is True)

    alt_scored = [row for row in alternative if row.get("selected_match") is not None]
    return {
        "total_stages": len(primary),
        "scored_stages": len(selected_scored),
        "split_scored_stages": len(split_scored),
        "dual_method_stages": len(dual),
        "joint_scored": len(joint),
        "joint_success": int(joint_ok),
        "bytewise_success": int(byte_ok),
        "two_byte_success": int(two_ok),
        "selected_success": int(sel_ok),
        "methods_agree": int(agree),
        "failed_stages_selected": [
            str(row.get("stage_key", row.get("target_word")))
            for row in selected_scored if row.get("selected_match") is False
        ],
        "failed_stages_bytewise": [
            str(row.get("stage_key", row.get("target_word")))
            for row in split_scored if row.get("bytewise_match") is False
        ],
        "failed_stages_two_byte": [
            str(row.get("stage_key", row.get("target_word")))
            for row in split_scored if row.get("two_byte_match") is False
        ],
        "detail": _detail(comparisons),
        "primary_detail": _detail(primary),
        "alternative_stages": len(alternative),
        "alternative_scored": len(alt_scored),
        "alternative_bytewise_success": int(sum(1 for row in alternative if row.get("bytewise_match") is True)),
        "alternative_two_byte_success": int(sum(1 for row in alternative if row.get("two_byte_match") is True)),
        "alternative_selected_success": int(sum(1 for row in alt_scored if row.get("selected_match") is True)),
        "alternative_detail": _detail(alternative),
    }

def _save_stage_comparison_rollup(
    plot_config: Optional[PlotConfig],
    *,
    param_set: str,
    comparisons: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    stage_chaining: str,
    primary_method: str,
    w9_128_primary_source: Optional[str] = None,
) -> Optional[Path]:
    """Write one JSON per capture holding every stage's method comparison."""
    if plot_config is None or not plot_config.save_plots or plot_config.root is None:
        return None
    directory = plot_config.root / "message_schedule"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stage_method_comparison.json"
    payload = {
        "format": "slh-dsa-cpa-stage-method-comparison-v2-full-recovery",
        "attack_variant": CPA_HELPER_VARIANT,
        "full_recovery_no_secret_assumptions": True,
        "cumulative_hw": bool(CUMULATIVE_HW),
        "parameter_set": param_set,
        "compiler_opt": plot_config.compiler_opt,
        "address_kind": plot_config.address_kind,
        "seed_id": int(plot_config.seed_id),
        "stage_chaining": str(stage_chaining),
        "primary_method": str(primary_method),
        "w9_128_primary_source": (
            None if w9_128_primary_source is None else str(w9_128_primary_source)
        ),
        "summary": dict(summary),
        "stages": [dict(row) for row in comparisons],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def normalize_compiler_opt(value: str) -> str:
    text = str(value).strip().lower().replace("-", "")
    aliases = {
        "o0": "O0", "0": "O0",
        "o1": "O1", "1": "O1",
        "o2": "O2", "2": "O2",
        "o3": "O3", "3": "O3",
        "os": "Os", "s": "Os",
    }
    if text not in aliases:
        raise ValueError("compiler_opt must be O0/O1/O2/O3/Os")
    return aliases[text]


def normalize_address_kind(value: str) -> str:
    text = str(value).strip().lower()
    aliases = {
        "wots": "wots",
        "wots+": "wots",
        "fors": "fors",
    }
    if text not in aliases:
        raise ValueError("address_kind must be 'wots' or 'fors'")
    return aliases[text]


@dataclass
class DiscoveredCapture:
    prefix: str
    config_path: str
    parameter_set: str
    compiler_opt: str
    address_kind: str
    seed_id: int
    random_rows: int
    trace_points: int
    match_source: str


def _config_build(config: Mapping[str, Any]) -> Dict[str, Any]:
    build = config.get("build", {})
    return dict(build) if isinstance(build, Mapping) else {}


def _compiler_from_config_or_path(
    config: Mapping[str, Any],
    path: Path,
) -> Optional[str]:
    build = _config_build(config)
    for key in (
        "compiler_optimization",
        "compiler_optimization_detected",
        "make_opt",
    ):
        value = build.get(key)
        if value is None:
            value = config.get(key)
        if value is not None:
            try:
                return normalize_compiler_opt(str(value))
            except ValueError:
                pass

    for part in reversed(path.parts):
        try:
            return normalize_compiler_opt(part)
        except ValueError:
            continue
    return None


def _kind_from_config_or_path(
    config: Mapping[str, Any],
    path: Path,
) -> Optional[str]:
    build = _config_build(config)
    for key in ("multi_seed_kind", "kind", "address_kind"):
        value = build.get(key)
        if value is None:
            value = config.get(key)
        if value is not None:
            try:
                return normalize_address_kind(str(value))
            except ValueError:
                pass

    lowered = [part.lower() for part in path.parts]
    if any("multi_seed_wots_fixed_random" in part for part in lowered):
        return "wots"
    if any("multi_seed_fors_fixed_random" in part for part in lowered):
        return "fors"

    name = path.name.lower()
    if "_wots_" in name:
        return "wots"
    if "_fors_" in name:
        return "fors"
    return None


def _seed_id_from_config_or_path(
    config: Mapping[str, Any],
    path: Path,
) -> Optional[int]:
    build = _config_build(config)
    for key in ("seed_id",):
        value = build.get(key)
        if value is None:
            value = config.get(key)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass

    for part in reversed(path.parts):
        match = re.fullmatch(r"seed_(\d+)", part.lower())
        if match:
            return int(match.group(1))
    return None


def _parameter_from_config_or_path(
    config: Mapping[str, Any],
    path: Path,
) -> Optional[str]:
    value = config.get("parameter_set")
    if value is not None:
        try:
            return normalize_param_set(str(value))
        except ValueError:
            pass

    pattern = re.compile(r"slh_dsa_(128[fs]|192[fs]|256[fs])_sha2", re.I)
    for part in reversed(path.parts):
        match = pattern.search(part)
        if match:
            return normalize_param_set(match.group(1))
    return None


def _candidate_capture_prefixes_from_root(root: Path) -> List[str]:
    prefixes: List[str] = []
    seen = set()
    if not root.exists():
        return prefixes

    # capture_config.json is authoritative for the latest acquisition helper.
    configs = (
        [root]
        if root.is_file() and root.name.endswith("_capture_config.json")
        else list(root.rglob("*_capture_config.json"))
    )
    for cfg in configs:
        cfg_s = str(cfg)
        prefix = cfg_s[:-len("_capture_config.json")]
        if prefix not in seen:
            prefixes.append(prefix)
            seen.add(prefix)

    # Fallback for partially copied captures lacking the JSON file.
    for traces in root.rglob("*_traces_random.npy"):
        prefix = str(traces)[:-len("_traces_random.npy")]
        if prefix not in seen:
            prefixes.append(prefix)
            seen.add(prefix)
    return prefixes


def _preferred_search_roots(
    base_folder: str | os.PathLike,
    *,
    param_set: str,
    compiler_opt: str,
    address_kind: str,
    seed_id: int,
) -> List[Tuple[Path, str]]:
    base = Path(windows_to_wsl_path(base_folder)).expanduser()
    p = normalize_param_set(param_set)
    opt = normalize_compiler_opt(compiler_opt)
    kind = normalize_address_kind(address_kind)
    seed_name = f"seed_{int(seed_id):03d}"

    roots: List[Tuple[Path, str]] = []
    seen = set()

    def add(path: Path, source: str) -> None:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            roots.append((path, source))
            seen.add(key)

    parameter_dir_name = f"SLH_DSA_{p}_sha2_compiler_compare"

    # BASE is the common Result folder.
    add(
        base / parameter_dir_name / opt
        / f"multi_seed_{kind}_fixed_random" / seed_name,
        "expected-hierarchy",
    )

    # BASE already points at SLH_DSA_<param>_sha2_compiler_compare.
    add(
        base / opt / f"multi_seed_{kind}_fixed_random" / seed_name,
        "parameter-root",
    )

    # BASE already points at the compiler folder.
    add(
        base / f"multi_seed_{kind}_fixed_random" / seed_name,
        "compiler-root",
    )

    # BASE already points at multi_seed_<kind>_fixed_random.
    add(base / seed_name, "kind-root")

    # BASE may already point at seed_XXX or even below it.
    add(base, "base-recursive")
    return roots


def discover_capture_prefix(
    base_folder: str | os.PathLike,
    *,
    param_set: str | int,
    address_kind: str = "wots",
    compiler_opt: str = "Os",
    seed_id: int = 0,
    min_random_rows: int = 1,
    prefer_largest_dataset: bool = True,
    verbose: bool = True,
) -> DiscoveredCapture:
    """Discover one random-split capture made by the latest capture helper.

    The first search pass follows the exact hierarchy generated by
    capture_multi_seed_fixed_random_experiment().  If BASE points at a higher or
    lower directory, the function recursively scans *_capture_config.json and
    validates parameter set / compiler / address kind / seed id before selecting
    a capture.
    """
    p = normalize_param_set(param_set)
    kind = normalize_address_kind(address_kind)
    opt = normalize_compiler_opt(compiler_opt)
    sid = int(seed_id)
    required_rows = max(1, int(min_random_rows))

    base = Path(windows_to_wsl_path(base_folder)).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"base folder not found: {base}")

    candidates: List[Tuple[Tuple[int, int, float], DiscoveredCapture]] = []
    visited_prefixes = set()

    for root, source in _preferred_search_roots(
        base,
        param_set=p,
        compiler_opt=opt,
        address_kind=kind,
        seed_id=sid,
    ):
        if not root.exists():
            continue

        for prefix in _candidate_capture_prefixes_from_root(root):
            if prefix in visited_prefixes:
                continue
            visited_prefixes.add(prefix)

            config_path = Path(prefix + "_capture_config.json")
            traces_path = Path(prefix + "_traces_random.npy")
            addrs_path = Path(prefix + "_addrs_random.npy")

            if not traces_path.is_file() or not addrs_path.is_file():
                continue

            config: Dict[str, Any] = {}
            if config_path.is_file():
                try:
                    with config_path.open("r", encoding="utf-8") as file:
                        config = json.load(file)
                except Exception:
                    continue

            detected_p = _parameter_from_config_or_path(config, Path(prefix))
            detected_kind = _kind_from_config_or_path(config, Path(prefix))
            detected_opt = _compiler_from_config_or_path(config, Path(prefix))
            detected_sid = _seed_id_from_config_or_path(config, Path(prefix))

            if detected_p != p:
                continue
            if detected_kind is not None and detected_kind != kind:
                continue
            if detected_opt is not None and detected_opt != opt:
                continue
            if detected_sid is not None and detected_sid != sid:
                continue

            try:
                traces = np.load(traces_path, mmap_mode="r")
                addrs = np.load(addrs_path, mmap_mode="r")
            except Exception:
                continue

            if traces.ndim != 2:
                continue
            if addrs.ndim != 2 or addrs.shape[1] != ADRS_FULL_BYTES:
                continue
            if traces.shape[0] != addrs.shape[0]:
                continue
            if int(traces.shape[0]) < required_rows:
                continue

            random_rows = int(traces.shape[0])
            trace_points = int(traces.shape[1])

            # Exact hierarchy is preferred; among equally specific matches the
            # larger dataset and then the newer capture wins.
            specificity = {
                "expected-hierarchy": 5,
                "parameter-root": 4,
                "compiler-root": 3,
                "kind-root": 2,
                "base-recursive": 1,
            }.get(source, 0)
            size_score = random_rows if prefer_largest_dataset else 0
            mtime = traces_path.stat().st_mtime

            record = DiscoveredCapture(
                prefix=prefix,
                config_path=str(config_path) if config_path.is_file() else "",
                parameter_set=p,
                compiler_opt=opt,
                address_kind=kind,
                seed_id=sid,
                random_rows=random_rows,
                trace_points=trace_points,
                match_source=source,
            )
            candidates.append(((specificity, size_score, mtime), record))

        # If the exact seed directory produced a valid match, do not recursively
        # crawl the entire BASE tree.
        if source == "expected-hierarchy" and candidates:
            break

    if not candidates:
        expected = (
            base
            / f"SLH_DSA_{p}_sha2_compiler_compare"
            / opt
            / f"multi_seed_{kind}_fixed_random"
            / f"seed_{sid:03d}"
        )
        raise FileNotFoundError(
            "No compatible capture was found.\n"
            f"  base          : {base}\n"
            f"  parameter set : {p}\n"
            f"  compiler      : {opt}\n"
            f"  address kind  : {kind}\n"
            f"  seed id       : {sid}\n"
            f"  min rows      : {required_rows}\n"
            f"  expected tree : {expected}\n"
            "The loader expects *_traces_random.npy and *_addrs_random.npy "
            "saved by the latest multi-seed fixed/random capture helper."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1]

    if verbose:
        print("===== Capture auto-discovery =====")
        print("base folder   :", base)
        print("parameter set :", selected.parameter_set)
        print("compiler      :", selected.compiler_opt)
        print("address kind  :", selected.address_kind)
        print("seed id       :", f"{selected.seed_id:03d}")
        print("prefix        :", selected.prefix)
        print("random traces :", selected.random_rows)
        print("trace points  :", selected.trace_points)
        print("matched via   :", selected.match_source)

    return selected


def find_shared_seed_manifest(
    base_folder: str | os.PathLike,
    *,
    param_set: str | int,
) -> Optional[str]:
    """Find shared_secret_seeds.json near the selected parameter-set captures."""
    p = normalize_param_set(param_set)
    base = Path(windows_to_wsl_path(base_folder)).expanduser()
    expected = base / f"SLH_DSA_{p}_sha2_compiler_compare" / "shared_secret_seeds.json"
    if expected.is_file():
        return str(expected)

    if base.name.lower() == f"slh_dsa_{p}_sha2_compiler_compare".lower():
        candidate = base / "shared_secret_seeds.json"
        if candidate.is_file():
            return str(candidate)

    matches = []
    for path in base.rglob("shared_secret_seeds.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            saved = data.get("parameter_set")
            if saved is not None and normalize_param_set(saved) == p:
                matches.append(path)
        except Exception:
            continue

    if not matches:
        return None
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(matches[0])


def load_seed_from_shared_manifest(
    base_folder: str | os.PathLike,
    *,
    param_set: str | int,
    seed_id: int = 0,
) -> Optional[bytes]:
    """Load the selected experimental ground-truth seed when a manifest exists."""
    manifest_path = find_shared_seed_manifest(base_folder, param_set=param_set)
    if manifest_path is None:
        return None

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    sid = int(seed_id)
    for index, row in enumerate(manifest.get("seeds", [])):
        row_id = int(row.get("seed_id", index))
        if row_id == sid:
            seed = bytes.fromhex(str(row["sk_seed_hex"]))
            p = normalize_param_set(param_set)
            if len(seed) != PARAM_N[p]:
                raise ValueError(
                    f"manifest seed_{sid:03d} has {len(seed)} bytes; "
                    f"{p} requires {PARAM_N[p]}"
                )
            return seed
    raise KeyError(
        f"seed_{sid:03d} not found in {manifest_path}"
    )




def run_selected_capture(
    *,
    param_set: str | int,
    capture_prefix: str | os.PathLike,
    address_kind: str = "wots",
    attacks: Sequence[str] = ("expansion", "compression"),
    assumed_recovered: Optional[Mapping[str, int]] = None,
    validation_seed: Optional[bytes] = None,
    pk_seed: Optional[bytes] = None,
    max_traces: Optional[int] = 5000,
    expansion_trace_slices: Optional[Mapping[str, Optional[slice]]] = None,
    compression_trace_slices: Optional[Mapping[int, Optional[slice]]] = None,
    compression_target: str = "e_update",
    backend: str = "tensorflow",
    tf_device: Optional[str] = None,
    tf_point_batch: int = 2048,
    tf_cache_traces: bool = True,
    tf_fallback_to_numpy: bool = True,
    candidate_batch: int = 256,
    beam_width: int = 4,
    plot_top: int = 4,
    top_16bit: int = 10,
    joint_stage1_top_classes: int = 10,
    joint_final_top: int = 16,
    downstream_sigma_top_low16: int = 10,
    downstream_sigma_top_final: int = 16,
    downstream_sigma_poi_radius: Union[int, Tuple[int, int]] = 30,
    w11_w12_primary_method: str = "2byte",
    stage_primary_method: Optional[str] = None,
    stage_chaining: str = "assume_previous_success",
    w9_128_primary_source: str = "auto",
    verbose: bool = True,
    result_dir: Optional[str | os.PathLike] = None,
    save_plots: bool = True,
    show_plots: bool = True,
    plot_dpi: int = 300,
    compiler_opt: str = "Os",
    seed_id: int = 0,
) -> Dict[str, Any]:
    """Run selected attacks and save/display figures for every CPA stage."""
    p = normalize_param_set(param_set)
    kind = str(address_kind).strip().lower()
    if kind in ("wots", "wots+"):
        group = "WOTS_RANDOM"
        kind_for_path = "wots"
    elif kind == "fors":
        group = "FORS_RANDOM"
        kind_for_path = "fors"
    elif kind in ("random", "all", "mixed"):
        group = "RANDOM"
        kind_for_path = "random"
    else:
        raise ValueError("address_kind must be 'wots', 'fors', or 'random'")

    requested = {str(v).strip().lower() for v in attacks}
    unknown = requested - {"expansion", "compression"}
    if unknown:
        raise ValueError(f"unknown attacks: {sorted(unknown)}")

    cap = load_capture_legacy(
        capture_prefix,
        param_set=p,
        group=group,
        max_traces=max_traces,
        mmap_mode="r",
    )

    opt = normalize_compiler_opt(compiler_opt)
    plot_root: Optional[Path] = None
    if result_dir is not None:
        base = Path(windows_to_wsl_path(result_dir)).expanduser()
        plot_root = (
            base
            / p
            / opt
            / kind_for_path
            / f"seed_{int(seed_id):03d}"
        )
        if save_plots:
            plot_root.mkdir(parents=True, exist_ok=True)

    plot_config = PlotConfig(
        root=plot_root,
        param_set=p,
        compiler_opt=opt,
        address_kind=kind_for_path,
        seed_id=int(seed_id),
        save_plots=bool(save_plots),
        show_plots=bool(show_plots),
        dpi=int(plot_dpi),
        print_saved_paths=bool(verbose),
    )

    if verbose:
        print("\n================ SELECTED CPA RUN ================")
        print("parameter set :", p)
        print("capture       :", cap.source)
        print("group         :", cap.group)
        print("trace shape   :", cap.traces.shape)
        print("attacks       :", sorted(requested))
        if plot_root is not None:
            print("plot root     :", plot_root)

    engine_options = {
        "backend": backend,
        "tf_device": tf_device,
        "point_batch": tf_point_batch,
        "cache_traces": tf_cache_traces,
        "fallback_to_numpy": tf_fallback_to_numpy,
    }

    output: Dict[str, Any] = {
        "parameter_set": p,
        "capture": cap,
        "plot_root": None if plot_root is None else str(plot_root),
    }

    if "expansion" in requested:
        output["expansion"] = run_expansion_cpa(
            cap,
            assumed_recovered=assumed_recovered,
            validation_seed=validation_seed,
            trace_slices=expansion_trace_slices,
            engine_options=engine_options,
            candidate_batch=candidate_batch,
            beam_width=beam_width,
            plot_top=plot_top,
            top_16bit=top_16bit,
            joint_stage1_top_classes=joint_stage1_top_classes,
            joint_final_top=joint_final_top,
            downstream_sigma_top_low16=downstream_sigma_top_low16,
            downstream_sigma_top_final=downstream_sigma_top_final,
            downstream_sigma_poi_radius=downstream_sigma_poi_radius,
            w11_w12_primary_method=w11_w12_primary_method,
            stage_primary_method=stage_primary_method,
            stage_chaining=stage_chaining,
            w9_128_primary_source=w9_128_primary_source,
            verbose=verbose,
            plot_config=plot_config,
        )

    if "compression" in requested:
        pk = get_default_pk_seed(p) if pk_seed is None else bytes(pk_seed)
        output["compression"] = run_compression_cpa(
            cap,
            target=compression_target,
            pk_seed=pk,
            validation_seed=validation_seed,
            trace_slices=compression_trace_slices,
            engine_options=engine_options,
            beam_width=beam_width,
            candidate_batch=candidate_batch,
            plot_top=plot_top,
            verbose=verbose,
            plot_config=plot_config,
        )

    return output


def list_available_captures(
    base_folder: str | os.PathLike,
    *,
    param_sets: Optional[Sequence[str | int]] = None,
    compiler_opts: Optional[Sequence[str]] = None,
    address_kinds: Optional[Sequence[str]] = None,
    seed_ids: Optional[Sequence[int]] = None,
    min_random_rows: int = 1,
    verbose: bool = True,
) -> List[DiscoveredCapture]:
    """Enumerate all compatible captures under BASE.

    This helper scans captures produced by the latest multi-seed fixed/random
    acquisition script and filters them by parameter set / compiler / address
    kind / seed id.  It is intended for automated batch runs.
    """
    base = Path(windows_to_wsl_path(base_folder)).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"base folder not found: {base}")

    allowed_params = None if param_sets is None else {
        normalize_param_set(value) for value in param_sets
    }
    allowed_opts = None if compiler_opts is None else {
        normalize_compiler_opt(value) for value in compiler_opts
    }
    allowed_kinds = None if address_kinds is None else {
        normalize_address_kind(value) for value in address_kinds
    }
    allowed_seeds = None if seed_ids is None else {int(value) for value in seed_ids}
    min_rows = max(1, int(min_random_rows))

    prefixes = _candidate_capture_prefixes_from_root(base)
    found: List[DiscoveredCapture] = []
    seen = set()

    for prefix in prefixes:
        config_path = Path(prefix + '_capture_config.json')
        traces_path = Path(prefix + '_traces_random.npy')
        addrs_path = Path(prefix + '_addrs_random.npy')
        if not traces_path.is_file() or not addrs_path.is_file():
            continue

        config = {}
        if config_path.is_file():
            try:
                with config_path.open('r', encoding='utf-8') as fh:
                    config = json.load(fh)
            except Exception:
                continue

        p = _parameter_from_config_or_path(config, Path(prefix))
        kind = _kind_from_config_or_path(config, Path(prefix))
        opt = _compiler_from_config_or_path(config, Path(prefix))
        sid = _seed_id_from_config_or_path(config, Path(prefix))
        if p is None or kind is None or opt is None or sid is None:
            continue
        if allowed_params is not None and p not in allowed_params:
            continue
        if allowed_opts is not None and opt not in allowed_opts:
            continue
        if allowed_kinds is not None and kind not in allowed_kinds:
            continue
        if allowed_seeds is not None and sid not in allowed_seeds:
            continue

        try:
            traces = np.load(traces_path, mmap_mode='r')
            addrs = np.load(addrs_path, mmap_mode='r')
        except Exception:
            continue
        if traces.ndim != 2 or addrs.ndim != 2:
            continue
        if addrs.shape[1] != ADRS_FULL_BYTES or traces.shape[0] != addrs.shape[0]:
            continue
        rows = int(traces.shape[0])
        cols = int(traces.shape[1])
        if rows < min_rows:
            continue

        key = (p, opt, kind, int(sid), prefix)
        if key in seen:
            continue
        seen.add(key)
        found.append(
            DiscoveredCapture(
                prefix=prefix,
                config_path=str(config_path) if config_path.is_file() else '',
                parameter_set=p,
                compiler_opt=opt,
                address_kind=kind,
                seed_id=int(sid),
                random_rows=rows,
                trace_points=cols,
                match_source='batch-scan',
            )
        )

    found.sort(key=lambda x: (x.parameter_set, x.compiler_opt, x.address_kind, x.seed_id, x.prefix))
    if verbose:
        print('===== Available captures =====')
        print('base folder:', base)
        print('count      :', len(found))
        for item in found:
            print(
                f"{item.parameter_set:4s}  {item.compiler_opt:2s}  {item.address_kind:4s}  "
                f"seed_{item.seed_id:03d}  traces={item.random_rows:5d}  prefix={item.prefix}"
            )
    return found


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "estimating"
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _checkpoint_key(capture: DiscoveredCapture) -> str:
    return (
        f"{capture.parameter_set}|{capture.compiler_opt}|"
        f"{capture.address_kind}|{capture.seed_id:03d}"
    )


def _load_batch_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"version": 2, "tasks": {}}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("checkpoint root is not an object")
        data.setdefault("version", 2)
        data.setdefault("tasks", {})
        return data
    except Exception:
        # Do not destroy an unreadable checkpoint; start a new in-memory state.
        return {"version": 2, "tasks": {}}


def _save_batch_checkpoint(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _attack_checkpoint_summary(result: Any, attack: str, duration: float) -> Dict[str, Any]:
    attack = str(attack).lower()
    row: Dict[str, Any] = {
        "status": "completed",
        "duration_seconds": float(duration),
        "completed_at_unix": float(time.time()),
        "error": "",
    }
    if attack == "expansion":
        row["recovered_seed_hex"] = result.recovered_seed.hex()
        row["validation_match"] = (
            None if result.validation_seed is None
            else bool(result.recovered_seed == result.validation_seed)
        )
        summary = dict(getattr(result, "method_summary", {}) or {})
        if summary:
            row["stage_scored"] = int(summary.get("scored_stages", 0))
            row["stage_bytewise_success"] = int(summary.get("bytewise_success", 0))
            row["stage_2byte_success"] = int(summary.get("two_byte_success", 0))
            row["stage_selected_success"] = int(summary.get("selected_success", 0))
            row["stage_methods_agree"] = int(summary.get("methods_agree", 0))
            row["stage_dual_method"] = int(summary.get("dual_method_stages", 0))
            row["stage_split_scored"] = int(summary.get("split_scored_stages", 0))
            row["stage_joint_scored"] = int(summary.get("joint_scored", 0))
            row["stage_joint_success"] = int(summary.get("joint_success", 0))
            row["stage_failed_selected"] = ";".join(summary.get("failed_stages_selected", []) or [])
            row["stage_detail"] = str(summary.get("primary_detail", summary.get("detail", "")))
            row["stage_alt_scored"] = int(summary.get("alternative_scored", 0))
            row["stage_alt_bytewise_success"] = int(summary.get("alternative_bytewise_success", 0))
            row["stage_alt_2byte_success"] = int(summary.get("alternative_two_byte_success", 0))
            row["stage_alt_detail"] = str(summary.get("alternative_detail", ""))
        stage_info = dict(getattr(result, "stage_info", {}) or {})
        row["w9_128_primary_source"] = str(stage_info.get("w9_128_primary_source", ""))
        row["stage_chaining"] = str(stage_info.get("stage_chaining", ""))
        row["full_recovery_no_secret_assumptions"] = True
        joint = dict(stage_info.get("w5_w13_joint", {}) or {})
        row["joint_stage1_true_class_rank"] = joint.get("stage1_true_class_rank", "")
        row["joint_stage1_true_class_retained"] = joint.get("stage1_true_class_retained", "")
        row["joint_final_true_pair_rank"] = joint.get("final_true_pair_rank", "")
        row["joint_pair_match"] = joint.get("pair_match", "")
    elif attack == "compression":
        row["target"] = result.target
        row["recovered_seed_hex"] = result.recovered_seed.hex()
        row["validation_match"] = (
            None if result.validation_seed is None
            else bool(result.recovered_seed == result.validation_seed)
        )
    return row


def _checkpoint_summary_row(
    capture: DiscoveredCapture,
    task: Mapping[str, Any],
    *,
    attacks: Sequence[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "parameter_set": capture.parameter_set,
        "compiler_opt": capture.compiler_opt,
        "address_kind": capture.address_kind,
        "seed_id": capture.seed_id,
        "trace_count": capture.random_rows,
        "trace_points": capture.trace_points,
        "capture_prefix": capture.prefix,
        "error": "",
    }
    attack_map = dict(task.get("attacks", {}))
    errors: List[str] = []
    for attack in [str(a).lower() for a in attacks]:
        info = dict(attack_map.get(attack, {}))
        status = info.get("status", "pending")
        row[f"{attack}_status"] = status
        row[f"{attack}_duration_seconds"] = info.get("duration_seconds", "")
        row[f"{attack}_recovered_seed_hex"] = info.get("recovered_seed_hex", "")
        match = info.get("validation_match", "")
        row[f"{attack}_validation_match"] = (
            "" if match is None or match == "" else int(bool(match))
        )
        if attack == "expansion":
            for key in (
                "stage_scored",
                "stage_bytewise_success",
                "stage_2byte_success",
                "stage_selected_success",
                "stage_methods_agree",
                "stage_dual_method",
                "stage_split_scored",
                "stage_joint_scored",
                "stage_joint_success",
                "stage_failed_selected",
                "stage_detail",
                "stage_alt_scored",
                "stage_alt_bytewise_success",
                "stage_alt_2byte_success",
                "stage_alt_detail",
                "stage_chaining",
                "w9_128_primary_source",
                "full_recovery_no_secret_assumptions",
                "joint_stage1_true_class_rank",
                "joint_stage1_true_class_retained",
                "joint_final_true_pair_rank",
                "joint_pair_match",
            ):
                row[f"expansion_{key}"] = info.get(key, "")
        if attack == "compression":
            row["compression_target"] = info.get("target", "")
        if info.get("error"):
            errors.append(f"{attack}: {info['error']}")
    row["error"] = " | ".join(errors)
    return row


def _xlsx_col_name(index1: int) -> str:
    n = int(index1)
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _write_rows_xlsx(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a simple Excel-compatible .xlsx summary with stdlib only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        fieldnames = ["no_data"]
        rows = [{"no_data": ""}]

    def cell_xml(ref: str, value: Any, style: int = 0) -> str:
        style_attr = f' s="{style}"' if style else ""
        if value is None or value == "":
            return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t></t></is></c>'
        if isinstance(value, (bool, np.bool_)):
            return f'<c r="{ref}"{style_attr} t="b"><v>{1 if bool(value) else 0}</v></c>'
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
        text = _xml_escape(str(value))
        return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'

    sheet_rows: List[str] = []
    header_cells = [cell_xml(f"{_xlsx_col_name(i+1)}1", name, 1) for i, name in enumerate(fieldnames)]
    sheet_rows.append(f'<row r="1">{"".join(header_cells)}</row>')
    for r_idx, row in enumerate(rows, start=2):
        cells = [cell_xml(f"{_xlsx_col_name(c_idx+1)}{r_idx}", row.get(name, "")) for c_idx, name in enumerate(fieldnames)]
        sheet_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    widths = []
    for i, name in enumerate(fieldnames, start=1):
        sample = [len(str(name))]
        for row in list(rows)[:200]:
            sample.append(len(str(row.get(name, ""))))
        width = min(60, max(10, max(sample, default=10) + 2))
        widths.append(f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<cols>{"".join(widths)}</cols>'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        f'<autoFilter ref="A1:{_xlsx_col_name(len(fieldnames))}{max(1, len(rows)+1)}"/>'
        '</worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Summary" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )

    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/styles.xml", styles_xml)
    os.replace(tmp, path)

def _write_batch_summary_csv(
    path: Path,
    captures: Sequence[DiscoveredCapture],
    checkpoint: Mapping[str, Any],
    *,
    attacks: Sequence[str],
) -> List[Dict[str, Any]]:
    tasks = dict(checkpoint.get("tasks", {}))
    rows = [
        _checkpoint_summary_row(
            cap,
            tasks.get(_checkpoint_key(cap), {}),
            attacks=attacks,
        )
        for cap in captures
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_rows_xlsx(path.with_suffix(".xlsx"), rows)
    return rows


def _estimated_remaining_seconds(
    checkpoint: Mapping[str, Any],
    captures: Sequence[DiscoveredCapture],
    attacks: Sequence[str],
) -> Optional[float]:
    tasks = dict(checkpoint.get("tasks", {}))
    duration_by_attack: Dict[str, List[float]] = {str(a).lower(): [] for a in attacks}
    global_durations: List[float] = []
    for task in tasks.values():
        for attack, info in dict(task.get("attacks", {})).items():
            if info.get("status") == "completed":
                try:
                    duration = float(info.get("duration_seconds"))
                except Exception:
                    continue
                if duration > 0 and np.isfinite(duration):
                    global_durations.append(duration)
                    if attack in duration_by_attack:
                        duration_by_attack[attack].append(duration)
    global_avg = None if not global_durations else float(np.mean(global_durations))
    estimate = 0.0
    remaining = 0
    for cap in captures:
        task = tasks.get(_checkpoint_key(cap), {})
        amap = dict(task.get("attacks", {}))
        for attack in [str(a).lower() for a in attacks]:
            if dict(amap.get(attack, {})).get("status") == "completed":
                continue
            remaining += 1
            values = duration_by_attack.get(attack, [])
            if values:
                estimate += float(np.mean(values))
            elif global_avg is not None:
                estimate += global_avg
            else:
                return None
    return 0.0 if remaining == 0 else estimate


def run_batch_from_base_folder(
    *,
    base_folder: str | os.PathLike,
    param_sets: Optional[Sequence[str | int]] = None,
    compiler_opts: Sequence[str] = ("Os",),
    address_kinds: Sequence[str] = ("wots", "fors"),
    seed_ids: Optional[Sequence[int]] = None,
    auto_discover_all_seeds: bool = True,
    attacks: Sequence[str] = ("expansion", "compression"),
    use_shared_seed_manifest_for_validation: bool = False,
    use_validation_for_conditional_assumptions: bool = False,
    assumptions_by_param: Optional[Mapping[str, Mapping[str, int]]] = None,
    result_dir: Optional[str | os.PathLike] = None,
    summary_csv_name: str = "batch_summary_192_known_w9_w10_w11.csv",
    checkpoint_json_name: str = "batch_progress_192_known_w9_w10_w11.json",
    resume: bool = True,
    overwrite_existing_results: bool = False,
    continue_on_error: bool = True,
    detailed_output: bool = False,
    progress_output: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run CPA across captures with attack-level checkpoint/resume and ETA.

    A task is one (capture, attack) pair. The checkpoint is written immediately
    after each completed expansion/compression task. If the kernel or process is
    restarted, completed tasks are skipped and only unfinished tasks are run.

    detailed_output=False suppresses per-byte/per-stage logging. Overall batch
    progress is shown on a single updating line to avoid large notebook output.
    """
    params = (
        ["128f", "128s", "192f", "192s", "256f", "256s"]
        if param_sets is None else [normalize_param_set(x) for x in param_sets]
    )
    opts = [normalize_compiler_opt(x) for x in compiler_opts]
    kinds = [normalize_address_kind(x) for x in address_kinds]
    seeds = None if seed_ids is None else [int(x) for x in seed_ids]
    requested_attacks = [str(a).strip().lower() for a in attacks]
    unknown = set(requested_attacks) - {"expansion", "compression"}
    if unknown:
        raise ValueError(f"unknown attacks: {sorted(unknown)}")

    captures = list_available_captures(
        base_folder,
        param_sets=params,
        compiler_opts=opts,
        address_kinds=kinds,
        seed_ids=None if auto_discover_all_seeds and seeds is None else seeds,
        min_random_rows=max(1, int(kwargs.get("max_traces", 1) or 1)),
        verbose=False,
    )
    if not captures:
        raise FileNotFoundError("No compatible captures were found for the requested batch conditions.")
    if seeds is not None:
        allowed = set(seeds)
        captures = [item for item in captures if item.seed_id in allowed]

    root = (
        Path(windows_to_wsl_path(result_dir)).expanduser()
        if result_dir is not None
        else Path(windows_to_wsl_path(base_folder)).expanduser()
    )
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / checkpoint_json_name
    summary_path = root / summary_csv_name
    summary_xlsx_path = summary_path.with_suffix(".xlsx")

    if overwrite_existing_results:
        for stale in (checkpoint_path, summary_path, summary_xlsx_path):
            if stale.exists():
                stale.unlink()
        resume = False

    checkpoint = (
        _load_batch_checkpoint(checkpoint_path)
        if resume
        else {"version": 3, "variant": CPA_HELPER_VARIANT, "tasks": {}}
    )
    old_variant = checkpoint.get("variant")
    if resume and old_variant not in (None, CPA_HELPER_VARIANT):
        raise ValueError(
            f"Existing checkpoint variant {old_variant!r} differs from {CPA_HELPER_VARIANT!r}. "
            "Set overwrite_existing_results=True to replace the old conditional results in place."
        )
    if resume and old_variant is None and checkpoint.get("tasks"):
        raise ValueError(
            "Existing checkpoint predates the full-recovery variant. "
            "Set overwrite_existing_results=True to overwrite it at the same path."
        )
    checkpoint["version"] = 3
    checkpoint["variant"] = CPA_HELPER_VARIANT
    checkpoint["full_recovery_no_secret_assumptions"] = True
    tasks = checkpoint.setdefault("tasks", {})

    # If a capture path changed, do not reuse an old completion record blindly.
    for cap in captures:
        key = _checkpoint_key(cap)
        task = tasks.setdefault(key, {"capture_prefix": cap.prefix, "attacks": {}})
        if task.get("capture_prefix") != cap.prefix:
            tasks[key] = {"capture_prefix": cap.prefix, "attacks": {}}

    total_units = len(captures) * len(requested_attacks)
    completed_before = 0
    for cap in captures:
        amap = dict(tasks[_checkpoint_key(cap)].get("attacks", {}))
        completed_before += sum(
            1 for attack in requested_attacks
            if dict(amap.get(attack, {})).get("status") == "completed"
        )

    failures: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    session_start = time.perf_counter()
    done_units = completed_before
    skipped_units = completed_before
    cleared_case_keys: set[str] = set()

    def show_progress(current: str = "") -> None:
        if not progress_output:
            return
        percent = 100.0 if total_units == 0 else (100.0 * done_units / total_units)
        elapsed = time.perf_counter() - session_start
        eta = _estimated_remaining_seconds(checkpoint, captures, requested_attacks)
        line = (
            f"[CPA batch] {percent:6.2f}% | {done_units}/{total_units} tasks | "
            f"session {_format_seconds(elapsed)} | ETA {_format_seconds(eta)}"
        )
        if current:
            line += f" | {current}"
        print(line.ljust(180), end="\r", flush=True)

    if progress_output:
        print(
            f"Batch: {len(captures)} captures x {len(requested_attacks)} attacks = "
            f"{total_units} tasks; resume-skip={completed_before}."
        )
    show_progress("starting")

    for cap in captures:
        key = _checkpoint_key(cap)
        task = tasks[key]
        assumptions = None
        if assumptions_by_param is not None:
            assumptions = assumptions_by_param.get(cap.parameter_set)

        if overwrite_existing_results and key not in cleared_case_keys and result_dir is not None:
            case_root = (
                Path(windows_to_wsl_path(result_dir)).expanduser()
                / cap.parameter_set / cap.compiler_opt / cap.address_kind / f"seed_{cap.seed_id:03d}"
            )
            if case_root.exists():
                shutil.rmtree(case_root)
            cleared_case_keys.add(key)

        for attack in requested_attacks:
            info = dict(task.setdefault("attacks", {}).get(attack, {}))
            if resume and info.get("status") == "completed":
                continue

            label = (
                f"{cap.parameter_set}/{cap.address_kind}/"
                f"seed_{cap.seed_id:03d}/{attack}"
            )
            show_progress(label)
            started = time.perf_counter()

            run_kwargs = dict(kwargs)
            run_kwargs["verbose"] = bool(detailed_output)
            try:
                run_result = run_from_base_folder(
                    base_folder=base_folder,
                    selected_param_set=cap.parameter_set,
                    selected_address_kind=cap.address_kind,
                    compiler_opt=cap.compiler_opt,
                    seed_id=cap.seed_id,
                    attacks=(attack,),
                    assumed_recovered=assumptions,
                    validation_seed=None,
                    use_shared_seed_manifest_for_validation=use_shared_seed_manifest_for_validation,
                    use_validation_for_conditional_assumptions=use_validation_for_conditional_assumptions,
                    result_dir=result_dir,
                    verbose_discovery=False,
                    **run_kwargs,
                )
                duration = time.perf_counter() - started
                attack_result = run_result[attack]
                task["attacks"][attack] = _attack_checkpoint_summary(
                    attack_result, attack, duration
                )
                task["capture_prefix"] = cap.prefix
                task["updated_at_unix"] = float(time.time())
                _save_batch_checkpoint(checkpoint_path, checkpoint)
                _write_batch_summary_csv(
                    summary_path, captures, checkpoint, attacks=requested_attacks
                )
                results.append(run_result)
                done_units += 1
                show_progress(label)
            except Exception as exc:
                duration = time.perf_counter() - started
                message = f"{type(exc).__name__}: {exc}"
                task["attacks"][attack] = {
                    "status": "failed",
                    "duration_seconds": float(duration),
                    "completed_at_unix": float(time.time()),
                    "error": message,
                }
                task["updated_at_unix"] = float(time.time())
                _save_batch_checkpoint(checkpoint_path, checkpoint)
                _write_batch_summary_csv(
                    summary_path, captures, checkpoint, attacks=requested_attacks
                )
                failures.append({"capture": cap, "attack": attack, "error": message})
                if progress_output:
                    print(" " * 180, end="\r")
                    print(f"[FAIL] {label}: {message}")
                if continue_on_error:
                    continue
                raise

    rows = _write_batch_summary_csv(
        summary_path, captures, checkpoint, attacks=requested_attacks
    )
    if progress_output:
        show_progress("complete")
        print()
        elapsed = time.perf_counter() - session_start
        completed_now = done_units - skipped_units
        print(
            f"Batch complete: newly completed={completed_now}, skipped={skipped_units}, "
            f"failures={len(failures)}, session={_format_seconds(elapsed)}"
        )
        print("checkpoint:", checkpoint_path)
        print("summary   :", summary_path)
        print("excel     :", summary_xlsx_path)

    return {
        "captures": captures,
        "results": results,
        "rows": rows,
        "failures": failures,
        "summary_csv": str(summary_path),
        "summary_xlsx": str(summary_xlsx_path),
        "checkpoint_json": str(checkpoint_path),
        "completed_before": completed_before,
        "completed_after": done_units,
        "total_tasks": total_units,
    }


def run_from_base_folder(
    *,
    base_folder: str | os.PathLike,
    selected_param_set: str | int,
    selected_address_kind: str = "wots",
    compiler_opt: str = "Os",
    seed_id: int = 0,
    attacks: Sequence[str] = ("expansion", "compression"),
    assumed_recovered: Optional[Mapping[str, int]] = None,
    validation_seed: Optional[bytes] = None,
    use_shared_seed_manifest_for_validation: bool = False,
    use_validation_for_conditional_assumptions: bool = False,
    max_traces: Optional[int] = 5000,
    verbose_discovery: bool = True,
    result_dir: Optional[str | os.PathLike] = None,
    save_plots: bool = True,
    show_plots: bool = True,
    plot_dpi: int = 300,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Auto-discover a capture, run CPA, and save all stage figures."""
    p = normalize_param_set(selected_param_set)
    kind = normalize_address_kind(selected_address_kind)
    opt = normalize_compiler_opt(compiler_opt)

    min_rows = 1 if max_traces is None else int(max_traces)
    discovered = discover_capture_prefix(
        base_folder,
        param_set=p,
        address_kind=kind,
        compiler_opt=opt,
        seed_id=int(seed_id),
        min_random_rows=min_rows,
        verbose=verbose_discovery,
    )

    validation = None if validation_seed is None else bytes(validation_seed)
    if validation is None and bool(use_shared_seed_manifest_for_validation):
        validation = load_seed_from_shared_manifest(
            base_folder,
            param_set=p,
            seed_id=int(seed_id),
        )
        if validation is None:
            raise FileNotFoundError(
                "use_shared_seed_manifest_for_validation=True, but "
                "shared_secret_seeds.json was not found"
            )
        if verbose_discovery:
            print("validation seed: loaded from shared_secret_seeds.json")

    assumptions = None if assumed_recovered is None else dict(assumed_recovered)
    if bool(use_validation_for_conditional_assumptions):
        raise ValueError(
            "Conditional secret assumptions are disabled in the full-recovery variant. "
            "Keep use_validation_for_conditional_assumptions=False; validation remains scoring-only."
        )

    result = run_selected_capture(
        param_set=p,
        capture_prefix=discovered.prefix,
        address_kind=kind,
        attacks=attacks,
        assumed_recovered=assumptions,
        validation_seed=validation,
        max_traces=max_traces,
        result_dir=result_dir,
        save_plots=save_plots,
        show_plots=show_plots,
        plot_dpi=plot_dpi,
        compiler_opt=opt,
        seed_id=int(seed_id),
        **kwargs,
    )
    result["discovered_capture"] = discovered
    result["compiler_opt"] = opt
    result["seed_id"] = int(seed_id)
    return result

def run_from_capture_registry(
    *,
    selected_param_set: str | int,
    selected_address_kind: str,
    capture_registry: Mapping[str, Mapping[str, str]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Select one of six parameter sets from a user-populated path registry.

    Example:
        CAPTURE_PREFIXES = {
            "128f": {"wots": r"...prefix...", "fors": r"...prefix..."},
            "128s": {"wots": r"...prefix...", "fors": r"...prefix..."},
            ...
        }
    """
    p = normalize_param_set(selected_param_set)
    kind = str(selected_address_kind).strip().lower()
    if kind == "wots+":
        kind = "wots"

    if p not in capture_registry:
        raise KeyError(f"capture_registry does not contain {p}")
    if kind not in capture_registry[p]:
        raise KeyError(
            f"capture_registry[{p!r}] does not contain address kind {kind!r}"
        )

    prefix = capture_registry[p][kind]
    if not prefix:
        raise ValueError(
            f"capture path for {p}/{kind} is empty"
        )

    return run_selected_capture(
        param_set=p,
        capture_prefix=prefix,
        address_kind=kind,
        **kwargs,
    )


def self_test_downstream_sigma1_relation() -> bool:
    """Check the exact low16 -> sigma1[22:28] relation used in Stage 1."""
    rng = np.random.default_rng(20260818)
    full = rng.integers(0, 2**32, size=1024, dtype=np.uint32)
    low = full & np.uint32(0xFFFF)
    exact = (
        (sigma1_32(full) >> np.uint32(22))
        & np.uint32(0x7F)
    ).astype(np.uint8)
    partial = _sigma1_low16_partial_value(low)
    if not np.array_equal(exact, partial):
        raise AssertionError("partial sigma1 low16 relation does not match full sigma1")
    return True
