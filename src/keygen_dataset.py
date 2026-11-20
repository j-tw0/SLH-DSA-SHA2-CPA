"""Read-only validation of key-generation WOTS_PRF capture datasets.

The dataset schema and FIPS 205 address order are from the capture repository.
This module contains NO trace-acquisition or secret-key-generation code. It
allows the CPA repository to work independently of the ChipWhisperer runtime.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Mapping
import numpy as np

ALL_PARAMS = ("128f", "128s", "192f", "192s", "256f", "256s")
VERSION = "2026-11-13-keygen-wots-prf-sequence-v1"
PARAMS = {
    "128f": (16, 22, 3), "128s": (16, 7, 9),
    "192f": (24, 22, 3), "192s": (24, 7, 9),
    "256f": (32, 17, 4), "256s": (32, 8, 8),
}
REQUIRED_CAPTURE_SUFFIXES = (
    "_traces_random.npy", "_addrs_random.npy", "_params_random.npy",
    "_meta_columns.npy", "_capture_config.json", "_keygen_sequence.json",
)

def native_path(path: str | os.PathLike) -> Path:
    s = str(path)
    if os.name != "nt" and len(s) >= 3 and s[1] == ":" and s[2] in ("/", "\\"):
        s = f"/mnt/{s[0].lower()}/{s[3:].replace(chr(92), '/')}"
    return Path(s).expanduser()

def read_json(path: str | os.PathLike) -> dict:
    val = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(val, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return val

def write_json_atomic(path: str | os.PathLike, obj: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + ".partial")
    temp.write_text(json.dumps(dict(obj), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temp, p)

def _wots_len(n: int) -> int:
    l1 = (8 * n) // 4
    x, l2 = l1 * 15, 0
    while x:
        x //= 16
        l2 += 1
    return l1 + l2

def params_info(param_set: str) -> dict:
    if param_set not in PARAMS:
        raise ValueError(f"Unsupported parameter set: {param_set}")
    n, d, h_prime = PARAMS[param_set]
    length = _wots_len(n)
    return {
        "parameter_set":param_set, "n":n, "d":d, "h_prime":h_prime,
        "top_layer":d-1, "tree":0, "wots_len":length,
        "wots_keypairs":1 << h_prime, "wots_prf_calls":(1 << h_prime)*length,
        "address_type":5, "hash_addr":0,
    }

def capture_condition_dir(root: str | os.PathLike, p: str, opt: str) -> Path:
    if opt not in ("O0","O1","O2","O3","Os"):
        raise ValueError(f"Unsupported compiler option: {opt}")
    return native_path(root) / f"SLH_DSA_{p}_sha2_keygen_prf_sequence" / opt

def keygen_address(param_set: str, index: int) -> bytes:
    spec = params_info(param_set)
    if not 0 <= int(index) < spec["wots_prf_calls"]:
        raise IndexError(index)
    kp, chain = divmod(int(index), spec["wots_len"])
    addr = bytearray(32)
    addr[0] = spec["top_layer"]                 # SHA2 compressed ADRS layer
    addr[9] = 5                                  # WOTS_PRF type
    addr[10:14] = kp.to_bytes(4,"big")           # keypair
    addr[17] = chain                             # chain number
    return bytes(addr)

def _validate_completed(prefix: Path, expected_rows: int, points: int, param: str) -> bool:
    """Same full-size and per-address validation contract as capture module."""
    try:
        prefix = Path(prefix)
        if not all(Path(str(prefix) + suf).is_file() for suf in REQUIRED_CAPTURE_SUFFIXES):
            return False
        t = np.load(str(prefix) + "_traces_random.npy", mmap_mode="r", allow_pickle=False)
        a = np.load(str(prefix) + "_addrs_random.npy", mmap_mode="r", allow_pickle=False)
        m = np.load(str(prefix) + "_params_random.npy", mmap_mode="r", allow_pickle=False)
        cfg = read_json(str(prefix) + "_capture_config.json")
        seq = read_json(str(prefix) + "_keygen_sequence.json")
        if (t.shape != (expected_rows, points) or a.shape != (expected_rows,32)
                or m.shape != (expected_rows,10) or a.dtype != np.uint8
                or t.dtype != np.float32 or m.dtype != np.uint64
                or cfg.get("parameter_set") != param or seq.get("format") != VERSION
                or seq.get("calls") != expected_rows):
            return False
        # Stream-validate all recorded addresses; do not materialize a huge copy.
        for row in range(expected_rows):
            if not np.array_equal(a[row], np.frombuffer(keygen_address(param,row),dtype=np.uint8)):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
