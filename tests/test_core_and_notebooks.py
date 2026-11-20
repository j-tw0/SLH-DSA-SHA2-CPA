from pathlib import Path
import ast
import hashlib
import json
import nbformat
import re
import sha256_prf_cpa_core as core
import signature_cpa_core as paper
import random_address_cpa as random_runner
from keygen_dataset import ALL_PARAMS, keygen_address, params_info

ROOT = Path(__file__).resolve().parents[1]

def test_cpa_layout_and_paper_settings():
    assert core.self_test_layouts()
    assert core.self_test_joint_w5_w13_split()
    paper.configure_paper_cpa()
    assert core.CUMULATIVE_HW is True
    assert core.COMPRESSION_PART_BITS == 8
    assert core.DUAL_METHOD_MODE == "selected"
    assert random_runner.argument_parser().parse_args(["--base","/temporary"]).max_traces == 5000

def test_public_demo_constants_match_capture_implementation():
    for p in ALL_PARAMS:
        bits=p[:3]
        n=core.PARAM_N[p]
        pub=hashlib.shake_256(f"SLH-DSA-CAPTURE-PUBLIC-DEMO/{bits}/pk_seed/{n}".encode()).digest(n)
        assert core.get_default_pk_seed(p)==pub
        sec=hashlib.shake_256(f"SLH-DSA-CAPTURE-PUBLIC-DEMO/{bits}/sk_seed/{n}".encode()).digest(n)
        assert core.get_default_sk_seed(p)==sec

def test_keygen_address_structure_and_counts():
    expected={"128f":280,"128s":17920,"192f":408,"192s":26112,"256f":1072,"256s":17152}
    for p in ALL_PARAMS:
        spec=params_info(p)
        assert spec["wots_prf_calls"] == expected[p]
        a=keygen_address(p,0)
        assert len(a)==32 and a[0]==spec["top_layer"] and a[9]==5 and a[17]==0
        last=keygen_address(p,spec["wots_prf_calls"]-1)
        assert int.from_bytes(last[10:14],"big")==(1<<spec["h_prime"])-1
        assert last[17]==spec["wots_len"]-1

def test_three_jupyter_notebooks_valid_and_clean():
    notebooks=sorted((ROOT/"notebooks").glob("*.ipynb"))
    assert len(notebooks)==3
    for path in notebooks:
        nb=nbformat.read(path,as_version=4)
        nbformat.validate(nb)
        assert all(not c.get("outputs") for c in nb.cells if c.cell_type=="code")
        assert all(c.get("execution_count") is None for c in nb.cells if c.cell_type=="code")
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert re.search(r"[A-Za-z]:[/\\]", cell.source) is None
            if cell.cell_type=="code":
                ast.parse(cell.source)

def test_no_original_experimental_seed_constants():
    text=''.join(p.read_text(encoding="utf8") for p in (ROOT/"src").glob("*.py"))
    assert "SLH-DSA-CAPTURE-PUBLIC-DEMO" in text
