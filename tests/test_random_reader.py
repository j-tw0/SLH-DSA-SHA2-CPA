import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import random_address_cpa as random_runner


def test_random_dataset_preflight():
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory)/"captures"
        paramroot=base/"SLH_DSA_128f_sha2_compiler_compare"
        root=paramroot/"Os"/"multi_seed_fors_fixed_random"/"seed_000"
        root.mkdir(parents=True)
        prefix=root/"capture"
        n=8
        np.save(str(prefix)+"_traces_random.npy", np.zeros((n,40),dtype=np.float32))
        addrs=np.zeros((n,32),dtype=np.uint8);addrs[:,9]=6
        np.save(str(prefix)+"_addrs_random.npy",addrs)
        np.save(str(prefix)+"_params_random.npy",np.zeros((n,10),dtype=np.uint64))
        np.save(str(prefix)+"_meta_columns.npy",np.asarray(["is_fixed","mode","kind","layer","tree","keypair","chain","hash","tree_height","tree_index"],dtype=object))
        Path(str(prefix)+"_capture_config.json").write_text(json.dumps({
          "parameter_set":"128f","n":16,"build":{
            "compiler_optimization":"Os","seed_id":0,"multi_seed_kind":"fors"}}),encoding="utf8")
        seed=bytes(range(16))
        (paramroot/"shared_secret_seeds.json").write_text(json.dumps({"parameter_set":"128f","n":16,
           "seeds":[{"seed_id":0,"sk_seed_hex":seed.hex()}]}),encoding="utf8")
        result=random_runner.main(["--base",str(base),"--output",str(Path(directory)/"out"),
                   "--params","128f","--opts","Os","--kinds","fors","--seed-ids","0",
                   "--max-traces","5","--dry-run"])
        assert result["inspected"]==1
        csv=(Path(directory)/"out"/"input_inventory.csv").read_text(encoding="utf8")
        assert "ready" in csv and "128f" in csv
