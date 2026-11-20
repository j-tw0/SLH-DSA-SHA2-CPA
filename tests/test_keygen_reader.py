import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import keygen_wots_cpa as runner
import keygen_dataset as metadata


def test_keygen_dataset_preflight():
    p="128f"; opt="Os"; seed=bytes(range(16))
    pub=bytes(range(16,32))
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory)/"data"
        prefix=base/f"SLH_DSA_{p}_sha2_keygen_prf_sequence"/opt/"seed_000"/"capture"
        prefix.parent.mkdir(parents=True)
        n=metadata.params_info(p)["wots_prf_calls"]
        np.save(str(prefix)+"_traces_random.npy",np.zeros((n,12),dtype=np.float32))
        np.save(str(prefix)+"_addrs_random.npy",np.asarray([list(metadata.keygen_address(p,i)) for i in range(n)],dtype=np.uint8))
        np.save(str(prefix)+"_params_random.npy",np.zeros((n,10),dtype=np.uint64))
        np.save(str(prefix)+"_meta_columns.npy",np.asarray(["a" for _ in range(10)],dtype=object))
        def dump(path, obj):path.write_text(json.dumps(obj),encoding="utf8")
        dump(Path(str(prefix)+"_capture_config.json"),{"parameter_set":p,"n":16})
        dump(Path(str(prefix)+"_keygen_sequence.json"),{"format":metadata.VERSION,"calls":n})
        spec={"parameter_set":p,"compiler_opt":opt,"seed_id":0,
              "sk_seed_sha256":hashlib.sha256(seed).hexdigest(),
              "pk_seed_hex":pub.hex(),"samples":12}
        dump(prefix.parent/"keygen_capture_manifest.json",{"format":metadata.VERSION,"status":"completed","capture_spec":spec})
        dump(prefix.parent.parent.parent/"shared_secret_seeds.json",{"parameter_set":p,"n":16,
              "seeds":[{"seed_id":0,"sk_seed_hex":seed.hex(),"sha256":hashlib.sha256(seed).hexdigest()}]})
        assert metadata._validate_completed(prefix,n,12,p)
        result=runner.main(["--base",str(base),"--output",str(Path(directory)/"analysis"),"--params",p,"--seed-ids","0","--dry-run"])
        assert result["completed_input_keygen_trials"]==1
        # One corrupted address must invalidate the existing capture.
        addresses=np.load(str(prefix)+"_addrs_random.npy",mmap_mode="r+")
        addresses[0,9]=6
        addresses.flush()
        assert not metadata._validate_completed(prefix,n,12,p)
