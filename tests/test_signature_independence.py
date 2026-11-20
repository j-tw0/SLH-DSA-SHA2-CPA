"""No hardware. Test input preservation, per-signature units, truth manifests and means."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import signature_cpa_batch as runner


def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2), encoding="utf-8")


def fixture():
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    input_root = root / "captures"
    param_root = input_root / "SLH_DSA_128f_sha2_message_signature_sequence"
    opt_dir = param_root / "Os"
    secrets = {0:bytes(range(16)), 1:bytes(range(16,32))}
    def pool(n):
        return dict(format="slh-dsa-shared-secret-seeds-v1", parameter_set="128f", n=16,
                    num_seeds=n, master_seed=20260807,
                    seeds=[dict(seed_id=i, sk_seed_hex=secrets[i].hex(),
                                sha256=hashlib.sha256(secrets[i]).hexdigest()) for i in range(n)])
    write_json(param_root/"shared_secret_seeds.json", pool(1))
    write_json(param_root/"shared_secret_seeds_extra_to_001.json", pool(2))
    cols = ["signature_id", "algorithm_call_index", "phase_code", "kind",
            "layer", "tree", "keypair", "chain", "tree_height", "tree_index",
            "tree_number", "selected_index"]
    for seed_id, completed_sigs in [(0,[0,1]), (1,[0])]:
        root_byte = (bytes([seed_id+11])*16).hex()
        seed_dir = opt_dir/f"seed_{seed_id:03d}"
        manifest = {
            "format":"slh-dsa-fips205-random-message-signing-prf-capture-v3",
            "parameter_set":"128f", "mode":"sha2", "num_signatures":2,
            "pk_root_hex":root_byte,
            "capture_phases":["fors_signature", "fors_treehash", "wots_signature", "xmss_treehash"],
            "completed":[dict(signature_id=s, status="completed-new-request") for s in completed_sigs],
        }
        write_json(seed_dir/"signature_sequence_capture_manifest.json", manifest)
        for sig in completed_sigs:
            pref = seed_dir/f"signature_{sig:03d}"/"capture"
            pref.parent.mkdir(parents=True)
            # 4 FORS and 4 WOTS contiguous rows, unique values per experiment.
            np.save(str(pref)+"_traces_random.npy", np.random.default_rng(seed_id*10+sig).normal(size=(8, 24)).astype(np.float32))
            addr = np.zeros((8,32), dtype=np.uint8)
            addr[:4,9] = 6
            addr[4:,9] = 5
            addr[:,20:22] = np.arange(8,dtype=np.uint16)[:,None] // np.array([256,1],dtype=np.uint16)
            np.save(str(pref)+"_addrs_random.npy",addr)
            np.save(str(pref)+"_params_random.npy",np.zeros((8,10),dtype=np.uint64))
            np.save(str(pref)+"_meta_columns.npy",np.array(["kind"],dtype=object))
            meta = np.zeros((8,len(cols)),dtype=np.uint64)
            meta[:,cols.index('signature_id')]=sig
            meta[:,cols.index('algorithm_call_index')]=np.arange(8)
            meta[:,cols.index('phase_code')]=[1,2,2,2,3,4,4,4]
            meta[:,cols.index('kind')]=[6]*4+[5]*4
            np.save(str(pref)+"_signature_sequence_meta.npy",meta)
            np.save(str(pref)+"_signature_sequence_meta_columns.npy",np.array(cols,dtype=object))
            write_json(Path(str(pref)+"_signature_sequence.json"), {
                "format":"slh-dsa-fips205-random-message-signing-prf-address-sequence-v3",
                "signature_id":sig,"parameter_set":"128f","pk_root_hex":root_byte,
                "capture_phases":manifest['capture_phases'], "captured_total_calls":8,
            })
            write_json(Path(str(pref)+"_capture_config.json"),{
                "parameter_set":"128f","n":16,"build":{
                    "secret_seed_id":seed_id,"hash_mode":"sha2",
                    "sk_seed_sha256":hashlib.sha256(secrets[seed_id]).hexdigest()}})
    return temp, input_root, secrets


def input_hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_independent_signature_and_macro_rates():
    temp, root, secrets = fixture()
    with temp:
        baseline = input_hashes(root)
        output = root.parent/"analysis"
        fake_calls = []
        def fake(cap, *, truth, **kwargs):
            path = str(cap.source).replace('\\','/')
            seedid = int(re.search(r'seed_(\d+)', path).group(1))
            sigid = int(re.search(r'signature_(\d+)',path).group(1))
            fake_calls.append((seedid, sigid, cap.traces.shape[0], cap.addrs.shape[0]))
            success = (seedid == 1 or sigid == 0)
            ret = SimpleNamespace(recovered_seed=(truth if success else bytes([0]*len(truth))))
            return ret, [
                dict(stage_key='W5',role='primary', selected_match=True,
                     target_word='W5',intermediate_word='A_update'),
                dict(stage_key='W6',role='primary', selected_match=success,
                     target_word='W6',intermediate_word='A_update'),
            ]
        args = ['--base',str(root),'--output',str(output), '--params','128f', '--opts','Os',
                '--source','message', '--seed-ids','auto','--signature-ids','all',
                '--kinds','fors','--attacks','compression','--protocol','paper',
                '--per-signature','auto','--no-summary-plots','--backend','numpy']
        with patch.object(runner,'run_attack',side_effect=fake):
            manifest = runner.main(args)
        assert manifest['completed_input_signature_trials']==3
        assert manifest['completed_cpa_runs']==3
        assert len(fake_calls)==3 and all(x[2:] == (4,4) for x in fake_calls)
        assert input_hashes(root)==baseline, 'Capture file was modified by analysis'
        with (output/'summary_by_condition.csv').open(encoding='utf-8-sig',newline='') as f:
            cond = list(csv.DictReader(f))
        assert len(cond)==1
        row=cond[0]
        assert int(row['evaluated_seed_count']) == 2
        assert int(row['evaluated_signature_trials'])==3
        assert abs(float(row['seed_macro_full_seed_success_rate'])-.75)<1e-9
        assert abs(float(row['trial_weighted_full_seed_success_rate'])-2/3)<1e-9
        with (output/'summary_by_signature.csv').open(encoding='utf-8-sig',newline='') as f:
            signatures=list(csv.DictReader(f))
        assert len(signatures)==2
        assert [int(x['signature_id']) for x in signatures]==[0,1]
        assert float(signatures[0]['full_seed_success_rate_across_seeds'])==1.0
        assert float(signatures[1]['full_seed_success_rate_across_seeds'])==0.0
        with (output/'stage_success_rates.csv').open(encoding='utf-8-sig',newline='') as f:
            stages=list(csv.DictReader(f))
        stage6=next(x for x in stages if x['stage_key']=='W6')
        assert abs(float(stage6['seed_macro_stage_success_rate'])-.75)<1e-9
        with (output/'missing_or_skipped.csv').open(encoding='utf-8-sig', newline='') as f:
            skips=list(csv.DictReader(f))
        assert any(r['status']=='incomplete_signature' and r['seed_id']=='1' for r in skips)
        fake_calls.clear()
        with patch.object(runner,'run_attack',side_effect=fake):
            runner.main(args)
        assert not fake_calls, 'Existing CPA results were not resumed'
        assert input_hashes(root)==baseline, 'Resume modified input files'
        print('PASS: single-signature independence, seed-macro .75, weighted 2/3, stage, resume, read-only')


def test_dry_run_and_manifest_conflict():
    temp, root, _ = fixture()
    with temp:
        baseline = input_hashes(root)
        output = root.parent / 'check'
        with patch.object(runner, 'run_attack', side_effect=AssertionError('dry-run ran CPA')):
            result = runner.main(['--base',str(root),'--output',str(output), '--params','128f',
                                  '--kinds','fors,wots','--dry-run','--no-summary-plots'])
        assert result['completed_cpa_runs']==0
        assert input_hashes(root)==baseline
        # Contradicting original and extra manifests must never be silently accepted.
        manifest = root/'SLH_DSA_128f_sha2_message_signature_sequence'/'shared_secret_seeds_extra_to_001.json'
        altered=json.loads(manifest.read_text())
        altered['seeds'][0]['sk_seed_hex']=(bytes([5])*16).hex()
        altered['seeds'][0]['sha256']=hashlib.sha256(bytes([5])*16).hexdigest()
        write_json(manifest,altered)
        try:
            runner.scan_trials(root,['128f'],['Os'],'auto','all','message')
        except ValueError as exc:
            assert 'Conflicting secret seed' in str(exc)
        else:
            raise AssertionError('Conflicting seed manifests accepted')
        print('PASS: dry-run and conflicting seed manifests rejected')


def test_sampled_subset_is_separate_cache():
    temp, root, _ = fixture()
    with temp:
        trials, truth, issues, inventory = runner.scan_trials(root,['128f'],['Os'],'0','0','message')
        assert len(trials)==1
        output=root.parent/'new_output'
        cap, selection = runner.select_capture(trials[0], 'fors', None, 3, 'uniform', 11, output, False)
        assert cap.traces.shape == (3,24) and not selection['zero_copy_contiguous_slice']
        assert (output/'_selected_trace_cache').is_dir()
        assert not (root/'_selected_trace_cache').exists()
        print('PASS: downsampled non-contiguous rows only cached under analysis root')


if __name__=='__main__':
    test_independent_signature_and_macro_rates()
    test_dry_run_and_manifest_conflict()
    test_sampled_subset_is_separate_cache()
    print('ALL ONE-SIGNATURE MULTI-SEED TESTS PASSED')
