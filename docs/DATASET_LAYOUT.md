# Expected dataset layout (local files, not uploaded to GitHub)

Use a dataset created by the separately published
[SLH-DSA PRF capture project](https://github.com/j-tw0/SLH-DSA-PRF-Capture).
All file paths below are **examples**.

## Random-address capture

```text
data/random_address/
  SLH_DSA_128f_sha2_compiler_compare/
    shared_secret_seeds.json            # experimental ground truth; DO NOT COMMIT
    Os/
      multi_seed_wots_fixed_random/
        seed_000/
          capture_traces_random.npy     # filenames may have a longer prefix
          capture_addrs_random.npy
          capture_params_random.npy
          capture_meta_columns.npy
          capture_capture_config.json
      multi_seed_fors_fixed_random/
        seed_000/ ...
```

The discovery routine can locate longer capture prefixes and alternate folder
positions by scanning `*_capture_config.json`; the prefix must contain matching
trace and address arrays.

## One-signature path capture

```text
data/signature_sequences/
  SLH_DSA_128f_sha2_message_signature_sequence/
    shared_secret_seeds.json            # DO NOT COMMIT
    Os/
      seed_000/
        signature_sequence_capture_manifest.json
        signature_000/
          capture_traces_random.npy
          capture_addrs_random.npy
          capture_params_random.npy
          capture_meta_columns.npy
          capture_signature_sequence_meta.npy
          capture_signature_sequence_meta_columns.npy
          capture_signature_sequence.json
          capture_capture_config.json
        signature_001/ ...
```

Each signature is analyzed separately. Use `--source message` for the
PRF_msg/H_msg-derived signing-path implementation.

## Keygen WOTS_PRF capture

```text
data/keygen/
  SLH_DSA_128f_sha2_keygen_prf_sequence/
    shared_secret_seeds.json            # DO NOT COMMIT
    Os/
      seed_000/
        keygen_capture_manifest.json
        capture_traces_random.npy
        capture_addrs_random.npy
        capture_params_random.npy
        capture_meta_columns.npy
        capture_capture_config.json
        capture_keygen_sequence.json
```

CPA reads these files and validates the keygen address sequence. The complete
key-generation procedure itself is **not** captured as a continuous waveform.

## Manifest and seed safety

The real secret seeds are stored in `shared_secret_seeds.json` and used only as
**ground truth** for evaluation. The CPA tools do not need these files to be
published. The runner output may also contain recovered secret values;
keep its output directory outside the public repository or in an ignored
`results/` directory.
