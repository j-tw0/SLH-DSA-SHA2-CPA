# Reproducibility boundaries

## Hardware and acquisition assumptions

Paper: CW308 UFO / STM32F303, ChipWhisperer Husky, target clock 7.37 MHz,
ADC sampling 29.48 MS/s, gain 15 dB, GCC `-Os`, SHA-256 compression and
message schedule inside the trigger interval. The capture repository provides
the target firmware and acquisition scripts. Those hardware settings must be
verified on a real board; the CPA repository cannot verify them.

## Distinguish evaluation protocols

- Random-address paper baseline: 30 independent SK.seed values with 5,000
  waveforms each per parameter set and address type. **Correct preceding
  words are provided at each separate word-level stage.** Do not present
  conditional assembled-key matches as real full-seed recovery.
- Signing path: 3 SK.seed values × 3 independent signing paths each, for each
  parameter set; a single trial uses one signature and one PRF address type.
  Chained analysis propagates recovered values, not true previous words.
- Key-generation path: 3 SK.seed values, one top-level XMSS WOTS_PRF sequence
  per key. Chained **compression** CPA only; the paper does not evaluate
  keygen message-schedule recovery.

## Reproducibility is conditional on matching data

All software input paths are configurable. Demonstration PK/SK values in the
public core are deliberately different from private research test vectors.
The original same-address and same-waveform data are not part of the release.
Different random messages, seed values and samples may yield different CPA
success rates despite identical attack equations.

The random-address wrapper is a thin adapter derived from the paper baseline;
it is not itself one of the original standalone research notebooks. Selected
attack modules keep their functions and candidate/scoring logic. The keygen
metadata reader preserves the recorded address validation contract but does
not require the acquisition firmware or ChipWhisperer to import.

For computationally expensive `s` cases start by validating a small slice of
captured data and use a sample window/NumPy backend when necessary. The
paper's full-sample analysis may require substantial compute resources.

## Publication and license safety

Do not publish actual secret-seed manifests, any credentials, device serial
identifiers, personal absolute paths, real traces, original recovered keys or
unreviewed measurement logs. Add a `LICENSE` only after confirming code
ownership and approval from the contributing researchers.
