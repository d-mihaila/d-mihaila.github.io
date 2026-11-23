# exp175 — Reconstruct the attended MNIST image

One compiled local MPS epoch has completed successfully after disabling the seven
frozen zero projections. The current physical/effective batch is 64. Historical
out-of-memory attempts are retained below; no benchmark process remains running.

1. Initialise the shared pixel→A/B encoder and frozen local E/I matrices from
   unpenalised TR02 PING: `exp022-r039-compute`, `ping__off__seed42`.
   Keep the native neuron ordering. The fixed A/B→C relay is defined in the recipe.
2. Train the encoder and a fresh nonnegative C→784-neuron spiking readout.
   As in exp082, the output population and its input synapse are leaky integrators
   with 2-ms decay, soft-reset threshold 1, and output spike-count decoding.
   Initialise the output matrix from a clamped normal with mean 0.05 and standard
   deviation 0.04. TR02's 10-class output weights are not reused.
   All attention connections remain frozen.
3. Decode each pixel as output spike count divided by the white-pixel reference.
   The initial reference is 500 Hz: 20 spikes in either 40-ms scoring window.
   Zero spikes represents black, 10 represents half intensity, and 20 represents
   white. These calibration settings are proposed for this experiment, not taken
   from exp082. Train with pixel MSE against the original attended 28×28 image.
   Counts above the white reference remain unclipped in the loss, so overshoot is
   penalised; image rendering uses a fixed 0–1 display scale.
   No class loss or rate penalty is applied. The clean image enters only the loss,
   never a neural input. SNNLab's dataset API supplies scalar image indices; the
   small custom objective retrieves the matching grayscale targets from the local
   official MNIST training IDX file. This cache must be available when training or
   resuming; the path is recorded in the resolved configuration.
4. Use stratified subsets of TR02's separate seed-42 partitions: 1,000 unique
   training images and 100 unique validation images. Pair targets with different-digit distractors from the same
   split. Vary partner, target location and supervised cue phase during training.
5. Keep both images present for 200 ms, switching the fixed 12× contrast cue at
   100 ms. Score 50–90 ms or 150–190 ms. Validation covers both target
   locations and both cue phases with three fixed noise draws. Select minimum
   validation MSE, keeping the earliest epoch on ties, including epoch zero.
6. Defaults: 10 epochs on the local Apple GPU (`mps`), AdamW learning rate 0.0004, zero weight decay, effective
   batch size 64, physical microbatch size 64, and native compilation enabled. Save native checkpoints and snapshots.
   After future training, retain reconstructions for the first two validation
   targets in all four cue conditions, without choosing examples by performance.
   Presentation shows image A, image B, the attended target and C's reconstruction.

The reused validation split is not an independent test set. This circuit does not
reproduce the previously validated scratch body. Whether reconstruction follows
attention, preserves fine image detail, or averages the inputs remains untested.
The shorter scoring windows may include settling transients; this is a pilot timing protocol.
The new readout adds 784 spiking output neurons and a projection evaluated every
neural timestep; it is not claimed to reduce training memory or runtime. No training epoch has completed; the GPU benchmark attempted validation and training forward execution.

## Commands

Inspect the scaffold and run non-simulating checks:

```sh
uv run python experiments/exp175/compute.py plan
uv run pytest experiments/exp175/test.py -q
```

Only after explicit execution authorisation:

```sh
uv run python experiments/exp175/compute.py train --device mps --train-images 1000 --validation-images 100 --epochs 10 --duration-ms 200
uv run python experiments/exp175/analyse.py --source COMPUTE_ID
uv run python experiments/exp175/present.py --source ANALYSE_ID
```

Training supports `--train-images`, `--validation-images`, `--epochs`, `--microbatch-size`, and
`--compile-mode none|default`. Nothing is published automatically.

## Batch-256 local GPU timing trial

The one-epoch, 1,000/100-image, 200-ms MPS trial failed after **66.62 seconds**
during the first training batch's forward pass. MPS reported 18.02 GiB allocated,
1.96 GiB in other allocations, and a configured limit of 19.97 GiB. The trial used
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` and low watermark `0.7`; this is evidence
of exceeding that cap, not a measurement of the machine's absolute capacity.
No epoch completed, so no epoch-time estimate is available. Physical and effective
batch sizes remain 256. No smaller-batch retry was launched.

Failure evidence: `.pingstore/runs/.exp175-r20261118103533219482192650-compute.tmp/export/failure.json`.
The directory is an incomplete failed run, not a completed scientific result.
The earlier extrapolated 20–30 hour estimate remains withdrawn.

The retry at the full **24.96 GiB** Metal recommended cap failed after **64.57
seconds** during training forward execution: 21.51 GiB allocated plus 3.45 GiB
in other allocations. Evidence: `.pingstore/runs/.exp175-r20261118103844074379738662-compute.tmp/export/failure.json`.
No epoch time was measured. The recipe now explicitly sets `mps_memory_fraction=1.0`,
so future MPS runs use the full recommended working-set cap.

## Streamed scoring-window spike counts

The graph supplies a native `(time, batch)` scoring mask directly and sums the
784 output neurons' spikes in that window. `custom/streaming.py` scopes the
SNNLab 0.32.3 dataset-mask and differentiable-reduction compatibility adaptations
to the experiment's execution calls. Values and gradients match the dense masked
spike sum. Execution planning does not retain a dense output spike history for
this readout; recurrent autograd states remain necessary.

The earlier CPU/MPS preflights and memory failures above used the previous
pooled-linear readout. The spiking-readout GPU benchmark below has since been attempted.

## Current spike-count GPU benchmark

The one-epoch benchmark with the streamed spiking readout, 1,000/100 images,
200-ms trials and physical/effective batch 256 failed after **67.04 seconds**
during the training forward pass. MPS reported 23.00 GiB allocated plus 1.96 GiB
in other allocations, reaching the 24.96 GiB recommended working-set cap. No
epoch completed; no epoch-time estimate or completed reconstruction result is
available. No smaller-batch retry was launched. Failure evidence is retained in
`.pingstore/runs/.exp175-r20261118111136167803611568-compute.tmp/export/failure.json`.

## Batch-128 retry

Halving both physical and effective batch sizes to 128 also exhausted the
24.96 GiB MPS cap during the training forward pass, after **70.26 seconds**.
MPS reported 23.24 GiB allocated plus 1.73 GiB in other allocations. No epoch
completed. Current batch defaults remain 128; no further retry was launched.
Evidence: `.pingstore/runs/.exp175-r20261118111403266978165490-compute.tmp/export/failure.json`.

## Batch-64 retry

Physical and effective batch sizes were halved again to 64. The one-epoch local
MPS trial failed after **89.32 seconds** during the training forward pass, with
24.09 GiB allocated and 886.72 MiB in other allocations against the 24.96 GiB
cap. No epoch completed. Current batch defaults remain 64; no further retry
was launched. Evidence: `.pingstore/runs/.exp175-r20261118111823840381481151-compute.tmp/export/failure.json`.

Compilation is now enabled by default (`compile_mode="default"`); the CLI
uses the same recipe default. Earlier timing and memory failures were eager runs.
No compiled training or GPU benchmark has been launched yet.

The six frozen zero E→E/I→I recurrence projections and the zero background→C
projection are now explicitly disabled in the native execution plan. Their
zero parameters remain in snapshots. Initialisation rejects nonzero values for
these disabled paths. Nonzero E→I/I→E feedback, sensory encoders, relays and
the readout remain enabled. No post-change simulation has been launched.

## Completed compiled batch-64 benchmark

Run: `exp175-r20261118122900578123908026-compute`. One epoch on 1,000 training
images and 100 validation images, 200-ms trials, physical/effective batch 64,
MPS compilation enabled, and seven frozen zero projections disabled completed
with 16 optimiser updates. Training: **82.26 s**; post-epoch validation:
**45.37 s**; epoch training plus validation: **127.64 s**. Initial validation
took **46.23 s**. End-to-end process wall time, including setup, compilation,
checkpoint/snapshot writing and illustrative reconstructions: **201.95 s**.
Validation pixel MSE improved from **0.098654** to **0.086386**; epoch 1 was
selected. Both trainable matrices changed and every frozen matrix stayed
bit-identical. Native MPS peak-memory counters were unavailable; the run
completed under the 24.96 GiB recommended memory cap. Training timing and
resolved compiler policy are retained in the completed run's `metrics.json`.

## Single-PING pretraining and relay transfer

`train --architecture single` trains the same encoder and spiking pixel decoder
around one fixed TR02 PING population. The input is the target sender's exact
contrast-modulated spike stream, including its high/low contrast timing. Its
partner is used only for the existing rate normalisation; no distractor drives
the single PING. Image split, loss, scoring windows and optimiser are unchanged.

After selection, the compute stage copies only the encoder and decoder into the
relay and evaluates without relay fine-tuning. `transfer-metrics.json` and
`transfer-reconstructions.npz` compare both architectures on all 400 validation
presentations using one paired simulate-phase encoding draw. This comparison is
separate from the three-draw validation used for checkpoint selection. Native
bundles, diagrams, inference-start snapshots and a transfer checkpoint are saved.
The transfer checkpoint contains no optimiser state or relay training updates.

## Compact Modal input

The cancelled Modal attempt never submitted a training call. Its app and local
uploader were stopped. Future runs use `exp175-r001-import`, a 5.06-MiB compact
input containing only `W_ff.0`, `W_ei.1`, `W_ie.1` and the required configuration.
The unused classifier and zero E→E/I→I matrices are omitted; zero matrices are
recreated locally. All 19 initialised parameters exactly match the original
source path. The input's provenance retains `exp022-r039-compute` as a dependency.

To recreate the compact input locally if needed (no simulation):

```sh
uv run python experiments/exp175/compute.py prepare-input
```

Use `--direct-inputs-only` so the dispatcher transfers only the compact input,
without its original TR02 ancestor. This command has **not** been relaunched:

```sh
PINGLAB_MODAL_CUDA_DEVEL=1 uv run --no-project --python 3.12 --with 'modal>=1.1,<2' \
  python -m libraries.modal_dispatcher dispatch --experiment exp175 \
  --gpu A100-80GB --cpu 4 --memory 32768 --timeout 3600 \
  --direct-inputs-only --input tr02=exp175-r001-import \
  -- train --device cuda --epochs 10 --train-images 1000 --validation-images 100 \
  --duration-ms 200 --microbatch-size 64
```
