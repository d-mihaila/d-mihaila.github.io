# exp001 — Van Rossum plasticity in trained PING and COBA networks

Start from the trained exp022 TR01 networks (canonical cells: no rate penalty,
all MNIST, seed 42, final epoch) and ask how van Rossum plasticity on the sensory
weights changes their behaviour. PING has a trained E→I→E gamma loop; COBA has the
same E population and readout but zero E–I weights, so it is authored without I.

Each network receives one continuous stream of MNIST validation digits
(default 50 × 200 ms = 10 s, no state reset) twice, with identical hash-checked
input: once frozen, once with van Rossum STDP (`NonlinearSTDP`, mu_plus = 0,
mu_minus = 1) plus synaptic scaling (`SynapticScaling`) on `input_to_E` only.
Compute records every E/I spike, weight summaries and 1,000 random tracked
synapses after every presentation, full weight matrices at five boundaries, and
held-out test accuracy every 5 presentations. The report shows rasters, weight
evolution, synapse trajectories, E/I rates, the scaling sensor, E ISI
distributions, CV and median per 1-s block, Eoin's rhythmicity contrast (exp054)
and rhythm frequency per block, population spectra, receptive fields and the
accuracy curve. Colour shows the condition (black frozen, blue van Rossum). The rule constants and their mapping onto
SNNLab are documented in `recipe.py` and the report appendix.

## Run

Compute simulates and requires explicit authorisation. From the repository root:

```sh
uv run python experiments/exp001/compute.py
uv run python experiments/exp001/analyse.py --source COMPUTE_ID
uv run python experiments/exp001/present.py --source ANALYSE_ID
```

Each stage prints its run ID last. `compute.py` takes `--presentations N`
(stream length, default 50), `--accuracy-every N` (default 5) and
`--accuracy-images N` (default 1000; 0 skips). Analyse and present can be
re-run on an existing compute run to add metrics or plots without simulating.
Plastic simulation is eager and per-step, so it dominates the run time; it uses
all CPU cores by default.

The source is the restored consolidated bank `exp022-r039-compute`; compute
checks each checkpoint's SHA-256 against the bank's provenance before loading.

Non-simulation checks:

```sh
uv run pytest experiments/exp001/test.py -q
```

## Folder guide

1. `recipe.py` — settings, the van Rossum rule mapping and the native networks.
2. `compute.py` — validated source loading, the frozen and plastic streams, accuracy.
3. `analyse.py` — rates, ISIs, spectra and weight summaries via `snnlab.analysis`.
4. `present.py` — circuit diagrams and figures.
5. `report/` — report, appendix and parameter table.
6. `test.py` — automated checks.
