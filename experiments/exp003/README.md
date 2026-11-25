# exp003 — Plasticity parameters that steer the PING rhythm

Can a plasticity parameter make the PING rhythm sharper, sparser, faster or
slower? The source is the trained exp022 TR01 PING network
(`ping__canonical__seed42`, no rate penalty, all MNIST), whose rhythmicity
contrast is moderate, so it can move either way. The full van Rossum rule (STDP +
synaptic scaling) acts on the sensory weights X→E, with one parameter changed at
a time:

1. **STDP window** τ = 2.5, 5, 10, 20 and 40 ms (setpoint ×1). The rhythm period
   is about 10 ms, so short windows pair spikes within a cycle and long ones
   across cycles. Amplitudes scale with 20 ms / τ, so every window has the same
   area and only timing resolution changes.
2. **Scaling setpoint** ν* = 0.5× and 2× the frozen E rate (τ = 20 ms).

Plus a frozen control. Every condition streams the same MNIST test digits outside
the 1,000-image accuracy subset (default 50 × 200 ms = 10 s, no state reset).
Measured per 1-s block: Eoin's rhythmicity contrast (exp054), rhythm frequency,
I volleys per second, the fraction of E cells per cycle, rates, ISIs and spectra;
also every spike, online decoding, weight summaries, 1,000 tracked synapses, and
test accuracy every 10 digits on 500 images.

## Run

Compute simulates and requires explicit authorisation. From the repository root:

```sh
uv run python experiments/exp003/compute.py
uv run python experiments/exp003/analyse.py --source COMPUTE_ID
uv run python experiments/exp003/present.py --source ANALYSE_ID
```

Each stage prints its run ID last. `compute.py` takes `--presentations N`,
`--accuracy-every N` and `--accuracy-images N`. Seven plastic conditions dominate
the run time (roughly 25–35 min each on a laptop). Analyse and present can be
re-run on an existing compute run without simulating.

Non-simulation checks:

```sh
uv run pytest experiments/exp003/test.py -q
```

## Folder guide

1. `recipe.py` — settings, sweep conditions, stream plan, rule mapping and network.
2. `compute.py` — validated source loading and every condition's stream.
3. `analyse.py` — rhythm, cycles, rates, ISIs, spectra, weights and decoding.
4. `present.py` — circuit diagram and figures, including the sweep summary.
5. `report/` — report, appendix and parameter table.
6. `test.py` — automated checks.
