# exp179 — Does the kind of PING rhythm change van Rossum plasticity?

exp022 TR03 trained six PING networks that differ only in the decay time of
inhibition: τ_GABA = 4.5, 6, 9, 12, 18 and 27 ms (seed 42, no rate penalty, 7,000
training images). Slower inhibition gives slower rhythms and lower E rates. Each
network streams the same MNIST test digits outside the 1,000-image accuracy subset
(default 50 × 200 ms = 10 s, no state reset) twice: frozen, and with the full van
Rossum rule (STDP + synaptic scaling, setpoint = its own frozen E rate) on the
sensory weights X→E.

Because STDP drift scales with pre × post rate, analysis also reports a pairing
proxy (input rate × mean pixel × E rate), so weight change can be compared per
pairing as well as per network.

Measured per 1-s block: Eoin's rhythmicity contrast (exp054), rhythm frequency,
I volleys per second, the fraction of E cells per cycle, rates, ISIs and spectra;
also every spike, online decoding, weight summaries, 1,000 tracked synapses, and
test accuracy every 10 digits on 500 images.

## Run

Compute simulates and requires explicit authorisation. From the repository root:

```sh
uv run python experiments/exp179/compute.py
uv run python experiments/exp179/analyse.py --source COMPUTE_ID
uv run python experiments/exp179/present.py --source ANALYSE_ID
```

Each stage prints its run ID last. `compute.py` takes `--presentations N`,
`--accuracy-every N` and `--accuracy-images N`. Six plastic conditions dominate
the run time (roughly 25–35 min each on a laptop). Analyse and present can be
re-run on an existing compute run without simulating.

Non-simulation checks:

```sh
uv run pytest experiments/exp179/test.py -q
```

## Folder guide

1. `recipe.py` — settings, conditions, stream plan, rule mapping and networks.
2. `compute.py` — validated source loading and every condition's stream.
3. `analyse.py` — rhythm, cycles, rates, ISIs, spectra, weights and decoding.
4. `present.py` — circuit diagrams and figures, including the τ_GABA summary.
5. `report/` — report, appendix and parameter table.
6. `test.py` — automated checks.
