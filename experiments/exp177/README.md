# exp177 — Rhythm versus rate: van Rossum plasticity with and without homeostasis

exp176 found that van Rossum plasticity eroded COBA far faster than PING, but COBA
also fired ten times faster. Here both networks come from exp022 TR02 with the
same 25 Hz activity ceiling (seed 42), so their E rates match: PING
(`ping__rt25hz__seed42`) and COBA (`coba__rt25hz__seed42`).

Each network receives the same continuous stream of MNIST test digits outside
the 1,000-image accuracy subset (default 50 × 200 ms = 10 s, no state reset)
under four rules on the sensory weights X→E: frozen, STDP only
(`NonlinearSTDP`, mu_plus = 0, mu_minus = 1), synaptic scaling only
(`SynapticScaling`), and both (the full van Rossum model). The scaling setpoint
is the network's own frozen mean E rate.

Measured per 1-s block: Eoin's rhythmicity contrast (exp054), rhythm frequency
from the autocorrelogram side peak, I volleys per second, the fraction of E cells
firing per cycle, E/I rates, ISI CV and median, and spectra. Also recorded: every
spike, the readout's score for every stream digit, weight summaries and 1,000
tracked synapses after every digit, and test accuracy every 10 digits on 500
images.

## Run

Compute simulates and requires explicit authorisation. From the repository root:

```sh
uv run python experiments/exp177/compute.py
uv run python experiments/exp177/analyse.py --source COMPUTE_ID
uv run python experiments/exp177/present.py --source ANALYSE_ID
```

Each stage prints its run ID last. `compute.py` takes `--presentations N`,
`--accuracy-every N` and `--accuracy-images N`. Six plastic conditions dominate
the run time (roughly 25–35 min each on a laptop). Analyse and present can be
re-run on an existing compute run without simulating.

Non-simulation checks:

```sh
uv run pytest experiments/exp177/test.py -q
```

## Folder guide

1. `recipe.py` — settings, conditions, stream plan, rule mapping and networks.
2. `compute.py` — validated source loading and every condition's stream.
3. `analyse.py` — rhythm, cycles, rates, ISIs, spectra, weights and decoding.
4. `present.py` — circuit diagrams and figures.
5. `report/` — report, appendix and parameter table.
6. `test.py` — automated checks.
