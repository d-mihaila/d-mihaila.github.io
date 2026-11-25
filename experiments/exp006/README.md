# exp006 — One digit for ten seconds

exp001–exp005 changed the digit every 200 ms, so the input to each synapse kept
switching and the direction of each weight change switched with it. Here each
network sees a single MNIST digit for the whole stream (default 50 × 200 ms =
10 s, no state reset), with fresh Poisson spikes in every 200-ms window.

- **Networks:** exp022's rate-matched TR02 pair (`ping__rt25hz__seed42`,
  `coba__rt25hz__seed42`; 25 Hz activity ceiling).
- **Digits:** three different random digits (seed 181001), one official MNIST
  test image each, outside the 1,000-image accuracy subset. Each digit starts
  from the trained weights, so the three are independent repeats.
- **Rules:** frozen, and van Rossum STDP + synaptic scaling on the sensory
  weights X→E (setpoint = the frozen E rate on the same digit).

Measured per 1-s block: Eoin's rhythmicity contrast (exp054), rhythm frequency,
I volleys per second, E participation per cycle, rates, ISI distributions and
power spectra. For the weights, after every 200-ms window and split into the
digit's own pixels (intensity > 0.25) and the other pixels: signed and absolute
mean change, the shares of synapses potentiated or depressed by more than 1% of
the trained mean, and the mean change against the initial weight (W0 / w_max)
from full matrices at 2, 4, 6, 8 and 10 s. Also 1,000 tracked synapses.

## Run

Compute simulates and requires explicit authorisation. From the repository root:

```sh
uv run python experiments/exp006/compute.py
uv run python experiments/exp006/analyse.py --source COMPUTE_ID
uv run python experiments/exp006/present.py --source ANALYSE_ID
```

Each stage prints its run ID last. `compute.py` takes `--presentations N`
(200-ms windows per digit) and `--digits D [D ...]` (for example
`--digits 3 7 1`) instead of the random draw. Six plastic conditions dominate
the run time (roughly 15 min each). Analyse and present can be re-run on an
existing compute run without simulating.

Non-simulation checks:

```sh
uv run pytest experiments/exp006/test.py -q
```

## Folder guide

1. `recipe.py` — settings, digits, conditions, stimulus plan, rule mapping and networks.
2. `compute.py` — validated source loading and every condition's trajectory.
3. `analyse.py` — rhythm, cycles, ISIs, spectra and the direction of weight change.
4. `present.py` — circuit diagrams and figures.
5. `report/` — report, appendix and parameter table.
6. `test.py` — automated checks.
