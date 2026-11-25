# exp007 — One digit for ten seconds: STDP alone against STDP + scaling

exp006 showed one MNIST digit for 10 s to trained PING and COBA networks, with
van Rossum STDP and synaptic scaling together. This experiment separates the two
halves of the rule. Every digit runs three times, each from the trained weights:

1. **Frozen**: the control, and the source of the scaling setpoint.
2. **STDP only**: weight-dependent STDP on the sensory weights X→E.
3. **STDP + scaling**: as exp006.

Same networks (exp022's rate-matched TR02 pair, `ping__rt25hz__seed42` and
`coba__rt25hz__seed42`), digits (2, 6, 0), images, Poisson spike trains and
tracked synapses as exp006, so the STDP + scaling runs repeat exp006's. Each
digit is shown for 50 × 200 ms = 10 s, no state reset.

## What is measured

- **Rhythm and firing**, per 1-s block: Eoin's rhythmicity contrast (exp054),
  rhythm frequency, I volleys per second, E participation per cycle, E rate,
  ISI CV and median, spectral peak; E rate in every 200-ms window against the
  setpoint.
- **ISIs and spectra**: the frozen first second against every rule's last
  second, and each plastic rule's ISI distribution minus the frozen one.
- **Direction of weight change**, for synapses from the digit's own pixels and
  from the others: signed and absolute change, shares potentiated or depressed,
  ΔW against W0 / w_max at 2–10 s, final-against-initial maps, distributions.
- **What scaling adds**, from the paired STDP-only and STDP + scaling runs:
  ΔW with scaling against ΔW without for every synapse; each E cell's total
  input with scaling over without, against its STDP-only rate over the setpoint
  (Spearman correlation); per-cell rate distributions.
- **Overview**: every measure per rule, one dot per digit, so you can see at a
  glance whether the three digits agree.

Synapses from pixels that stay dark are a direct probe of scaling: their inputs
hardly fire, so STDP cannot move them, while scaling multiplies every input of
a cell.

## Run

Compute simulates and requires explicit authorisation. From the repository root:

```sh
uv run python experiments/exp007/compute.py
uv run python experiments/exp007/analyse.py --source COMPUTE_ID
uv run python experiments/exp007/present.py --source ANALYSE_ID
```

Each stage prints its run ID last. `compute.py` takes `--presentations N`
(200-ms windows per digit) and `--digits D [D ...]`. Twelve plastic conditions
dominate the run time (roughly 15 min each, so about 3 hours). Analyse and
present can be re-run on an existing compute run without simulating.

Non-simulation checks (including every figure drawn from synthetic results):

```sh
uv run pytest experiments/exp007/test.py -q
```

## Folder guide

1. `recipe.py` — settings, digits, conditions, stimulus plan, rule mapping and networks.
2. `compute.py` — validated source loading and every condition's trajectory.
3. `analyse.py` — rhythm, ISIs, spectra, weight change and the paired STDP/scaling comparison.
4. `present.py` — circuit diagrams and the 19 figures.
5. `report/` — report (with hidden write-up templates), appendix and parameter table.
6. `test.py` — automated checks.
