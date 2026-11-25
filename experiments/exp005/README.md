# exp005 — Van Rossum's own tests inside a PING network

Van Rossum tested his rule with three probes: a biased input, a change of input
rate, and a long run to equilibrium. This experiment runs the same probes in
exp022's rate-matched TR02 pair (`ping__rt25hz__seed42`, `coba__rt25hz__seed42`;
25 Hz activity ceiling), with plastic sensory weights X→E:

1. **Biased stream** (PING; frozen, STDP only, STDP + scaling). 50 presentations
   of digit 3 (10 s), then 50 balanced presentations. Per-class test accuracy
   and the weights from the digit's pixels show competition and forgetting.
2. **Input-rate step** (PING; frozen, STDP only, STDP + scaling). 25
   presentations at the trained input rate, then 25 at half of it. Does the E
   rate follow the input (STDP alone) or return to the setpoint (with scaling)?
   The drive also sets the PING frequency.
3. **Long stream** (PING and COBA; frozen, STDP + scaling). 300 balanced
   presentations (60 s). Exponential fits give the forgetting time constant of
   the trained weights and the relaxation of the weight distribution.

Stream images are official MNIST test images outside the 1,000-image accuracy
subset; no state reset. The scaling setpoint is the frozen E rate over the
balanced half (biased), the pre-step half (rate step) or the whole stream
(long). Also measured per 1-s block: Eoin's rhythmicity contrast (exp054), rhythm
frequency, I volleys per second, the fraction of E cells per cycle, rates, ISIs
and spectra; every spike, online decoding, weight summaries, 1,000 tracked
synapses, and test accuracy every 10 digits on 500 images.

## Run

Compute simulates and requires explicit authorisation. From the repository root:

```sh
uv run python experiments/exp005/compute.py
uv run python experiments/exp005/analyse.py --source COMPUTE_ID
uv run python experiments/exp005/present.py --source ANALYSE_ID
```

Each stage prints its run ID last. `compute.py` takes `--biased-presentations N`,
`--rate-step-presentations N`, `--long-presentations N` (phases must span whole
5-presentation blocks), `--accuracy-every N` and `--accuracy-images N`. The two
60-s plastic streams dominate the run time (each about six times a 50-digit
stream). Analyse and present can be re-run on an existing compute run without
simulating.

Non-simulation checks:

```sh
uv run pytest experiments/exp005/test.py -q
```

## Folder guide

1. `recipe.py` — settings, design, stream plans, setpoint windows and networks.
2. `compute.py` — validated source loading and every condition's stream.
3. `analyse.py` — rhythm, cycles, rates, weights, competition and relaxation fits.
4. `present.py` — circuit diagrams and figures, including the three probe figures.
5. `report/` — report, appendix and parameter table.
6. `test.py` — automated checks.
