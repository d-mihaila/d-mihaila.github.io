# exp168 — AI and PING with and without EE/II recurrence

Compare asynchronous and rhythmic activity with and without excitatory and inhibitory
self-recurrence. The conditions use selected single-seed network and input settings.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp168/compute.py simulate
uv run python experiments/exp168/analyse.py
uv run python experiments/exp168/present.py --source ANALYSE_ID
```

Replace `ANALYSE_ID` with the completed analysis run ID.

Analysis reads the byte-preserved Figure 1/2 evidence in the consolidated
`exp168-r038-compute` run, with original configurations resolved by `recipe.py`.
A new comparison simulation is not automatically analysed.

## Fixed-wiring transition

The third results section reuses Figure 1B's actual sampled dense-input EE/II-on
network. External→E gain stays at 1; External→I gain falls from 1 to 0 between
1,050 and 1,550 ms, followed by a hold through 3,550 ms. Wiring, source rate and
all sampled matrices stay fixed. The baseline is AI-like; low firing limits
single-cell CV estimates. All retained evidence is independent of scratch files.

To import an already completed trajectory, without simulating:

```sh
uv run python experiments/exp168/compute.py import-transition COMPLETED_TRAJECTORY_DIRECTORY
uv run python experiments/exp168/analyse.py --transition-source COMPUTE_ID
uv run python experiments/exp168/present.py --source ANALYSE_ID --base-present exp168-r040-present
```

The import directory must contain the native bundle, start snapshot, events and
summary, with the original fixed checkpoint in its parent directory. Import
validates the graph and every sampled matrix against the scientific source.
The optional `--base-present` preserves the earlier figures byte-for-byte after
checking that their comparison configurations and metrics match the analysis.

An explicitly authorised replay uses retained Pingstore evidence:

```sh
uv run python experiments/exp168/compute.py transition --source COMPUTE_ID
```

Pass its new compute ID to analysis as above. Figures and results are selected
through the new presentation run; report prose contains no selected run ID.

To run automated checks:

```sh
uv run pytest experiments/exp168/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `dataset.py` — input preparation and encoding.
3. `compute.py` — run simulations and save evidence.
4. `analyse.py` — calculate measurements from saved evidence.
5. `present.py` — generate figures and report assets.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
