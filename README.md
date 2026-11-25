# Daria Mihaila — Pinglab fork

A personal fork of [Pinglab](https://github.com/eoinmurray/pinglab) by Eoin Murray:
conductance-based spiking neural networks with excitatory/inhibitory populations
and PING (pyramidal–interneuron gamma) dynamics. This fork hosts Daria Mihaila's
experiments.

Site: [d-mihaila.github.io](https://d-mihaila.github.io).
Upstream site: [pinglab.eoinmurray.info](https://pinglab.eoinmurray.info).

Main documentation lives in [`notes/`](notes/):

1. [Experiment Rules](notes/rules/experiment.md): experiment structure, stages and scientific inputs.
2. [Compute Rules](notes/rules/compute.mdx): compute providers, access and execution workflows.
3. [Pingstore Rules](notes/rules/pingstore.md): run storage, provenance, explicit dependencies, backups and deletion.
4. [Writing Rules](notes/rules/writing.md): scientific writing, article metadata and editorial conventions.
5. [Publishing Rules](notes/rules/publishing.md): local site builds, independent static deployment.

Upstream notes (canonical parameters, COBA/PING dynamics, the mean-field model,
the research workflow and the manuscript) stay on the
[upstream site](https://pinglab.eoinmurray.info/notes).

See [Astro setup and GitHub Pages deployment](site/README.md)
for site development and deployment. Coding agents must read
[`AGENTS.md`](AGENTS.md) for task-specific instruction routing.

Use the [Modal dispatcher](libraries/modal_dispatcher/README.md) to launch experiment
compute on Modal and collect validated Pingstore results.

Pingstore is a directory of runs with small JSON metadata, a Python `Store` API,
and a CLI: `insert`, `get`, `list`, `delete`, `backup` and `restore`.
Import from `pingstore.library`; run `uv run pingstore --help`.
Backups are incremental and do not download uploaded payloads for validation.
See [operations](notes/rules/pingstore-operations.md) for usage and migration.

For generic CSD3 / Wilkes3 experiment deployment, see [HPC deployment](libraries/hpc/README.md).

Python formatting uses [Ruff](https://docs.astral.sh/ruff/formatter/) with an
88-character line length. Function signatures that fit stay on one line; longer
signatures put each parameter on its own indented line, with the closing `):`
on a separate line. A trailing comma keeps an expanded signature multiline.
The formatter also applies its standard layout to calls and collections.

```sh
uv run ruff format         # Apply formatting.
uv run ruff format --check # Check formatting without editing files.
uv run ruff check          # Check lint rules separately.
```

CI checks both lint and formatting.

Install the local Git hook once per checkout after installing development
dependencies:

```sh
uv sync --dev
uv run pre-commit install
```

## Working with upstream

`origin` is this fork; `upstream` is Eoin Murray's repository, with pushing disabled.
Experiments here are numbered from `exp001` (exp001–exp007 were exp176–exp182
before 2026-11-25). This fork owns `experiments/`, `notes/` and `papers/`;
upstream updates bring only tooling and site changes.

To take upstream's tooling and site improvements:

```sh
./sync-upstream.sh   # merge upstream/main, keeping experiments/, notes/ and papers/ as they are
git status           # resolve any remaining conflicts in files this fork changed
git commit
```

Site identity (name, URL, repository link, custom domain) lives in
[`site/src/lib/site.ts`](site/src/lib/site.ts).
