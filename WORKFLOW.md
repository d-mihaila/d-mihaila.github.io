# Working in pinglab_plasticity

How Daria builds experiments, runs them and publishes them, all from VS Code.
Claude reads this file too. The last section is written for Claude.

## The setup

| What | Where |
|---|---|
| Your repository | `~/Desktop/code/oscil-work/d-mihaila.github.io` |
| GitHub | https://github.com/d-mihaila/d-mihaila.github.io |
| Blog | https://d-mihaila.github.io |
| VS Code workspace | `pinglab_plasticity.code-workspace` (File → Open Workspace from File…) |
| Eoin's pinglab | `../pinglab`, read-only in the workspace |
| Results | `.pingstore/runs/`, on your Mac only, not in Git |

- `origin` is your repository. `upstream` is Eoin's; pushing to it is disabled.
- `bun run publish` publishes from the folder it runs in. Run it only here, never in `../pinglab`.

The cycle: write code → compute → analyse → present → write the report → preview →
commit and sync → publish.

## 1. Start a new experiment

1. Pick the next free number. Run `ls experiments` and add one; the next is `exp008`.
   Skip `exp022`. Never reuse `exp176`–`exp182`.
2. Copy the closest experiment as a template:

   ```sh
   cp -R experiments/exp007 experiments/exp008
   ```

3. Rename inside the copy. Press Cmd+Shift+H, search `exp007`, replace with `exp008`,
   set "files to include" to `experiments/exp008`, then Replace All.
4. Remove the renumbering helper. Only exp001–exp007 need it.
   - `recipe.py`: delete the `LEGACY_SLUG` lines and the `source_experiment` function.
   - `analyse.py` and `present.py`: change `recipe.source_experiment(...)` back to `recipe.SLUG`.
5. Change the science: settings in `recipe.py`, then `compute.py`, `analyse.py` and `present.py`.
6. Update the frontmatter of `report/report.mdx`: `title`, `description`, `created_at`
   and `updated_at` (quoted UTC, for example `"2026-11-26T09:00:00Z"`), and `thumbnail`
   (a figure file that `present.py` exports). Keep `id: exp008` and `inputs: ["exp008"]`.
7. Update the experiment's `README.md`: the goal and the commands.

Claude in VS Code can do steps 2–7. Ask, for example: "Create exp008 from exp007
following WORKFLOW.md. The new question is …".

## 2. Run it

Use the VS Code terminal, from the repository root. Each stage prints its run ID on
the last line. Paste that ID into the next command.

```sh
caffeinate -i uv run python experiments/exp008/compute.py
uv run python experiments/exp008/analyse.py --source COMPUTE_ID
uv run python experiments/exp008/present.py --source ANALYSE_ID
```

- `caffeinate -i` keeps the Mac awake during long simulations.
- Analyse and present take seconds. Re-run them any time on an existing compute run.
- List an experiment's runs: `uv run pingstore list --experiment exp008`.
- Never paste `#` comments into the terminal. zsh treats them as arguments.

Checks before committing:

```sh
uv run pytest experiments/exp008/test.py -q
uv run ruff check experiments/exp008
uv run ruff format experiments/exp008
```

## 3. Write and preview

- `report.mdx` holds the abstract, results and captions. `appendix.mdx` holds the
  network diagrams and parameter tables.
- Figures and numbers come from the newest present run automatically. Never type run
  IDs into a report.
- Preview: Terminal → Run Task… → **Blog: preview at localhost:3000**. Open
  http://localhost:3000/exp008. Stop it with Ctrl+C in that terminal.

## 4. Save to GitHub

Source Control (Ctrl+Shift+G): write a message → **Commit** → **Sync Changes**.
This saves code and text. Results in `.pingstore` are not in Git.

## 5. Publish

Terminal → Run Task… → **Blog: publish to d-mihaila.github.io**
(or `cd site` then `bun run publish`).

1. Wait about a minute for **Deploy prebuilt site** to turn green at
   https://github.com/d-mihaila/d-mihaila.github.io/actions.
2. Reload the blog with Cmd+Shift+R.

The blog shows exactly what is on your Mac at publish time: your code and your newest
results.

## Changing an existing experiment

- Text only: edit the report, update `updated_at`, preview, commit, publish.
- New results: re-run the stages. The page switches to the newest present run.

## Using Eoin's work

- Read or copy code from the read-only pinglab folder. Copy it; never import from
  another experiment.
- To use one of his results as an input, copy the run into your store. Then name it in
  `recipe.py`, as exp001–exp007 do with `exp022-r039-compute`:

  ```sh
  uv run pingstore insert ../pinglab/.pingstore/runs/RUN_ID
  ```

- To take his tooling and site updates: Run Task → **Pull Eoin's tooling updates** →
  review in Source Control → Commit → Sync Changes. Your `experiments/`, `notes/` and
  `papers/` are never touched. Do this every few weeks, not daily.

## Backups

`.pingstore` (about 4 GB) exists only on your Mac. Keep Time Machine on. For an
off-site copy, configure an rclone remote named `backup`, then run
`uv run pingstore backup`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Blog unchanged | You previewed but did not publish, or the browser cached it: Cmd+Shift+R. |
| "Figure unavailable" on a page | That experiment has no present run yet. Run analyse and present. |
| Deploy: "No artifacts named github-pages" | Actions → Deploy prebuilt site → Run workflow → branch `gh-pages`. |
| Deploy: "Branch gh-pages is not allowed" | Settings → Environments → github-pages → allow `gh-pages`. |
| Push rejected | Click Sync Changes again; it pulls first. |

## Instructions for Claude

- Daria's standing guidance: write the code, with many visualisations and explanations.
  Never run code for her. She runs simulations, analysis, figures, tests, builds and
  publishing herself. Give her the commands.
- Read `AGENTS.md`, then only the rule file in `notes/rules/` that the task needs.
- This repository is Daria's fork of `eoinmurray/pinglab`. `origin` is
  `d-mihaila/d-mihaila.github.io`; `upstream` is Eoin's, with pushing disabled.
  Publish only from this repository, never from `../pinglab`.
- The fork owns `experiments/`, `notes/` and `papers/`. `sync-upstream.sh` merges only
  upstream tooling and site changes.
- Numbering: the next free ID from `exp008`. Skip `exp022`: its Pingstore runs are
  Eoin's trained networks, used as inputs here. `exp176`–`exp182` are retired.
  exp001–exp007 were renumbered from them on 2026-11-25. Their compute runs keep the old
  numbers; `LEGACY_SLUG` and `recipe.source_experiment()` accept those runs, and
  `site/src/lib/site.ts` maps old numbers to new ones for redirects and links. New
  experiments must not carry these helpers.
- Site identity (name, URL, repository link, custom domain) lives in `site/src/lib/site.ts`.
- Commands for Daria: zsh, no inline `#` comments, run from the repository root unless
  stated otherwise.
- Writing for Daria: formal but plain, short sentences, clear structure, argumentative.
