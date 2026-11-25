---
title: Compute Access
author: Daria Mihaila
collection: notes
description: Host access and command examples, read only when needed.
created_at: "2026-10-29T17:42:34Z"
updated_at: "2026-11-25T13:40:20Z"
---

# Compute access

Read the relevant host section when connecting or preparing a job. Authorization
and host roles are defined in [Compute Rules](/notes/rules/compute).
Commands below are examples, not startup steps.

## Local Mac — development and analysis

Use for editing, small tests, plotting and analysis. Not for CUDA training.

```sh
cd /Users/daria/Desktop/code/oscil-work/d-mihaila.github.io
uv sync --dev
uv run python experiments/exp001/compute.py --help
```

## Adding a host

No remote hosts are configured yet. For each new host, add a section here with:

1. Its role: development, orchestration or heavy compute.
2. How to connect, for example `ssh <alias>`, without credentials.
3. Where heavy files live and which scheduler or GPU-selection rules apply.
4. Example submission, monitoring and cancellation commands.
