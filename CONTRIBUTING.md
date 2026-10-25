# Contributing

Application development takes place in `SystemCode/`. Identify the owning module and keep the
change inside that boundary. See [the repository layout](SystemCode/docs/REPOSITORY_LAYOUT.md)
for dependency direction and upload rules. Reports, videos and supplementary submission material
belong in `ProjectReport/`, `Video/` and `Miscellaneous/` respectively.

## Repository address

The official repository is [IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI](https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI).
From an existing clone, update the remote before fetching or pushing:

```bash
git remote set-url origin https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI.git
git fetch origin
git remote -v
```

Keep an existing checkout directory and populated virtual environment at their current paths.
The repository name is independent of local folder names, Python packages and production services.

For Python changes run:

```bash
cd SystemCode
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

Never commit `.env`, API keys, virtual environments, IDE settings, build outputs or runtime data.
Changes to `SystemCode/shared/contracts/` should be isolated because they affect multiple teams.

GitHub Actions stays under the repository-root `.github/`. Its shell steps run in `SystemCode/`,
and production archives contain the contents of that subtree without an extra directory layer.
Keep source code, tests and deployment configuration changes consistent with that boundary.

Existing Windows environments may remain at the repository root `.venv/`: activate that
environment before entering `SystemCode/`, then invoke `python -m ...`. Do not move a populated
virtual environment; its installed entry points can embed its absolute path.
