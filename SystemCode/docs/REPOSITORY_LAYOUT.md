# Repository layout and collaboration rules

The submission repository is organized around the IRS template. `SystemCode/` is the
application root; moving that directory as a unit preserves imports, data paths and module
boundaries. The project display name is IntelliHire AI. Existing Python package identifiers,
service names and dependency locks retain their established values.

Official GitHub repository: [IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI](https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI).
The tree below uses the retained local checkout name; a fresh clone defaults to the formal
repository name. Both layouts use `SystemCode/` as the application root.

```text
AiInterviewer/                    Git checkout root (local path retained)
├── README.md                     seven-section submission overview
├── SUBMISSION.md                 formal name and delivery checklist
├── ProjectReport/                proposal, technical documents and user guide
├── Video/                        pending promotion and system-design videos
├── Miscellaneous/                supplementary submission material
├── .github/workflows/            CI; shell commands target SystemCode
└── SystemCode/                   application root, expanded below
```

```text
SystemCode/
├── agents/       deterministic interview Agent core
├── app/          runnable terminal MVP and temporary concrete adapters
├── backend/      Django/ASGI API and current browser application in backend/frontend
├── frontend/     frontend team ownership boundary
├── evaluation/   evaluation evidence and scoring implementation
├── rag/          corpus ingestion, query encoders and retrieval implementation
├── ai_security/  AI safety and security module boundary
├── shared/       versioned cross-module contracts
├── tests/        Python contract, unit and integration tests
├── docs/         architecture, module specifications, guides and examples
├── DigitalHuman/ independent UE project and local streaming tools
├── deploy/       existing Linux release and service configuration
├── training/     reproducible model training and recorded results
└── security_evaluation/ security experiments and evidence
```

## Dependency direction

```text
frontend → backend API
backend  → shared contracts + Agent public service
app      → shared contracts + Agent public service + temporary adapters
agents   → shared contracts + ports
evaluation / rag → shared contracts + the port they implement
ai_security → shared contracts + explicit public integration points
shared   → no implementation module
```

## Upload rules for new modules

1. Put application changes inside the owning module under `SystemCode/`.
   Backend diagnostic pages stay under `backend/diagnostics/`; the current product UI remains
   under `backend/frontend/`. The separate `frontend/` directory is still reserved for future work.
2. Preserve module-specific manifests such as `package.json`, `pyproject.toml` or lock files
   inside the owning module unless the team explicitly adopts one shared workspace.
3. Do not copy shared request/response models into multiple modules. Interface mapping will be
   handled in a later integration change.
4. Do not import `agents/policies/`, `agents/domain/` or other module internals.
5. Keep generated files, local environments, build output and secrets out of Git.
6. Submit structural moves separately from interface or business-logic changes.

## Runtime and delivery boundaries

- `.env` belongs to `SystemCode/`; local databases and generated assets stay with their modules.
- The existing Windows `.venv` may remain at the Git root. Activate it before changing into
  `SystemCode/`; fresh clones can create their environment within `SystemCode/`.
- GitHub Actions must stay at the Git root. Explicit action artifact paths include `SystemCode/`,
  while shell commands use it as their working directory.
- The production archive exports the `SystemCode` Git subtree without a leading directory.
  The server still receives `backend/`, `deploy/`, and the existing runtime layout.
- Unreal LFS and export rules live in `SystemCode/.gitattributes`, next to the subtree they govern.
- Do not move experiment data out of its owning module merely to fill `Miscellaneous/`; link to it.
