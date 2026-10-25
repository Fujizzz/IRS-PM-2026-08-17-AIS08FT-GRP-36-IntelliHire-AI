# Repository reorganization — 2026-10-18

Project display name: **IntelliHire AI**. Submission basename:
`IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI`.

## Changes and preserved contracts

The previous application tree was moved as a unit into `SystemCode/`. The Git root now contains
the template README, `ProjectReport/`, `Video/`, `Miscellaneous/`, contribution instructions and
the submission checklist. Existing proposal and technical-report files were copied without
editing their contents; their filenames were standardized. The final report, member details
and both videos remain explicitly pending.

Module relationships, Python package names, dependency locks, model settings, experimental
inputs and stored results were preserved. The only executable source edit is the native
Windows RAG helper's workspace ancestor calculation, accounting for the extra `SystemCode/`
level while keeping its original external `.rag-runtime` location. Its comments and variable
index were updated together.

GitHub Actions stays at the Git root, with commands and artifact paths adjusted to the new
source directory. A release exports `<revision>:SystemCode` as the archive root and includes
the verified player bundle. Existing server-side `backend/`, `deploy/`, service and virtual
environment paths therefore retain their previous layout. Unreal LFS/archive rules moved
with their source tree; backend archives continue to exclude the independent Unreal project.

The initial structural migration retained the local checkout name `AiInterviewer`, its Git
remote and root `.venv/`. A subsequent repository rename uses the formal submission name
`IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI` at
<https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI>, and local `origin` is updated to its `.git` URL.
The existing local checkout and virtual environment paths remain in place. The workflow uses
`GITHUB_WORKSPACE` and a `SystemCode` subtree archive, so the GitHub name does not determine
server paths, service names or application package identifiers.
Private `.env`, local databases, generated files and runtime data moved with their owning
application directories. The configured recommendation-catalog path was updated to refer to
the same relocated file; no credentials are recorded in this document. A private ignored
`.env.before-layout-20261018` retains the pre-migration configuration.

Eighteen local pnpm directory junctions embedded old Windows paths. They were recreated to
point to the same installed packages at their new locations, and pnpm's local store-directory
metadata was corrected. The subsequent build retained the existing pnpm lock and dependency
versions. No application fallback or retry behavior was introduced.

## Verification

The baseline was captured at `ad889e749238c309b6627fa50be815e30ee85c52` before moving the source.
Both Python runs used the same existing Windows `.venv` and test selection.

| Check | Before | After |
| --- | --- | --- |
| Core `python -m pytest -q` | 1652 passed, 14 failed, 9 errors, 24 skipped; 5 subtests passed | Identical outcomes for every collected test |
| Backend `python manage.py test interviews --noinput` | 308 passed | 308 passed |
| Backend `python tools/check_docs.py` | 314 existing diagnostics | Identical diagnostic output |
| Browser/streaming `node --test` suites | Not rerun as a baseline | 157 passed |
| Django `manage.py check` | Covered in baseline backend run | Passed |
| Django `makemigrations --check --dry-run` | Not run separately | No changes detected |
| CLI `main.py --help` | Existing entry point | Passed at the new application root |
| Player `pnpm run build` and `node --check` | Existing generated bundle | Passed after junction repair |
| Local ASGI login request using the existing configuration | Not run separately | HTTP 200 and expected login form; temporary process stopped |
| Workflow YAML, command directories and artifact paths | Previous layout | Parsed and checked against the new layout |
| Release archive plus real rebuilt player | Previous archive convention | Existing `deploy/smoke.py --check-release-bundle` passed |
| Native RAG PowerShell helper | Workspace-level runtime | Syntax and workspace resolution verified |

The archive check used a temporary Git index and an uncommitted tree snapshot. It did not
create a commit or change the real index while generating the preview archive. The archive
contained the application at its root, excluded Unreal and private environment files, and
passed the existing size, path, duplicate-member and player-byte checks.

The known core failures include the existing rubric tests, missing optional RAG dependencies,
Windows symlink permission and the existing CRLF-sensitive ONNX test fixture. They were not
silenced, skipped by new rules or repaired as part of this structural change. Existing
documentation-index findings were not treated as newly passing. The new PowerShell header
and edited workflow/attribute comments were manually checked against their actual behavior.

Detailed before/after logs, JUnit comparisons, source hashes and archive-check results are
saved locally outside the repository at `../output/repository-layout-20261018/`, relative to
the Git checkout root. They are engineering records, not final submission deliverables.

## Boundaries of the pre-commit checks

- GitHub Actions execution or a production release: these local checks preceded publication.
- Real external LLM, RAG corpus/model, PDF sandbox, microphone, camera or UE/GPU acceptance.
- Final report accuracy after its 2026-10-06 technical-document evidence cutoff.
- Complete course submission: the outstanding items remain in `../../SUBMISSION.md`.

The local verification above used uncommitted changes. The structural renames, corresponding
documentation/configuration edits and new submission files form one commit; its eventual CI
and deployment results must be checked separately before claiming production acceptance.

Before publication, remote main advanced to `92d0b1ac5486dfee32ca6d9f9e930760d7f3abe5`.
That upstream change guards optional study-module import tests when `pgvector` or
`psycopg_pool` is unavailable; it does not modify the deployment dependency lock or runtime
behavior. The structural commit is rebased onto that upstream version, retaining the fix at
`SystemCode/tests/rag/test_studies.py`. The before/after counts above describe the original
`ad889e7` comparison, not a new baseline that includes this later upstream test change.
