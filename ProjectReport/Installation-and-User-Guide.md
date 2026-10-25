# Installation and user guide — IntelliHire AI

## Directory convention

The Git repository root contains the submission README, report/video directories and GitHub
Actions. **`SystemCode/` is the application root** and contains the complete previous development
layout. Paths such as `backend/`, `rag/`, `.env` and `pyproject.toml` in module guides are relative
to `SystemCode/` unless an explicit repository-root path is shown.

The official GitHub repository is [IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI](https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI).
The existing local checkout remains `AiInterviewer` to preserve installed environment paths.
Its local folder name is independent of the remote repository name; update existing clones
using the commands in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Existing Windows development environment

The existing environment stays at the repository root `.venv/`, preserving absolute paths in
its installed launchers. The private application configuration moved to `SystemCode/.env`;
the local database and generated assets moved with their owning modules.

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location .\SystemCode
python main.py --help
python backend/manage.py check
```

If activation is restricted, call the interpreter explicitly from `SystemCode/`:

```powershell
..\.venv\Scripts\python.exe backend/manage.py check
..\.venv\Scripts\python.exe -m uvicorn config.asgi:application --app-dir backend --host 127.0.0.1 --port 8765 --ws websockets-sansio
```

## Fresh installation

Install Git and Git LFS, then clone the official repository:

```powershell
git clone https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI.git
Set-Location .\IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI
git lfs pull
```

Git LFS content is required for the digital-human source. A downloaded Git source archive excludes the independent
Unreal project under the existing archive policy, so use a full clone for that component.

Use Python 3.12 for the backend. From the repository root, create a fresh environment **inside
`SystemCode/`** (this is distinct from the retained environment on the existing development machine):

```powershell
Set-Location .\SystemCode
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
```

Copy `.env.example` only for a fresh installation. Fill in your own `DJANGO_SECRET_KEY` and
provider settings; retain existing private values on an already configured machine. External
LLM, TTS and ASR credentials are not supplied in this repository. See the existing
[backend setup](../SystemCode/backend/README.md) and [environment template](../SystemCode/.env.example).

Initialize a new development database, then start the server from `SystemCode/`:

```powershell
python backend/manage.py migrate
python -m uvicorn config.asgi:application --app-dir backend --host 127.0.0.1 --port 8765 --ws websockets-sansio
```

Open <http://127.0.0.1:8765/>. The resume workspace is `/resumes/` and the interview workspace
is `/agent/`. Import and confirm a resume version, select interview settings, and use the
interview controls. Full PDF processing requires the sandbox prerequisites described in
[the PDF guide](../SystemCode/backend/docs/resume-pdf.md); speech requires the configured providers.

The terminal interface can be invoked from the same application root:

```powershell
python main.py resume.pdf --duration-minutes 30 --job-title "AI Engineer"
```

## Digital-human component

The independent UE 5.8 project lives at `SystemCode/DigitalHuman/`. Its internal layout is
unchanged. Use [the setup guide](../SystemCode/docs/guides/DIGITAL_HUMAN_SETUP.md) to obtain the
engine, required plugins, streaming dependencies and built executable. From `SystemCode/`,
run the existing launch scripts in separate terminals after those prerequisites are met:

```powershell
.\DigitalHuman\Tools\start-signalling.ps1
.\DigitalHuman\Tools\start-digital-human.ps1
```

Moving the repository does not build the Unreal executable or verify a physical microphone,
camera, GPU renderer or complete end-to-end voice session.

## External models and RAG corpus

The RAG corpus dump and full ONNX query-encoder artifacts are not in Git. Follow the
[RAG integration guide](../SystemCode/rag/docs/TEAM_INTEGRATION.md) to obtain or reproducibly
build them, install the appropriate RAG dependencies, verify hashes and import the corpus.
The answer-completion model also has separate artifact requirements described in
[its guide](../SystemCode/backend/docs/answer-completion.md). Directory reorganization does not
silently change model choices, feature flags, database contents or experimental settings.

## Development checks and deployment

With the development dependencies available, run application tests from `SystemCode/`:

```powershell
python -m pytest -q
Set-Location backend
python manage.py test interviews --noinput
python tools/check_docs.py
```

GitHub Actions remains in `.github/workflows/` at the repository root. Its release archive
exports the contents of `SystemCode/` directly, so existing Linux service paths such as
`/opt/ai-interviewer/current/backend` remain valid. See
[the deployment guide](../SystemCode/deploy/README.md). Local checks do not establish that
the remote workflow or production deployment has run.
