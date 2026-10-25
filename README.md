# SECTION 1 : PROJECT TITLE

## IntelliHire AI

An AI-assisted interview practice and feedback system.

**Live Demo:** [https://47.239.50.129/](https://47.239.50.129/)

**IRS Practice Module · AIS08 FT · Group 36 · Course start: 2026-08-17**

Submission name: `IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI`.

Official repository: [IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI](https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI).
This repository follows the [IRS project submission template](https://github.com/IRS-PM/Workshop-Project-Submission-Template).

**Deploy your own instance:** after installing Python 3.11+ and Docker with Compose 2.30+,
run `python deploy.py local`, or `python3 deploy.py server --domain interview.example.com`.
The first run guides model/voice configuration. See the [end-user deployment guide](DEPLOYMENT.md).

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT

IntelliHire AI supports interview preparation through resume-grounded questions, time-bounded
interview planning, conversational follow-ups and evidence-based feedback. The application
combines a deterministic interview controller with language-model adapters, structured
evaluation, optional retrieval of technical knowledge, and a browser interface with speech and
a digital interviewer. A separate experimental job-recommendation module supports incomplete
candidate profiles; its scores are not hiring probabilities.

The implementation includes a Python interview core, a Django/ASGI backend, browser clients,
RAG ingestion and retrieval tools, and a UE digital-human project. Module tests, experimental
conditions, recorded results and known limitations remain alongside their implementations.
Real interviews require configured external services, and some models and corpus artifacts
must be obtained separately; see the installation guide.

| Repository area | Contents |
| --- | --- |
| [SystemCode](SystemCode/README.md) | Runnable source, dependency locks, tests, training and deployment tools |
| [ProjectReport](ProjectReport/README.md) | Final group report, proposal, technical reports, and installation/user guides |
| [Video](Video/README.md) | Promotion and system-design presentation recordings |
| [Miscellaneous](Miscellaneous/README.md) | Supplementary material and links to reproducibility records |

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

Team: **Group 36 (AIS08 FT)**. Contributions below follow Appendix E of the
[final group report](ProjectReport/IntelliHire_AI_Group_Report.pdf).

| Member | Main contribution |
| --- | --- |
| Du Ying Zeng Andrew | RAG corpus, hybrid retrieval, query encoder and retrieval evaluation |
| Guo Lexin | Digital-human interviewer, expression Agent, speech interaction and Pixel Streaming |
| Luo Ziyu | Competency rubric, evidence-first evaluation, deterministic scoring and scoring validation |
| Sun Zhiquan | Job recommendation, resume processing, answer completion, AI safety and application platform |
| Wang Shunyao | Interview Agent, time-budgeted orchestration, background replanning and question-quality validation |

Student IDs and detailed work allocations are recorded in the final report. The Canvas
submission also includes `member-github.txt` with the full member list and repository URL.

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

| Video | Recording |
| --- | --- |
| Promotion / business value / use-case presentation | [Promotion video](Video/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI-Promotion.mp4) |
| System design / technical explanation | [System video](Video/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI-System.mp4) |

Both recordings are supplied in `Video/` through Git LFS and as separate playable files in
the Canvas submission ZIP. See [video access instructions](Video/README.md).

## SECTION 5 : USER GUIDE

For step-by-step instructions with screenshots, read the
[IntelliHire AI User Guide](ProjectReport/IntelliHire-AI-User-Guide-Annotated.docx).
It covers account access, resume preparation, voice interviews, saved reviews and job matches.
For installation and startup, use the [installation guide](ProjectReport/Installation-and-User-Guide.md).
The detailed [development guide](SystemCode/README.md), [backend guide](SystemCode/backend/README.md),
[RAG integration guide](SystemCode/rag/docs/TEAM_INTEGRATION.md) and
[digital-human guide](SystemCode/docs/guides/DIGITAL_HUMAN_SETUP.md) describe their respective prerequisites.

Application commands are run from **`SystemCode/`**, or from `SystemCode/backend/` when a command
explicitly targets Django. Git operations are run from the repository root. The local checkout
directory may remain `AiInterviewer` to preserve existing development paths; `origin` points
to the official repository URL above. See [contribution instructions](CONTRIBUTING.md) to
update an existing clone.

## SECTION 6 : PROJECT REPORT / PAPER

The [final group report](ProjectReport/IntelliHire_AI_Group_Report.pdf) is available. It covers the business case,
system design, implementation, experiments, findings and discussion. Appendix B contains
the MR/RS/CGS functionality mapping, and Appendix C contains installation and user instructions.

Supporting material:

- [Existing project proposal](ProjectReport/Proposal/Project-Proposal.pdf).
- [Five existing technical reports](ProjectReport/TechnicalDocumentation/README.md), with evidence
  through 2026-10-06; they require review for subsequent changes.
- [Installation and user guide](ProjectReport/Installation-and-User-Guide.md).

The report and supporting documents are indexed in [ProjectReport/README.md](ProjectReport/README.md).
Confidential individual reflection and peer evaluation are submitted separately to Canvas.

## SECTION 7 : MISCELLANEOUS

See [supplementary material](Miscellaneous/README.md), [submission filenames and checklist](SUBMISSION.md),
and [contribution instructions](CONTRIBUTING.md). The Canvas group package contains the member
file, two recordings, repository ZIP and final group report.
