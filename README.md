# SECTION 1 : PROJECT TITLE

## IntelliHire AI

An AI-assisted interview practice and feedback system.

**Live Demo:** [https://47.239.50.129/](https://47.239.50.129/)

**IRS Practice Module · AIS08 FT · Group 36 · Course start: 2026-08-17**

Submission name: `IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI`.

Official repository: [IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI](https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI).
This repository follows the [IRS project submission template](https://github.com/IRS-PM/Workshop-Project-Submission-Template).

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
| [ProjectReport](ProjectReport/README.md) | Existing proposal, technical reports and installation/user guide; final report pending |
| [Video](Video/README.md) | Reserved for the two required videos; both pending |
| [Miscellaneous](Miscellaneous/README.md) | Supplementary material and links to reproducibility records |

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

Team: **Group 36 (AIS08 FT)**. Confirm the official member details and work allocations before
submission; they have not been inferred from commit authors or local account names.

| Official full name | Student ID | Work items / contribution | Email (optional) |
| --- | --- | --- | --- |
| Pending team confirmation | Pending | Pending | — |

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

| Video | Status |
| --- | --- |
| Five-minute promotion / business-value / use-case video | Pending — [requirements](Video/README.md) |
| Five-minute system-design / technical-explanation video | Pending — [requirements](Video/README.md) |

## SECTION 5 : USER GUIDE

Start with the [installation and user guide](ProjectReport/Installation-and-User-Guide.md).
The detailed [development guide](SystemCode/README.md), [backend guide](SystemCode/backend/README.md),
[RAG integration guide](SystemCode/rag/docs/TEAM_INTEGRATION.md) and
[digital-human guide](SystemCode/docs/guides/DIGITAL_HUMAN_SETUP.md) describe their respective prerequisites.

Application commands are run from **`SystemCode/`**, or from `SystemCode/backend/` when a command
explicitly targets Django. Git operations are run from the repository root. The local checkout
directory may remain `AiInterviewer` to preserve existing development paths; `origin` points
to the official repository URL above. See [contribution instructions](CONTRIBUTING.md) to
update an existing clone.

## SECTION 6 : PROJECT REPORT / PAPER

The final group report is **pending**. Available material:

- [Existing project proposal](ProjectReport/Proposal/Project-Proposal.pdf).
- [Five existing technical reports](ProjectReport/TechnicalDocumentation/README.md), with evidence
  through 2026-10-06; they require review for subsequent changes.
- [Installation and user guide](ProjectReport/Installation-and-User-Guide.md).

The report outline and required appendices are recorded in [ProjectReport/README.md](ProjectReport/README.md).
Confidential individual reflection and peer evaluation are submitted separately to Canvas.

## SECTION 7 : MISCELLANEOUS

See [supplementary material](Miscellaneous/README.md), [submission filenames and checklist](SUBMISSION.md),
and [contribution instructions](CONTRIBUTING.md). Missing final deliverables are explicitly marked
as pending; empty placeholder reports or videos are not presented as completed work.
