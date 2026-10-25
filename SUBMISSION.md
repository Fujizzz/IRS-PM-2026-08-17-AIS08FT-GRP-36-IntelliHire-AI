# Submission checklist — IntelliHire AI

| Field | Confirmed value |
| --- | --- |
| Course | IRS-PM |
| Course start date | 2026-08-17 |
| Class | AIS08 FT; filename token `AIS08FT` |
| Team | Group 36; filename token `GRP-36` |
| Project | IntelliHire AI; filename token `IntelliHire-AI` |
| Submission basename | `IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI` |
| GitHub repository | [IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI](https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI) |

The local course specification v2.16, pages 9–10, requires the template repository plus a
separate Canvas delivery package. Page 14 lists the deadline as **2026-10-25 23:59**.
The repository template is <https://github.com/IRS-PM/Workshop-Project-Submission-Template>.

## Repository contents

- [x] README organized into the template's seven sections.
- [x] Runnable project under `SystemCode/`, with tests, dependency locks and module documentation.
- [x] Existing proposal and technical reports under `ProjectReport/`.
- [x] Presentation recordings under `Video/`; supplementary material under `Miscellaneous/`.
- [x] Final group report supplied, including the MR/RS/CGS functionality mapping in Appendix B.
- [x] Promotion and system-design recordings supplied and linked from the README.
- [x] Member names, student IDs and work allocations recorded in the final report; `member-github.txt` supplied in the Canvas package.
- [x] GitHub repository renamed to the formal submission name; use the URL above in `member-github.txt`.
- [ ] Obtain required external models/data and verify installation on the demonstration machine.

## Canvas group delivery

The Canvas group package contains the following inside the final `IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI.zip`:

1. `member-github.txt`: group name, all official member names and student IDs, repository URL.
   This file is supplied in the Canvas package; personal details need not be copied into public
   repository material merely to satisfy this package requirement.
2. `IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI-Promotion.mp4`.
3. `IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI-System.mp4`.
4. `IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI-main.zip`: the downloaded repository archive,
   placed inside the outer delivery ZIP as specified by the course.
5. `IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI-Report.pdf`: the final group report.

The template documents the pattern `CourseCode-StartDate-BatchCode-TeamName-ProjectName`;
the example names above use the team's supplied values. Individual confidential reflection
and peer evaluation go to the separate individual Canvas assignment.

## Reproducibility and archives

`SystemCode/.gitattributes` keeps Unreal content in Git LFS and excludes `DigitalHuman/`
from Git source archives, as the existing backend deployment requires. A GitHub source ZIP
therefore does not by itself supply the complete digital-human runtime. The user guide must
remain with the submission and explains full clone/LFS acquisition and external model/data
requirements. Verify those prerequisites on the demonstration machine before final delivery.

Do not zip the working directory indiscriminately: `.env`, virtual environments, databases,
runtime output and caches are local state rather than tracked submission content.
