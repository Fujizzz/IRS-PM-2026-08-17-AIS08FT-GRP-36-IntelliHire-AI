# IntelliHire AI — 开发与运行指南

本目录 `SystemCode/` 是完整运行项目的根目录。仓库提交首页位于
[`../README.md`](../README.md)，报告、视频与补充材料位于仓库根目录的同名目录。
本指南及各模块指南中的相对路径以 `SystemCode/` 为基准；启动前请先进入本目录。

正式仓库：[IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI](https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI)。
已有克隆按[协作说明](../CONTRIBUTING.md)更新 `origin`；现有本地目录和虚拟环境继续使用原路径。

现有 Windows 环境 `.venv/` 保留在上一级仓库目录，先在上一级激活再进入本目录即可；
也可以从本目录使用 `..\.venv\Scripts\python.exe`。新机器按下面步骤在本目录创建环境。
私有配置随应用迁入本目录 `.env`，现有数据库、模型与本地资源随所属模块保留。
完整说明见[安装与用户指南](../ProjectReport/Installation-and-User-Guide.md)。

远端可运行 MVP 已与本地确定性 Agent 核心合并。当前应用保留简历读取、OpenAI/千问接入和终端交互，同时以 `InterviewAgentService` 作为唯一面试决策中心。

## 当前能力

- 读取 UTF-8 TXT 和文本型 PDF 简历；
- 将简历转换为版本化 `CandidateProfile`、结构化项目和可核验 claims；
- 按面试时长规划项目、话题、考察目标、时间分配和预计题数；
- Planner 只规划目标，Question Agent 通过有边界的 ReAct 循环生成具体问题；
- 根据实际耗时和回答进度动态调整剩余计划，题数参数仅作为 safety guardrail；
- 对每轮回答分析缺失信息与是否追问，并提取多个能力维度的原文证据；不再使用 confidence；
- 原子提交状态、问题、反馈和决策日志，重复反馈保持幂等；
- 按岗位能力权重在代码中计算最终分数，LLM 仅负责报告文字；
- 支持 OpenAI 和千问 DashScope；
- 支持网页数字人面试、英文问题朗读、语音实时字幕、自动答题倒计时和可选保存的提前结束；
- 输出问题历史、能力状态、决策日志和 JSON 报告。

后端另提供支持缺失资料的实验性人岗双向排序接口，使用仓库附带的v4-B树模型，
与面试评分流程独立。模型尚未通过整体效果门槛，分数不是录用概率；
安装、输入输出和调用示例见[推荐模型接入说明](backend/docs/recommendation.md)。

## 架构

```text
TXT / PDF resume
       ↓
MVP application shell (app/)
       ↓
canonical contracts (shared/contracts/)
       ↓
InterviewAgentService (agents/)
       ├── deterministic policies
       ├── LLMPort → OpenAI / Qwen adapter
       ├── EvaluationPort → evidence adapter
       ├── RAGPort → optional adapter
       └── RepositoryPort → in-memory MVP adapter
```

旧版 LangGraph 流程、旧 `TypedDict` 状态和独立问题路由已移除，避免出现两个决策中心。

ReAct 实现、Windows 启动命令和执行轨迹查看方法见
[ReAct 架构实现与运行指南](docs/ReAct架构实现与运行指南.md)。

## 仓库模块

```text
agents/      Agent 决策核心
app/         当前终端 MVP 与临时适配器
shared/      跨模块版本化契约
backend/     Django/DRF API、Agent 接入及数字人语音服务
  frontend/  当前面试网页、语音交互与 Pixel Streaming 播放器
DigitalHuman/ UE 5.8 MetaHuman 工程、原生运行插件及本机启动工具
frontend/    产品前端团队预留目录
evaluation/  证据提取、评分标准与评估实现
rag/         语料处理、索引、检索与查询编码器
ai_security/ AI 安全：面试行为许可、语义合格性与输入输出发布边界（已接入文字面试输入输出）
tests/       按 Agent 和 App 分类的 Python 测试
docs/        架构、模块规范、指南与示例
```

依赖方向和团队上传规则见 [`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md)，
中文文件索引见 [`docs/guides/FILE_GUIDE_ZH.md`](docs/guides/FILE_GUIDE_ZH.md)。

AI 安全接口、接入边界及评测结果见 [`ai_security/README.md`](ai_security/README.md)。

## 安装

推荐使用 `uv`：

```bash
uv sync --extra dev
```

也可使用 pip：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## 模型配置

终端 MVP 与 Django 后端统一读取运行项目根目录 `SystemCode/` 下的 `.env`，配置模板也统一为根目录 `.env.example`。
首次配置可在根目录执行 `Copy-Item .env.example .env`；已有文件请直接编辑，避免覆盖。
进程环境变量优先；修改 `.env` 后重启后端。后端启动还需填写独立生成的 `DJANGO_SECRET_KEY`，
语音开关和 TTS/STT 配置也位于同一模板中。

千问：

```dotenv
LLM_PROVIDER=dashscope
DASHSCOPE_MODEL=qwen-plus
DASHSCOPE_API_KEY=你的密钥
```

OpenAI：

```dotenv
LLM_PROVIDER=openai
OPENAI_MODEL=你的模型
OPENAI_API_KEY=你的密钥
```

## 运行

```bash
uv run python main.py resume.pdf \
  --duration-minutes 30 \
  --job-title "AI Engineer"
```

不提供 `--job-title` 时使用通用 AI / 软件工程岗位和均衡能力权重。

`--duration-minutes` 默认 30。可选安全上限为 `--max-questions`（默认 40）、
`--max-questions-per-project`（默认 20）、`--max-questions-per-topic`（默认 8），
均包含主问题和追问。上限过低可能在时间用完前结束面试，并不会自动增加时长。
旧的 `--max-follow-up-per-topic N` 等价于话题安全上限 `N + 1`。
规划、计时、降级行为与输出字段见 [Plan and Execute](docs/PLAN_AND_EXECUTE.md)。

终端运行默认将关键节点静默保存到 `output/interview_时间戳_唯一编号.md`，每次面试只有一个文件。
记录选题依据、工具调用摘要、最终问题、回答评价、修复/兜底原因和最终结果；
不保存完整 Prompt、State、工具返回正文或额外 JSON 文件。中断时保留已有记录。
可用 `--output-dir` 更改目录；这些记录包含面试资料，已加入 Git 忽略规则。
记录不包含模型未返回的内部推理文本，也不会在面试过程中打印。

## 独立后端与流式诊断

`backend/` 提供 Django/DRF 练习接口、SQLite 业务存储和只在内存中处理的 WebSocket 音视频回传测试。
其中 `/ws/agent/` 已接入上面的 Agent 决策流程，并通过 Django/SQLite 保存面试数据；练习接口与音视频回传诊断仍独立运行。
当前面试网页、语音交互和数字人播放器统一维护在 `backend/frontend/`；后续由后端团队迁入根目录 `frontend/`。
安装与启动见 [后端说明](backend/README.md)，模块职责和函数注释规范见 [代码阅读指南](backend/docs/code-guide.md)。

三个浏览器页面均支持中文、English 和跟随系统；语言选择保存在浏览器中，只影响界面和诊断文案，不改变面试协议或评分逻辑。

## 数字人面试官

使用 UE 5.8 与 MetaHuman 构建面试官角色和固定面试场景，通过 Pixel Streaming 将画面和声音传输到网页。
数字人负责面试呈现和语音交互；问题、追问、回答评价和报告继续由既有 Agent 流程处理。
默认输出为 **1920×1080、30 FPS**，采用 H264 编码。支持本机 GPU 配合部署后端提供
数字人网页串流，通过 SSH 连接私有信令，并提供登录认证的同源 WSS 入口及 TURN 中继。

网页支持自动朗读英文问题、重新朗读、打断朗读、麦克风回答和实时转录。
题目朗读结束后，主界面倒计时准备 10 秒，到点自动开启麦克风；关闭朗读时从题目显示开始计时。连续 5 秒没有语音活动或收到独立结束检测事件时，自动完成转写并提交；开始／结束回答按钮已移除。准备和录音时间沿用原服务端面试计时，不修改题数上限、评分规则或模型实验条件。
也可用自然语言表达回答结束：配置训练产物后，服务器上的 INT8 多语 MiniLM 二分类
模型先粗筛定稿片段，通过后交给独立 Qwen Flash 确认。合并连续转写并限制请求频率，
无关键词规则筛选；确认结束且 3 秒无补充后
收尾完整转写并通过 MCP 提交，进入评价及下一步。
补充内容会撤销语义结束凭据；完整最终转写仍走普通回答审查路径自动提交，不使用无效凭据。配置和协议见[独立回答结束检测](backend/docs/answer-completion.md)。
页面不提供文字回答输入。空白最终转写以 `skip` 记录未作答，不生成能力评分；转写失败明确停止，不使用字幕草稿代替回答，也不自动重录。录音保持原有 120 秒上限，到限后的完整最终转写会自动提交。

用户可随时选择“结束面试”：评判并保存会完成当前语音转写、评价已完成回答并直接生成报告；结束且不保存会取消本次工作并从历史移除当前面试，保留简历。若模型请求已被接收，保存选择等待该请求完成再结束；无保存选择可立即取消本地请求。已经发送给供应商的同步调用仍遵循原有取消边界。打开结束选择时暂停自动提交，选择继续可恢复。

此更新需要从 `backend/` 运行 `python manage.py migrate`，应用 `0010_agent_automatic_end` 请求类型约束。
控制器提供 Idle、Listening、Thinking、Speaking、Interrupted 状态，供角色动画扩展使用。

| 目录 | 职责 |
| --- | --- |
| `DigitalHuman/` | MetaHuman 角色、`L_Interview` 场景、原生音频播放与音频驱动接口 |
| `DigitalHuman/Tools/` | 本机信令、串流依赖安装、UE 启动和 Windows 打包 |
| `backend/interviews/speech/` | 百炼 TTS/STT、临时 WAV 与语音 WebSocket |
| `backend/interviews/avatar/` | 网页串流配置与经过认证的私有信令代理 |
| `backend/frontend/` | 面试页面、语音交互、PCM 采集与语音字幕 |
| `backend/frontend/digital-human/` | UE 5.8 官方 Pixel Streaming SDK 包装、构建配置与前端测试 |
| `deploy/pixel-streaming/` | 独立信令服务、固定版本依赖安装与服务器运行配置 |

语音配置与模型密钥统一放在根目录 `.env`：

```dotenv
SPEECH_ENABLED=true
SPEECH_TTS_MODEL=qwen-audio-3.0-tts-flash
SPEECH_TTS_VOICE=loongeva_v3.6
SPEECH_STT_MODEL=qwen-audio-3.1-asr-flash-streaming
```

使用新加坡业务空间的 `DASHSCOPE_API_KEY`；语音 WebSocket 地址与文本模型的 `DASHSCOPE_BASE_URL` 分别管理。
配置模板见 [`.env.example`](.env.example)，密钥只由服务端读取。

首次准备依赖，在运行项目根目录 `SystemCode/` 执行：

```powershell
python -m pip install -r backend/requirements.txt
python backend/manage.py migrate
.\DigitalHuman\Tools\setup-streaming.ps1
```

在三个终端分别启动后端、信令和数字人：

```powershell
# 终端 1：后端语音与面试接口。
python -m uvicorn config.asgi:application --app-dir backend --host 127.0.0.1 --port 8765 --ws websockets-sansio

# 终端 2：本机 Pixel Streaming 信令。
.\DigitalHuman\Tools\start-signalling.ps1

# 终端 3：已打包数字人程序；开发时可追加 -EditorGame。
.\DigitalHuman\Tools\start-digital-human.ps1
```

打开 [数字人面试页面](http://127.0.0.1:8765/agent/)，点击“连接／播放数字人”，填写简历后开始面试。
角色组装、场景配置与打包步骤见 [数字人操作说明](docs/guides/DIGITAL_HUMAN_SETUP.md)，

## 测试

```bash
uv run pytest -q
uv run ruff check .
python -m tests.app.smoke_interview
```

离线测试不会调用真实模型。`tests/app/live_interview.py` 是需要主动运行的真实 API 测试入口。

## 已知边界

- 终端 CLI 的 PDF 只支持可提取文本；后端页面另提供规则提取与多模态转写，见 [PDF 简历解析](backend/docs/resume-pdf.md)；
- MVP Repository 仍是内存实现，关闭进程后状态不会保留；
- RAG 端口和数据库端口已定义，生产适配器仍需由对应模块接入；
- 最终报告是辅助评估结果，不应直接作为自动化录用决定。

## 回滚点

合并前的本地 Agent 原始版本保存在：

- 分支：`codex/local-pre-mvp-merge-20260911`
- 标签：`local-pre-mvp-merge-20260911`

查看原始版本：

```bash
git switch codex/local-pre-mvp-merge-20260911
```
