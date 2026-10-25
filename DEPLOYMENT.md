# 使用者一键部署

从 GitHub 克隆后，在仓库根目录（能看到 `deploy.py` 的目录）运行。此入口安装独立实例，
不需要项目维护者的 SSH 密钥、GitHub Actions secrets 或现有服务器权限。

若只部署核心应用，可跳过体积较大的 UE Git LFS 资源下载（数字人文件保留为 LFS 指针）：

```bash
git -c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.required=false clone --depth 1 https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI.git intellihire
cd intellihire
```

## 本机使用

前置条件：Python **3.11+**，以及已安装并启动的 Docker：Windows/macOS 使用 Docker Desktop
的 **Linux containers**，Linux 使用 Docker Engine；Docker Compose **2.30+**。
首次构建需要访问 Docker Hub、Debian 软件源、PyPI 和 npm。
脚本检查这些条件，不自动安装系统软件或更改 Docker context。

```powershell
python deploy.py local
```

Linux/macOS 可使用 `python3 deploy.py local`。第一次按提示输入面试模型供应商、模型名、
API key 和 DashScope 语音区域；密钥输入不回显。之后打开 **http://localhost:8765/register/**
注册自己的账号。默认仅本机可访问；需要另一端口时首次使用 `--port 9000`。

账号、简历、面试记录保存在 PostgreSQL Docker volume；停止容器不会清空数据。
本机模式也启用账号登录、CSRF 和跨账号隔离，使用显式配置的 PostgreSQL/Celery，
不使用开发模式的匿名访问或 SQLite。

## 服务器使用

在目标服务器克隆项目，安装 Python 3.11+、Docker Engine 和 Compose 2.30+，
将域名的 A 记录指向该服务器，并在安全组/防火墙开放 TCP **80、443**。
这两个端口需要空闲；不要与现有网站抢占端口。IPv6 记录也必须指向可达的服务器。

```bash
python3 deploy.py server --domain interview.example.com
```

将示例域名换成自己的域名。Caddy 自动申请、保存和续期 HTTPS 证书，成功后访问
`https://你的域名/register/`。服务器模式要求域名，不把公网明文 HTTP 作为安装成功。
ASGI、数据库和 Redis 均不直接暴露宿主机端口。

首次证书签发仍受 DNS、端口可达性和 CA 限额影响；HTTPS 检查失败会明确返回失败。
先检查日志和证书签发情况，再显式重新执行原命令。不会自动重试业务请求、回滚数据库，
也不会自动修改防火墙。Docker 权限由操作系统管理员配置，脚本不自动提权。

## 模型配置与功能范围

支持现有 `openai`、`dashscope`、`deepseek` 面试供应商。模型名由使用者提供，不猜测模型，
不更换模型、不修改温度/评分/题目策略以绕过错误。
**当前网页面试使用语音回答，因此还需要有语音服务权限的 DashScope key**；即使面试模型
使用 OpenAI/DeepSeek，语音仍使用 DashScope。区域和兼容 API 地址必须与 key 匹配。
API 供应商账户、额度和使用费用由部署者自行提供。

核心安装包含：账号、简历管理和 PDF 隔离解析、面试网页与 WebSocket、语音接入、
后台评分/报告、历史记录，以及仓库自带的实验岗位目录。现有 4 个面试连接、2 个 PDF
上传名额、模型超时和解析限额保持不变。

独立评分 worker 与 Celery 自动启动。部署探针检查真实数据库、账号访问控制、
WebSocket 握手、Celery 消费以及 PDF 沙箱的文件/网络/进程隔离和合成 PDF 渲染。
探针不调用付费模型、不上传个人简历；模型 key 的有效性和真实麦克风体验仍须完成一次
实际面试验收，不能把容器检查通过解释为真实模型效果已验证。

以下资源在首次核心安装中**明确不启用**，没有失败后静默切换：

- UE/MetaHuman/GPU、像素流信令及 TURN；沿用应用已有的浏览器音频播放路径。
- 外部 RAG 索引和 embedding 模型（`RAG_ENABLED=false`）。
- 独立离线回答结束粗筛模型（路径为空，沿用应用已定义的 Qwen-only 配置）。
- 可选视觉简历校对；需要单独填写 `RESUME_VISION_PROVIDER`、`RESUME_VISION_MODEL`。

首次默认保持模板中的 `EVALUATION_MODE=shadow`、`AI_SECURITY_ENABLED=false`、
`QUESTION_QUALITY_MODE=relaxed`、温度 0 和既有语音模型/音色；相关值写入配置可核对。
这是新的核心部署配置，不是对维护者现有生产配置的修改。

## 无交互配置

首次部署可准备一个**不要提交到 Git**的 `models.env`，然后运行：

```bash
python3 deploy.py local --config /private/path/models.env
# 或
python3 deploy.py server --domain interview.example.com --config /private/path/models.env
```

文件格式为字面量 `KEY=value`，不加引号，不使用 `export`，不展开 `$变量`，
不支持多行值；空行和以 `#` 开头的注释可用。示例：

```dotenv
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=填入自己的密钥
DASHSCOPE_MODEL=填入自己有权限使用的模型名
SPEECH_ENABLED=true
SPEECH_REGION=beijing
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

使用其他面试供应商时改 `LLM_PROVIDER`，另提供相应的 `OPENAI_API_KEY/OPENAI_MODEL`
或 `DEEPSEEK_API_KEY/DEEPSEEK_MODEL`，仍保留 DashScope 的语音配置。
新实例的 Django secret、数据库应用密码、数据库管理员密码和 Redis 密码分别随机生成。
不读取开发者的 `SystemCode/.env`，不把它或 `server_info.txt` 发送给 Docker 构建。

私有状态默认位于仓库根目录 `.deploy/`，已被 Git 忽略；Unix 目录权限 0700、文件 0600，
Windows 设置当前用户及 SYSTEM 的私有 ACL。不要分享 `app.env`、`db.env`、`redis.conf`
或数据库备份；不要使用会展开显示环境值的 `docker compose config` 输出做日志。

以后修改 `.deploy/app.env` 中的模型配置，再执行原部署命令。`--config` 只用于第一次，
重复运行不会覆盖已有密钥。不要单独修改生成的数据库/Redis 密码，这些值必须与服务中
现有凭据一致。可用 `--state-dir /private/path` 指定其他目录；该目录须由你自行排除版本控制。
安装中断产生的部分配置不会被自动覆盖，应先检查保留现场。

## 更新、停止、日志与备份

```bash
python3 deploy.py status
python3 deploy.py logs
python3 deploy.py backup
python3 deploy.py stop
```

`logs` 显示最近 100 行并持续跟踪；Ctrl+C 只停止日志查看。`stop` 保留全部 volumes 和配置。
容器设置 `restart: no`，沿用项目“失败可见、显式重启”的约定；重启主机后需要重新执行部署命令。

更新时先拉取源码，再执行同一部署命令：

```bash
git pull --ff-only
python3 deploy.py local
# 服务器使用 python3 deploy.py server；已保存域名，无须重复输入。
```

过程会先构建和预检，再将 PostgreSQL 备份到 `.deploy/backups/*.dump`，随后停止应用和
worker、执行迁移、重新启动并验证。**更新会中断当前面试，应安排在空闲时段。**
备份失败或沙箱预检失败不会停止正在运行的应用；迁移失败后保持停服，保留日志、备份及
当前数据库状态。备份不会自动删除，也不会自动上传到异机；恢复必须由管理员审查迁移和
备份后操作。禁止用删除 volume 的方式“解决”初始化问题。

同机多实例首次使用不同的 `--state-dir`、`--name` 和本地 `--port`；后续管理命令携带相同
`--state-dir`。Docker context 必须指向运行应用的本机；远程 Docker daemon 的 bind mount
路径不属于此入口的支持范围。

## PDF 容器运行边界

宿主 Linux 内核必须支持非特权 user namespace。只有 PDF worker 允许嵌套 namespace：
禁用该容器的默认 seccomp/AppArmor 外层 profile，使既有 bubblewrap 能建立子沙箱；
worker 仍以 UID 10001、零 capabilities、只读根文件系统和 no-new-privileges 运行。
PDF 子进程继续使用项目原来的空网络、只读挂载、空环境、seccomp 和资源限额。
没有 `privileged` 容器，也没有非隔离解析回退。禁止 user namespace 的宿主机会在预检时失败，
需管理员评估内核策略，脚本不会自动放宽宿主机全局策略。

编排使用 [Docker 的启动依赖](https://docs.docker.com/compose/how-tos/startup-order/)和
[raw env_file](https://docs.docker.com/reference/compose-file/services/#env_file)，
反向代理使用 [Caddy 自动 HTTPS](https://caddyserver.com/docs/automatic-https)和
[无缓冲流式代理](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#streaming)。
原有维护者专用发布流程见 [生产维护文档](SystemCode/deploy/README.md)，与本入口分开管理。

本次实际执行的检查、通过项及尚未验证的边界见
[验证记录](SystemCode/deploy/standalone/VALIDATION.md)。
