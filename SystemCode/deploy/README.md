# 单机生产部署与维护

## 提交模板整理后的路径

本目录位于 `SystemCode/deploy/`；开发命令以 `SystemCode/` 为运行根目录。
GitHub Actions 仍位于仓库根目录 `.github/workflows/deploy.yml`，其 shell 步骤默认在
`SystemCode/` 执行，上传和下载 artifact 的路径显式包含 `SystemCode/`。
打包使用 `git archive <提交SHA>:SystemCode`，将子树内容直接作为发布包根目录，再追加同次
CI 验证的播放器文件。因此服务器的 `current/backend`、`current/deploy` 和 `.venv` 路径
保持原样，现有 systemd 单元及固定部署入口无需随源码外层目录迁移。

正式 GitHub 仓库为 [IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI](https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI)，
工作流入口为 [Verify and deploy](https://github.com/Fujizzz/IRS-PM-2026-08-17-AIS08FT-GRP-36-IntelliHire-AI/actions/workflows/deploy.yml)。
工作流使用 `GITHUB_WORKSPACE` 定位检出目录，不依赖仓库名称；`main` 推送仍触发完整验证和部署。
部署所需的仓库 secrets 为 `DEPLOY_SSH_KEY` 和 `DEPLOY_KNOWN_HOSTS`；SSH 用户、服务器地址、
`/opt/ai-interviewer`、数据库和 systemd 服务名称沿用原配置。改名后的发布须同时核对
Actions 结果、`deployed-revision`、当前发布目录及真实业务探针，不能仅以仓库页面可访问作为成功依据。

应用归档属性位于 `SystemCode/.gitattributes`，仍将独立 UE 工程排除在后端发布包之外。
仅同步提交结构不会发布到服务器；后续推送 main 时才按既有 workflow 执行验证和部署。

应用部署到 `https://47.239.50.129/`，文字面试入口为 `/agent/`。
主页公开；通过 `/register/` 注册、`/login/` 登录，每个账号的面试与练习记录独立，题库共用。
账号仅需非空用户名与密码，无邮箱、验证码或密码复杂度要求。

## 实际部署

| 项目 | 配置 |
| --- | --- |
| 系统 | Ubuntu 22.04.5 LTS |
| 应用解释器 | Python 3.12.14，独立安装，不替换系统 Python |
| 数据库 | PostgreSQL 14，数据库 `ai_interviewer`，应用角色 `aiinterviewer` |
| 应用服务 | `ai-interviewer.service`，低权限用户 `aiinterviewer`，仅监听 `127.0.0.1:8765` |
| 公网入口 | Nginx 80/443，HTTP 跳转 HTTPS，Django session 账号认证，WebSocket/NDJSON 不缓冲 |
| TLS | Let's Encrypt IP 短期证书，`ai-interviewer-certbot-renew.timer` 每六小时检查续签 |
| 发布目录 | `/opt/ai-interviewer/releases/<完整提交 SHA>`；`current` 符号链接指向当前版本 |
| 应用依赖 | 每个发布目录的 `.venv`；固定依赖保存在 `requirements-linux.lock.txt` |
| 凭据 | `/etc/ai-interviewer/app.env`，root 所有，权限 `0600`；模型配置沿用本地 |
| 用户账号 | PostgreSQL 中的 Django 用户与 session；密码仅保存哈希 |
| PDF 隔离环境 | `/opt/ai-interviewer/pdf-sandbox`，系统 Python 3.10 专用解析依赖和 bubblewrap |
| 容量锁 | `/var/lib/ai-interviewer/capacity`；保持原 4 个面试连接、2 个 PDF 上传限额 |
| 回答结束粗筛模型 | `/var/lib/ai-interviewer/models/answer-completion/v2-20261003`；离线 INT8，配置 `ANSWER_COMPLETION_GATE_PATH`，通过后仍需 Qwen |
| 生产语音 | `SPEECH_ENABLED=true`、`SPEECH_REGION=beijing`；沿用原 TTS/STT 模型、音频格式与超时 |
| 数据备份 | `backup-postgresql.sh` 手动创建 `/var/backups/ai-interviewer/*.dump` |

服务器使用新建数据库，只由既有迁移初始化两道练习题；未导入本地 SQLite 历史。
部署使用 Git 提交的源码归档，并加入同次 CI 验证的数字人播放器；
未上传 `.git`、本地数据库、日志、缓存或虚拟环境。
`.gitattributes` 的 `export-ignore` 将 `DigitalHuman/` 排除在后端发布归档之外；
UE 工程与 MetaHuman 资源仍保存在 Git/Git LFS，通过克隆仓库获取，在 GPU 机器上单独打包运行。
后端归档保留网页、语音服务及面试模块，并遵守压缩包 128 MiB、解压后 256 MiB 的发布上限。
Redis/Celery 接入不修改模型、采样参数、评分策略、推荐权重、PDF 参数或容量限制。

回答结束模型的 ZIP 来自私有 Kaggle V2 输出，单独上传到上述版本目录，未放入源码归档。
校验 ZIP 和 `manifest.json` 摘要、固定文件白名单、单条输入契约及 `release_approved`
后再启用；版本目录须允许应用用户读取，环境文件仍使用 root 私有权限。
服务器只需已有依赖锁中的 ONNX Runtime/tokenizers/NumPy，不安装 PyTorch 训练依赖。
部署探针校验模型真实加载/推理，并用临时认证会话访问线上 WebSocket，检查 MCP 握手
和 `finish_current_answer` 工具注册；它不调用结束工具、不录音，也不能证明意图准确率。
训练/推理及真实 Qwen 检查边界见[训练记录](../backend/docs/answer-completion-training.md)。
2026-10-03 已将功能提交 `6d3e2ca4b2cd8974502949931c89ec8b7aeb1f9a` 部署到生产，
模型单独校验后启用；HTTP、数据库、Redis、Celery、真实模型加载与 MCP 注册检查通过。
发现原语音开关关闭且默认新加坡端点与当前 key 区域不匹配；新加坡握手返回 401，
北京握手及真实合成/ASR 检查通过后，显式更新上述两个语音配置，并验证运行中进程已读取。
未增加自动区域切换或改变供应商模型/期限，修改前的私有环境备份保存在
`/etc/ai-interviewer/app.env.before-completion-20261003` 和
`/etc/ai-interviewer/app.env.before-speech-20261003`；原始文件权限保持 `0600`。
当前机器的 SSH 管理信息按用户要求保存在运行项目根目录 `SystemCode/server_info.txt`，已加入 Git 忽略，
不进入 Git 提交、Kaggle 数据或源码发布归档。真实麦克风及完整设备体验仍须现场验证。

2026-10-05 已完成安全引擎连接保活专项更新：私有 `app.env` 显式配置
`AI_SECURITY_KEEPALIVE_SECONDS=60`，只替换安全传输模块并重启 ASGI；原模型、5 秒
审查门槛、SDK 零重试及其他服务配置不变。更新前的原传输文件和私有配置保存在
`/var/backups/ai-interviewer/security-network-20261005-181442/`，目录 0700、备份 0600。
当前整版本标识仍是 `c3d96fd617419f96ef0e44d2845b63ab1e710339`，专项文件哈希另存回执；
本地代码尚未提交推送，后续整版本发布必须包含本次修复，避免覆盖已生效的传输代码。
原部署记录称服务器的 324 次完整安全回归/稳定性/并发检查均通过原门槛，部署后 5 次实际安装
引擎检查均正确。该记录引用的专项文档 `security_evaluation/network_optimization/README.md`
当前未包含在仓库中，上述部署结论仍需对照原回执复核。其他已归档实验见
[安全评测目录](../security_evaluation/README.md)，不将其作为这次专项部署的替代证据。
本次同时合并远程 main 的 Plan and Execute 版本；面试时长和题数安全上限沿用该远程版本。
生产配置通过 `DJANGO_SETTINGS_MODULE=config.production` 显式启用；默认开发启动仍使用 SQLite。
PostgreSQL 配置不全或连接失败时明确报错，不回退到 SQLite。

### 2026-10-12 面试画面比例修复

在当时的 `480b5be528d48fe594436feb104b8f4e2a32dae2-ttspool-20261012` 发布目录中，
仅更新 `backend/frontend/agent.css` 与 `backend/frontend/workspace.css`。
画面改为按主栏宽度采用 16:9、最低 320px；不超过 520px 的窄屏采用 4:3、最低 260px。
已移除按窗口高度减去 510px 的覆盖规则，短窗口通过页面滚动访问后续控件。
文件替换前取得发布锁、核对原摘要并备份；本次没有重启、数据库迁移或配置变更。
原文件与摘要回执保存在 `/var/backups/ai-interviewer/interview-layout-20261012-214128/`。
认证公网 HTTPS 已验证两份 CSS 与修复文件一致，面试页、健康接口及服务状态正常；
临时探针账号和会话已清理。线上获取的 HTML/CSS 在六组浏览器视口下通过布局检查，
未启动真实面试、摄像头、麦克风或 UE 视频流。
本地回执位于 `output/interview-responsive/deployment/deployment-receipt.json`。
这次样式专项更新当时未创建 Git 提交；10 月 16 日的后续源码发布未包含该修复，
重新带入旧高度规则，导致短窗口中的面试画面再次变扁。

### 2026-10-16 将面试布局修复纳入源码发布

上述两份 CSS 的比例与最低高度规则、实现注释及本说明一同维护在 Git 中，
由 `main` 的既有验证与部署流程打包发布。`agent.css` 统一定义画面尺寸，
`workspace.css` 只安排主栏和侧栏，不再覆盖面试画面的高度。
发布后核对认证 HTTPS 返回的两份样式与该提交一致，并检查固定宽度下改变窗口高度
不会压缩画面；窄屏检查 4:3 与 260px 最低高度。最低高度优先于首选比例，
短窗口允许正常页面滚动，不要求全部控件挤入首屏。

## 维护命令

以下命令通过 SSH 在服务器以 root 执行：

```bash
systemctl status ai-interviewer ai-interviewer-celery redis-server nginx postgresql --no-pager
journalctl -u ai-interviewer -n 100 --no-pager
systemctl restart ai-interviewer
systemctl list-timers ai-interviewer-certbot-renew.timer --no-pager
bash /opt/ai-interviewer/current/deploy/backup-postgresql.sh
```

修改 `/etc/ai-interviewer/app.env` 后重启应用。不要把凭据文件放进发布包。
维护账号可在加载生产环境后使用 `manage.py changepassword 用户名`；不要直接写入明文密码。
应用退出后保留失败状态和日志，不自动重放面试或模型请求；排查后显式重启。
数据库备份尚未配置周期任务或异机保存；磁盘上既有归档不会自动删除。

### 个人岗位推荐来源

本次生产部署明确选用训练数据同源的 100 个实验岗位。ASGI 的 systemd 服务在私有
`app.env` 之后读取当前发布的 `deploy/recommendation.env`，仅覆盖
`RECOMMENDATION_JOB_CATALOG`，指向
`/opt/ai-interviewer/current/backend/interviews/recommendation/data/experience-jobs.json`。
此文件不含密钥，随发布记录并审查岗位来源；`current` 切换后路径自动对应新版本。
开发配置仍须显式选择来源，未配置/文件无效仍报错，不切换备用目录。

界面保留“非实时招聘职位”与实验模型标记；不更改粗排权重、特征单位、缺失处理或训练条件；个人中心按用户指定 20/5 进行两阶段推荐。
来源及转换记录见 [体验岗位说明](../backend/interviews/recommendation/data/README.md)。

发布验收使用临时账号、保存简历及数据库会话，向实际运行的 ASGI 发送带 Session/CSRF 的
推荐请求，要求完整目录 100 岗粗排、前 20 岗送入 API，最终返回 5 个唯一合法 ID、
双语简短理由及实验标记，并在成功或异常后清理临时账号、简历和会话。
验收会发起一次真实 LLM API 请求并产生服务商用量，复用私有 `app.env` 的既有文字模型
配置与超时；不重试、不增加等待、不使用粗排替代失败精排。HTTP/数据库/Redis/Celery
与岗位验收全部通过才记为部署成功。此检查验证功能接通，不证明真实招聘效果。

更新应用前先备份。把新源码解压到新的发布目录，在独立环境中验证锁定依赖与迁移，
停止应用后切换 `current` 并执行迁移，再启动和检查健康接口。
数据库 schema 变更的回滚必须单独审查迁移或恢复备份，不能仅靠切回源码。
更新期间不得删除共享容量锁；由服务完全停止后再进行需要清理运行状态的维护。

```bash
set -a
. /etc/ai-interviewer/app.env
set +a
cd /opt/ai-interviewer/current/backend
runuser -u aiinterviewer -- /opt/ai-interviewer/current/.venv/bin/python manage.py migrate --noinput
runuser -u aiinterviewer -- /opt/ai-interviewer/current/.venv/bin/python manage.py check --deploy
curl -fsS -H 'Host: 47.239.50.129' -H 'X-Forwarded-Proto: https' \
  http://127.0.0.1:8765/login/ > /dev/null
```

首次重建服务器时，先安装 Nginx、PostgreSQL、bubblewrap、libseccomp2、libgomp1 和系统
Python 的 pip/venv；另装 Python 3.12.14 与独立应用虚拟环境，安装锁定依赖。
PDF 专用依赖须由 `/usr/bin/python3` 根据 `backend/sandbox/requirements.txt` 安装到
`/opt/ai-interviewer/pdf-sandbox/lib`；`tools/usr/bin/bwrap` 指向 `/usr/bin/bwrap`。
建立应用 Unix 用户、PostgreSQL 非超级用户与数据库；根据 `app.env.example` 填写独立随机
数据库密码、Django secret 和原模型配置。不得把系统 PostgreSQL 管理员凭据交给应用。

先配置只开放 `/.well-known/acme-challenge/` 的 80 端口 Nginx 站点，并在云安全组允许
TCP 80/443，随后申请证书：

```bash
/opt/ai-interviewer/certbot/bin/certbot certonly --non-interactive --agree-tos \
  --register-unsafely-without-email --preferred-profile shortlived \
  --webroot --webroot-path /var/www/ai-interviewer-acme \
  --ip-address 47.239.50.129 --cert-name 47.239.50.129
```

签发后安装 `nginx.conf` 和三个 systemd 文件。续签 service/timer 的实际文件名分别为
`ai-interviewer-certbot-renew.service`、`ai-interviewer-certbot-renew.timer`。
运行 `nginx -t`、`systemctl daemon-reload`，启用应用与续签 timer 后重载 Nginx。
禁止直接将 8765 或 PostgreSQL 端口开放到公网。80 需要持续保留 ACME 路径供续签。

## 本机 GPU 与远端像素流

Windows GPU 机器运行数字人 EXE，默认 **1920×1080、30 FPS**；本机
`DigitalHuman/Tools/start-render-tunnel.ps1` 通过 SSH 将本地 8888 转发到服务器的
`127.0.0.1:8888`。服务器仅运行信令与 coturn，浏览器通过已登录的 Django
`/ws/avatar/` 连接信令，视频/声音通过 WebRTC 直连或 TURN 中继传输。
后端私有配置设置 `AVATAR_REMOTE_ENABLED=true`；上游保持
`AVATAR_SIGNALLING_UPSTREAM=ws://127.0.0.1:8889`，不向公网开放 8888/8889/8765。
渲染端使用更新后的 EXE，运行 `start-digital-human.ps1 -SpeechOrigin https://47.239.50.129`。
后端 TTS 返回绑定音频 UUID、10 分钟有效的签名下载 URL；EXE 无需浏览器会话即可
下载该段 WAV，并用同一份音频驱动声音和口型。合成、识别及其他接口继续要求登录。

信令独立安装到 `/opt/ai-interviewer-pixel`，使用 Git 和 Node.js 22.14 以上的
22 LTS 或 24 LTS 版本；现有服务保留专用 Node 24.19。在该目录放入
`deploy/pixel-streaming/` 的文件后执行：

```bash
npm run bootstrap
npm test
```

bootstrap 固定 Epic 基础设施提交 `6b8cfb460bda09703e85178f1f77aa6faec9e890`，
按该提交的 npm 锁文件仅安装并编译 Common/Signalling；生成内容在忽略的
`.infrastructure/` 中。该版本原生包含每连接 `peerOptionsProvider`，无需补丁或未提交的
`libs/*.tgz`。首次构建需要访问 GitHub 与 npm；重新发布信令时再次构建并重启其服务。
信令目录由 root 安装，运行用户仅需读取；日志写入 systemd 的 `StateDirectory`。

信令 systemd 模板为 `deploy/pixel-streaming/ai-interviewer-pixel.service`，其独立私有配置
`/etc/ai-interviewer/avatar.env` 保持 root 所有、0600，设置 `AVATAR_PUBLIC_ORIGIN`
为当前 HTTPS 网站 origin、`AVATAR_TURN_HOST` 为 TURN 公网地址，并配置至少 32 字符的
随机 `AVATAR_TURN_SECRET`；同一共享密钥用于 coturn `static-auth-secret`。不要提交密钥。
`AVATAR_FORCE_RELAY=true` 可验证中继，正常可设置 `false`。严格检查浏览器 Origin，
每连接生成一小时 TURN 凭证；超过一小时的演示先重启 EXE，以获得新的 streamer 凭证。
保持 warning 日志级别，避免 Epic 调试日志记录 ICE 凭证。

云安全组与主机防火墙均允许 TURN **UDP/TCP 3478** 和中继 **UDP 49160–49200**，
coturn 显式设置公网/内网映射。沿用 `user-quota=4,total-quota=16,max-bps=1250000,
bps-capacity=5000000` 与私网/回环 Peer deny；总带宽为预留容量，旧连接未释放可能
导致新 Allocate 返回 486，应先清理旧测试连接再复测。UDP/TCP 强制中继均已验证。
这套独立渲染服务不修改现有 main 自动部署入口或容量限制。

## 已验证与边界

- 队列版本在本地 SQLite 和服务器临时 PostgreSQL 均通过 109 项后端测试，核心 211 项测试和前端 26 项测试通过。
- PostgreSQL migrations 与 `makemigrations --check --dry-run` 通过；健康接口报告 `postgresql`。
- 公网 HTTPS 主页与账号页公开；未登录 API 返回 401、面试页跳转登录，登录后可访问自己的记录。
- 注册、登录、退出、CSRF、跨账号读写隔离及退出后现有 WebSocket 失效均通过公网实测。
- 登录/注册页面通过真实浏览器中英文布局、登录跳转和退出检查；6 项 PDF 客户端测试通过。
- 公网 WSS 两个入口握手通过，并通过真实千问模型完成一题合成面试、评价与报告。
- 真实 PDF 沙箱通过文件/环境/网络/派生进程隔离、内存限制、超时、异常和取消测试。
- 实际 systemd 服务下完成合成 PDF 上传、隔离提取、视觉校对和最终 NDJSON result。
- IP 证书签发与 Certbot 续期 dry-run（含 Nginx 重载钩子）成功。
- 合成面试测试记录已按其唯一 ID 清理；初始数据库备份已实际恢复到临时库验证，随后删除临时库。
- 修改的 Python 文件通过 Ruff、原检查器的 Python 声明/目录校验，并已人工核对注释。
- 项目全量 `python tools/check_docs.py` 在本地出现访问违规，在服务器退出 139；未修改
  检查器或降低校验标准，不能声称全量注释自动检查通过。
- `check --deploy` 保留 IP 部署不启用子域 HSTS 或 preload 的两项警告；CSRF 保护已启用。
- 账号使用 Django 密码哈希与安全 session Cookie；HTTP 写请求含 CSRF 校验，WebSocket 验证 session 归属。
- 历史空归属记录不会分配给新账号；生产启用账号迁移前没有面试或练习记录。
- 未进行并发负载测试、真实浏览器设备授权测试或重启整台主机的演练。
- 本次功能代码、迁移、测试、对应注释与部署文档纳入同一次 Git 提交；全量注释检查仍受上述崩溃阻碍。

ASGI 下的数据库连接配置依据 [Django 5.2 数据库说明](https://docs.djangoproject.com/en/5.2/ref/databases/)。
IP 证书与续签设置依据 [Let's Encrypt / Certbot 官方说明](https://letsencrypt.org/2026/03/11/shorter-certs-certbot)。

## Redis 与 Celery

生产的 PDF 上传由独立 `ai-interviewer-celery.service` 执行，页面仍使用同一条 NDJSON
连接接收排队、提取、校对和完成事件；取消/关闭连接通过 Redis 消费标记通知 worker。
面试模型仍按既有 WebSocket 流程执行。Celery 两个进程保持两份 PDF 容量，每份三页视觉并发不变。

Redis 仅绑定 `127.0.0.1`/`::1`，随机密码保存在 root 私有配置；0 号库存消息，1 号库存
短期输入与进度。禁用 RDB/AOF，512 MiB 上限、noeviction，避免候选人资料写入 Redis 持久文件。
Celery 消息只包含随机任务 ID；正文被 worker 原子取走并删除，禁止重复消息重新调用模型。
正常完成或取消清理临时键；进程崩溃时正文/事件最多保留一小时，存活标记在秒级过期。
客户端停止续期十秒后 worker 取消；worker 心跳失效则前端明确报错。已被模型服务接收的请求
仍可能计费。禁用任务重试、发布重试和 broker 自动重连；故障后需排查并显式重启，不切回 inline。

本地默认 `PDF_TASK_EXECUTION=inline`；显式配置 `celery` 时需要启动 Redis 并设置
`CELERY_BROKER_URL` 和 `PDF_TASK_REDIS_URL`，随后在 backend 运行：

```bash
celery -A config.celery:app worker --loglevel=WARNING --concurrency=2
```

生产配置强制 Celery。环境、密钥与模型参数位于 `/etc/ai-interviewer/app.env`；
Redis 专用配置为 `/etc/redis/ai-interviewer.conf`。两个应用服务均由 systemd 管理并开机启动。

## 自动部署

仓库 `.github/workflows/deploy.yml` 在 **push 到 main** 时自动运行，也可在 Actions 手动触发。
本地修改或仅 commit 尚未 push 不会发布。流程固定 pnpm 11.25.0，以 frozen lockfile
构建 `backend/frontend/digital-human/dist/pixel-player.js`，检查语法及网页需要的三个导出；
核心、后端、前端测试全部成功后保存单文件 artifact。deploy job 下载同 run 的 artifact，
在 `git archive --format=tar` 后用 GNU tar 仅追加上述精确路径，预检非空、摘要一致且只出现一次，
再 gzip 上传至原固定发布入口。归档仍排除 UE、依赖缓存及真实 env；dist 继续忽略、不提交 Git。
发布后的 smoke 使用临时真实登录会话，要求播放器路由返回 JavaScript 且字节与该发布文件一致，
随后清理探针账号/session；不放宽匿名访问规则，也不新增模型请求。
Actions 使用独立 `DEPLOY_SSH_KEY` 和固定 `DEPLOY_KNOWN_HOSTS`；密钥只允许执行发布入口，
禁用交互 shell、端口转发和代理转发，不使用 root 密码。服务器入口由 root 安装为
`/usr/local/sbin/ai-interviewer-deploy`，更新该入口须显式审查并安装新版脚本。

发布按完整 SHA 建立独立目录及虚拟环境，进行配置检查、迁移差异检查和 PostgreSQL 备份，
再停止 ASGI/worker、执行迁移、切换 current 并启动服务。发布会短暂重启服务，现有面试连接将断开，请在空闲时段推送。随后检查 HTTP、数据库、Redis，
并实际发送/消费一个无模型调用的 Celery 探针。验证成功后写入 `/opt/ai-interviewer/deployed-revision`。
并发 workflow 串行运行，服务器另有文件锁；同一成功版本再次部署只验证健康。

失败在 Actions 中明确显示，日志保留；没有自动重试或数据库回滚。数据库迁移失败时保持停服，
人工审查备份和 schema 后恢复。失败发布目录保留，修正代码用新提交部署。旧版本和数据库备份
不会自动删除，目前需管理员按磁盘占用维护。

验证记录：main 推送已实际触发 GitHub Actions 完成测试和服务器切换；线上 PDF 经 Redis/Celery、沙箱和真实视觉模型返回 result，取消在 worker 日志确认，损坏 PDF 明确返回 error。发布归档已验证普通文件可用，拒绝路径穿越、链接和环境秘密文件。

## 面试安全引擎临时停用

后端当前默认 `AI_SECURITY_ENABLED=false`，与 `.env.example` 一致；服务器环境显式设置的值优先。
关闭时面试网关不创建安全引擎或审查模型，跳过安全扫描预算和语义审查。
这是配置选择，不是审查失败后的自动放行；显式设置 `true` 并重启即可恢复原有检查。
独立安全引擎 CLI 与离线评测不受此后端开关影响。

登录、数据归属、请求生命周期、公开输出结构、正文摘要及保存时的状态版本检查仍执行。
开发输出的内部收据为 `status=disabled`，区别于实际审查通过的 `status=allow`；
历史请求接口通过 `security_review_status` 返回实际状态。重新启用不会把旧开发结果改成已审查。
生产配置位于 `/etc/ai-interviewer/app.env`；修改前备份，修改后重启应用服务。
