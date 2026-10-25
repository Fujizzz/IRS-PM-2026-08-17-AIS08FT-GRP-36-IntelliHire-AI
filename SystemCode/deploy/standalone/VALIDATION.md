# 一键部署验证记录

验证日期：2026-10-21。仅使用独立测试实例和合成数据，没有操作维护者现有生产服务器。

## 已完成

- Windows Python 3.12.7：部署相关测试 **40 项通过**，含本次新增的 22 项配置、权限边界、
  失败阶段、原始备份字节、环境污染和本机/服务器设置测试。Docker 调用在这些单元测试中
  使用替身；不将它们视为真实 Docker/模型服务的证明。
- Ruff 对本次新增 Python 实现及测试检查通过；`git diff --check` 通过。
- 使用项目已有 `backend/tools/check_docs.py` 的检查函数验证本次 4 个 Python 文件，
  35 个声明的注释与索引结构检查通过；另人工核对输入、状态、异常和副作用说明。
- Windows 实际创建私有目录 ACL 和私有配置文件，通过读写验证。
- Ubuntu 24.04（WSL2）、独立 Docker Engine 28.5.1、Compose 2.39.4：实际构建镜像，
  Python 3.12.14、原 Linux 锁定依赖安装和 `pip check` 通过，播放器构建/导出契约通过。
- 用 `python3 deploy.py local --state-dir …` 实际完成配置检查、PostgreSQL/Redis 启动、
  数据备份、全新数据库迁移和六个服务启动。测试端口为 18765，项目名为
  `intellihire-test`，模型 key 为明确无效的合成值；探针没有发起模型请求。
- PDF worker 真实 bubblewrap 运行：宿主目录不可见、密钥环境未继承、socket/fork 被拒绝、
  内存限制生效，合成单页 PDF 的文本提取和 PNG 渲染通过。未修改原解析算法或资源限额。
- 真实 Caddy → ASGI 路径：匿名 API 返回 401、临时账号的受保护页面/健康 API/播放器
  可访问，带 Session 和 Origin 的 WebSocket 返回 hello。探针账号/session 已清理。
- 真实 Celery 收到一次无模型 probe，Redis 返回成功标记并清理；评分 worker 实际启动
  并持续运行。没有提交需要真实模型的评分任务。
- 重复执行部署命令后，数据库中的合成标记保持不变；部署生成的 PostgreSQL dump 已
  实际恢复至独立临时数据库，并读回同一标记，随后删除该临时恢复数据库。
- 应用数据库角色的 superuser、createdb、createrole 均为 false；镜像中不含开发 `.env`、
  `server_info.txt` 或 UE 目录。
- `status`、`stop` 实际执行成功；停止后数据库 volume 仍存在。测试容器已停止。
- 服务器 Compose 配置解析通过：仅发布 80/443，数据库/Redis/8765 未映射至宿主；
  生成的服务器 Caddyfile 经 Caddy 2.10.2 实际 `validate` 通过。

## 验证边界

- 没有使用真实模型密钥，未验证供应商权限、配额、计费接口、实际面试评分或麦克风体验。
- 未在真实公网域名申请证书；服务器端 ACME 签发、DNS/防火墙和公网 HTTPS 需在目标环境验收。
- 没有实测 Docker Desktop 或 macOS；Windows CLI/权限逻辑与 Linux 容器分别验证。
- 项目根目录没有 `tools/check_docs.py`，因此按项目既有约定在 `SystemCode/backend`
  执行 `python tools/check_docs.py`。全量检查返回 **433 项已有问题**，分布在未修改的
  源码/测试中；本次新增 `config/standalone.py` 无问题。未修改检查器、降低规则或声称
  全量注释检查通过。
- 未创建 Git commit、未推送；没有验证“同一提交内原子更新”。本次实现、注释、索引、
  测试及部署说明均保留在同一工作区改动中，原有未提交用户文档改动保留。
