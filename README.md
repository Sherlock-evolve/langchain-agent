# Workspace Agent

一个以安全边界、事务一致性和可测试性为核心的中文工作区 Agent。
项目使用 LangChain 的消息与工具协议，但保留自研 Agent 循环，便于明确控制
流式输出、工具审批、上下文预算、会话持久化、审计和知识检索行为。

## 能力概览

- 流式模型回答和多轮工具调用；
- 受工作区边界保护的文件列出、读取、搜索和原子写入；
- 写入前 diff 审批、审批后文件冲突检测和可恢复审批；
- Agent 循环、工具次数、工具结果和上下文预算；
- 事务式消息提交、长期摘要记忆和命名会话；
- 模型/工具耗时、事件信封和脱敏 JSONL 审计；
- 可选的 `docs/` 语义检索、增量 Embedding 缓存和引用校验；
- `observe` 与 `require-valid` 两种引用策略。
- 强制工具风险注册、协作式取消、有限重试、退避、熔断和总耗时预算；
- 异步模型/工具与 `astream_turn()` 双向事件流；
- FastAPI + SSE、独立审批 API、断连取消和请求幂等；
- API Key 身份认证、用户/工作区隔离和 AES-256-GCM 静态加密；
- 请求、token、并发、成本限额和 Prometheus 文本指标；
- 审计日志轮转、保留、按会话删除和发布质量门槛。

## 运行链路

```text
CLI / FastAPI + SSE
  └─ PersistentSession（每会话串行）
       ├─ WorkspaceAgent（stream_turn / astream_turn）
       │    ├─ Sync / Async Chat Model
       │    └─ ToolExecutionMiddleware
       │         └─ Tenant Workspace / Knowledge Tools
       ├─ Encrypted Session Store
       └─ Rotating JSONL Audit Log
```

`contracts.py` 定义跨界事件和审批协议；CLI 只负责输入与事件渲染。
只有模型成功生成最终回答后，本轮消息和长期摘要才会提交。
Web 服务把同一 `EventEnvelope` 编码为 SSE，不另建一套 Agent 状态机。

## 环境要求

- Python 3.10 或更高版本；
- OpenAI 兼容的聊天模型服务；
- 启用知识库时，需要兼容 OpenAI Embeddings 的服务。

创建虚拟环境并安装锁定依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

开发模式也可以从 `pyproject.toml` 安装：

```bash
python -m pip install -e ".[dev]"
```

## 配置

在项目根目录创建不纳入 Git 的 `.env`：

```dotenv
ZHIPU_MODEL=your-chat-model
ZHIPU_API_KEY=your-api-key
ZHIPU_BASE_URL=https://your-provider.example/v1

# 仅在启用知识库时需要
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://your-provider.example/v1
```

`EMBEDDING_API_KEY` 和 `EMBEDDING_BASE_URL` 未配置时，会分别回退到聊天模型的
`ZHIPU_API_KEY` 和 `ZHIPU_BASE_URL`。启用知识库可能把文档分块和查询发送给
所配置的外部 Embeddings 服务。

## 使用

普通模式：

```bash
python main.py
python main.py --session project-a
```

启用知识库：

```bash
python main.py --enable-knowledge
python main.py \
  --enable-knowledge \
  --knowledge-directory docs \
  --citation-policy require-valid
```

CLI 命令：

```text
:session              显示当前会话状态
:sessions             列出会话
:switch <session_id>  切换或新建会话
:delete <session_id>  删除非当前会话
:pending              显示等待恢复的审批
:resume               恢复并重新确认审批
:retry                重试保存 dirty 会话
:help                 显示帮助
exit / quit / 退出    退出
```

等待审批时退出不会执行工具。下次打开同名会话后，使用 `:resume` 重新生成预览
并再次确认。批准记录会在副作用执行前清除，因此崩溃恢复遵循至多一次语义：
不会凭旧审批重复执行写入，但极端情况下可能需要用户重新发起未执行的操作。

## Web 服务

Web 入口默认要求认证、加密密钥和租户存储根目录：

```bash
export AGENT_SERVICE_ROOT=/srv/workspace-agent
export AGENT_API_KEYS='{"replace-with-a-long-random-key":"user-a"}'
export AGENT_ENCRYPTION_KEY="$(
  python -c 'from storage_security import SnapshotCipher; print(SnapshotCipher.generate_key())'
)"

workspace-agent-api --host 127.0.0.1 --port 8000
```

生产环境应通过 Secret Manager 注入 API Key 和加密密钥，不要写入仓库或日志。
同一加密密钥必须稳定保存；丢失后无法恢复已有会话和待审批记录。

发起 SSE 对话：

```bash
curl -N \
  -H 'Authorization: Bearer replace-with-a-long-random-key' \
  -H 'Idempotency-Key: request-20260724-001' \
  -H 'Content-Type: application/json' \
  -d '{"message":"请总结当前工作区"}' \
  http://127.0.0.1:8000/v1/workspaces/default/sessions/demo/turns
```

流中出现 `ApprovalRequiredEvent` 后，客户端通过独立请求提交决定：

```bash
curl -X POST \
  -H 'Authorization: Bearer replace-with-a-long-random-key' \
  -H 'Content-Type: application/json' \
  -d '{"approved":true}' \
  http://127.0.0.1:8000/v1/workspaces/default/sessions/demo/approvals/TOOL_CALL_ID
```

主要接口：

- `POST /v1/workspaces/{workspace}/sessions/{session}/turns`：开始 SSE 轮次；
- `POST /v1/workspaces/{workspace}/sessions/{session}/resume`：恢复持久化审批；
- `POST .../approvals/{tool_call_id}`：提交独立审批；
- `GET .../sessions`、`DELETE .../sessions/{session}`：列出或删除自有会话；
- `GET /metrics`：认证后读取低基数指标；
- `GET /healthz`：存活检查。

对话和恢复请求必须携带 `Idempotency-Key`。相同用户、工作区、会话、键和请求
正文会重放完全相同的 SSE；同一键配不同正文返回 `409`。客户端断开会发出
`client_disconnect` 取消，尚未提交的 Agent 事务回滚。

可通过环境变量调整 `AGENT_MAX_REQUEST_BYTES`、
`AGENT_MAX_MESSAGE_CHARACTERS`、`AGENT_MAX_INPUT_TOKENS`、
`AGENT_MAX_OUTPUT_TOKENS`、`AGENT_MAX_CONCURRENT_GLOBAL`、
`AGENT_MAX_CONCURRENT_PER_USER` 和 `AGENT_MAX_COST_USD_PER_TURN`。
通过 `AGENT_INPUT_COST_PER_MILLION_TOKENS` 和
`AGENT_OUTPUT_COST_PER_MILLION_TOKENS` 配置每百万 token 的价格后，服务会累计
估算成本并执行单轮上限。
`AGENT_ENABLE_KNOWLEDGE=true` 可为每个租户工作区独立启用知识索引。

## 测试

```bash
python -m pytest -q
python -m compileall -q .
python -m pytest -q --junitxml=test-results.xml
python quality_gate.py test-results.xml --min-tests 150
```

测试使用确定性脚本模型和受控 Embeddings，默认不访问网络。
真实模型契约通过 GitHub Actions 的手动 `Real model integration` 工作流运行，
本地也可设置 `RUN_REAL_MODEL_TESTS=1` 后执行 `pytest -m real_model`。

## 安全边界

- 文件工具只接受工作区内相对路径，拒绝绝对路径、`..` 和符号链接逃逸；
- `.env`、密钥、会话、审计和知识索引目录不会暴露给模型文件工具；
- 写入使用同目录临时文件、`fsync()` 和 `os.replace()`；
- 审批预览与执行之间使用文件 SHA-256 检测冲突；
- 工具事件和审计记录不会保存工具结果正文，敏感参数会被脱敏；
- 只读工具可配置超时并在取消后停止等待；
- 工具必须在生产注册表中声明 `read_only`、`workspace_write` 或
  `external_side_effect`；
- 只有只读工具或携带有效幂等键的副作用工具允许重试；
- 外部副作用强制携带请求幂等键；
- 长时间工具通过 `invoke_with_cancellation(args, token)` 接收
  `CancellationToken`，并在安全检查点主动检查；
- 写入等非协作式副作用一旦开始会完成原子执行边界；
- 知识索引更新使用有超时的跨进程文件锁；
- 检索内容始终被标记为不可信资料，不能提供系统权限或工具授权。

## 主要目录和状态文件

```text
.agent_sessions/   会话快照和等待恢复的审批，权限 0700/0600
.agent_audit/      每会话 JSONL 审计日志
.knowledge_index/  按模型配置隔离的增量 Embedding 缓存
docs/              默认知识文档目录
tests/             确定性单元和集成测试

$AGENT_SERVICE_ROOT/
  users/{user}/workspaces/{workspace}/
    files/          用户工作区
    sessions/       AES-GCM 加密快照和待审批记录
    audit/          轮转、保留和可删除的审计日志
    knowledge/      租户知识状态保留目录
```

这些运行时目录均被 Git 忽略，也被工作区文件工具屏蔽。

## 开发路线

详细里程碑、已完成能力和后续服务化方向见
[`docs/agent-development-roadmap.md`](docs/agent-development-roadmap.md)。
