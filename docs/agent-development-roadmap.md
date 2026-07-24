# Workspace Agent：进度总结与开发路线图

> 更新日期：2026-07-24
> 当前阶段：M6 执行可靠性、M7 异步 Web 服务和 M8 多用户生产边界已经完成
> 当前测试基线：`159 passed, 1 skipped`

## 1. 项目目标

本项目用于实现一个可复用、可测试、可恢复且安全边界明确的工作区 Agent。
它保留自研 Agent 循环，以便完整理解并控制模型消息、工具调用、人工审批、
状态提交、知识检索和审计行为。

目标能力包括：

- 流式模型对话和多轮工具调用；
- 安全读取、搜索和修改工作区文件；
- 对副作用执行进行人工审批；
- 在异常、取消、保存失败和文件冲突时保持状态一致；
- 支持命名会话、长期摘要、审计和恢复；
- 支持带可验证来源的语义检索；
- 最终提供异步、多用户的服务接口。

## 2. 当前架构

```text
main.py
  └─ PersistentSession
       ├─ WorkspaceAgent
       │    ├─ Chat Model
       │    ├─ ToolExecutionMiddleware
       │    └─ Workspace / Knowledge Tools
       ├─ session_store.py
       └─ audit_log.py

knowledge_runtime.py
  ├─ knowledge_base.py
  ├─ knowledge_index.py
  ├─ knowledge_retriever.py
  ├─ knowledge_tools.py
  └─ knowledge_citation_validator.py
```

主要文件职责：

| 文件 | 职责 |
| --- | --- |
| `contracts.py` | 不可变事件、审批决定和事件信封 |
| `agent.py` | Agent 循环、事务、预算、记忆、引用策略和审批状态 |
| `tool_execution.py` | 工具风险策略、同步执行、只读超时和显式取消 |
| `tools.py` | 受限工作区文件工具和原子写入 |
| `persistent_session.py` | Agent、会话快照和待审批记录的组合层 |
| `session_store.py` | 安全 JSON 存储、原子替换和命名会话 |
| `audit_log.py` | 脱敏 JSONL 审计 |
| `knowledge_base.py` | 确定性文档发现、读取和分块 |
| `knowledge_index.py` | 模型隔离的增量 Embedding 持久缓存 |
| `knowledge_retriever.py` | 内存语义检索和稳定排序 |
| `knowledge_tools.py` | 有预算、带引用的知识检索工具 |
| `citations.py` | 规范引用提取和校验 |
| `rag_evaluation.py` | Hit Rate、MRR 和 Recall 评测 |
| `async_runtime.py` | 同步双向生成器、异步迭代器和阻塞调用桥接 |
| `web_service.py` | FastAPI、SSE、审批、认证和租户会话编排 |
| `production_runtime.py` | 限额、幂等、审批 Broker、指标和故障注入 |
| `storage_security.py` | AES-GCM 静态加密和租户路径 |
| `quality_gate.py` | JUnit 发布基线检查 |

## 3. 已完成能力

### 3.1 Agent 内核

- 模型流式输出和 Tool Calling；
- 默认最多 5 轮 Agent 循环；
- 最多 8 次真实工具执行；
- 默认每轮 12,000 个工具结果字符；
- 重复调用检测和最后一轮强制收尾；
- 模型与工具结果体积限制；
- 模型耗时、首块延迟、token 和工具耗时指标。

### 3.2 事务与上下文

- 每轮在独立 `working_messages` 中运行；
- 只有最终回答成功时才提交；
- 流关闭、异常和取消会回滚未提交消息；
- 同一 Agent 和持久会话均禁止并发轮次；
- 上下文按完整消息协议组裁剪；
- 被移除的完整历史轮次可合并为长期摘要；
- 摘要失败不会阻止当前回答。

### 3.3 安全文件工具

- `list_files`、`read_file`、`search_text` 和 `write_file`；
- 拒绝绝对路径、Windows 根路径、`..` 和符号链接逃逸；
- 屏蔽环境文件、密钥、会话、审计和知识索引目录；
- 文件读写、搜索结果和预览均有大小限制；
- 写入使用同目录临时文件、`fsync()` 和 `os.replace()`；
- 写入审批后使用 SHA-256 再次检查文件状态；
- 外部修改发生时保留外部内容并报告冲突。

### 3.4 工具执行中间件、超时与取消

`ToolExecutionMiddleware` 为工具声明：

- `read_only`；
- `workspace_write`；
- `external_side_effect`。

只读工具可以在独立 daemon worker 中执行，达到截止时间或收到取消信号后，
Agent 停止等待并回滚当前轮次。worker 可能继续完成只读计算，但结果会被丢弃。

写入和外部副作用不会在执行中途被遗弃。取消会阻止尚未开始的调用；一旦进入
副作用执行边界，就让该边界完成。当前同步 Python 工具没有通用的强制终止机制，
未来需要为长时间副作用工具增加显式的协作式取消协议。

### 3.5 审批与恢复

- 审批通过双向事件流完成；
- 决定必须携带匹配的工具调用 ID；
- 缺失、错误类型、错误 ID 和拒绝均 fail-closed；
- preparer 在审批前生成 diff 和带版本检查的执行闭包；
- 等待写入审批时，会把未提交轮次保存为 `.pending.json`；
- 重启后使用 `:resume` 重新调用 preparer、生成最新预览并再次审批；
- 已提交会话快照变化时，旧待审批记录拒绝恢复；
- 批准记录在副作用开始前持久删除，避免崩溃后凭旧审批重复执行。

恢复采用至多一次副作用语义。进程在清除审批记录后、执行工具前崩溃时，操作可能
没有执行，需要用户重新发起；系统不会自动重放一个结果未知的写入。

待审批文件可能包含恢复工具所需的原始写入参数，存储目录权限为 `0700`、文件为
`0600`，并且不会暴露给模型文件工具。生产多用户场景还应增加独立密钥加密。

### 3.6 会话和审计

- 版本化 Agent 快照；
- 会话 ID 校验、原子保存、加载、列出、切换和删除；
- 保存失败进入 dirty 模式，重试不会重放模型或工具；
- 每个事件使用 `session_id + turn_id + sequence` 关联；
- 审计采用字段白名单，不记录 token 文本、工具结果正文或 diff；
- 审计目录和文件使用受限权限并拒绝符号链接。

### 3.7 知识库与引用

- 显式 `--enable-knowledge` 后才构建知识库或创建 Embeddings 客户端；
- 安全扫描 `.md` 和 `.txt` 文档；
- 记录来源、行号、文档 SHA-256 和稳定 `chunk_id`；
- 使用内存向量检索，按分数和元数据稳定排序；
- 返回有限 JSON、内容截断标志和规范 citation；
- 检索正文明确标记为不可信资料；
- 回答引用只允许使用本轮实际检索到的来源；
- `observe` 只记录引用状态；
- `require-valid` 会缓存候选回答，并在检索后引用无效时拒绝显示和提交；
- 提供 Hit Rate、MRR 和 Mean Recall 的确定性评测。

### 3.8 增量持久化知识索引

- Embedding 缓存按模型和服务地址指纹隔离，不保存 API Key；
- 缓存以文档分块内容 SHA-256 为键，只保存哈希和浮点向量；
- 未变化分块直接复用向量；
- 新增或变化分块才调用 `embed_documents()`；
- 删除的分块会在下次成功提交索引时清理；
- 缓存使用原子写入、受限权限、大小限制和严格 JSON 校验；
- 损坏、超限或符号链接缓存会使知识库安全启动失败，不会静默使用。

当前持久化的是增量 Embedding 数据；进程启动时仍会重新构建轻量的内存向量结构。

### 3.9 工程化基线

- `README.md` 提供安装、配置、运行、安全边界和恢复说明；
- `pyproject.toml` 定义项目、依赖、CLI 入口和 pytest 配置；
- `requirements.lock` 固定 Python 3.10 依赖闭包；
- GitHub Actions 在 Python 3.10、3.11 和 3.12 上编译并运行测试；
- 本地测试不依赖真实模型或网络。

### 3.10 M6：执行可靠性深化

- 生产工具注册表严格要求每个工具声明风险级别；
- `CooperativeCancellationTool` 通过
  `invoke_with_cancellation(args, CancellationToken)` 注入取消协议；
- 重试只允许只读调用，或携带有效幂等键的副作用调用；
- 外部副作用无条件要求幂等键；
- 每个工具支持有限尝试、指数退避、单次超时、总耗时预算和熔断恢复探针；
- 熔断器按工具隔离，连续失败达到阈值后 fail-fast；
- 知识索引命名空间使用 `flock` 跨进程互斥和有限等待；
- 所有策略保持不可变并在进入工具边界前完成校验。

### 3.11 M7：异步化和 Web 服务

- `WorkspaceAgent.astream_turn()` 和
  `PersistentSession.astream_turn()` 保留双向 `asend()` 审批协议；
- 异步入口优先使用模型 `astream()` 和工具 `ainvoke()`；
- FastAPI 使用 SSE 输出原始 `EventEnvelope` 类型、序号和 turn ID；
- 审批通过独立 API 与活跃流 rendezvous，不依赖同一 HTTP 请求正文；
- 流取消和客户端断开触发 `client_disconnect`，未提交事务回滚；
- 同一用户/工作区/会话由独立异步锁串行；
- 对话和审批恢复要求 `Idempotency-Key`，支持活跃请求等待和完成结果重放；
- 相同幂等键配不同请求摘要返回冲突，不会启动第二次 Agent 事务。

### 3.12 M8：多用户与生产化

- Bearer API Key 通过常量时间摘要比较映射到用户身份；
- 用户和工作区 ID 经过严格校验，并映射到互不重叠的物理目录；
- 文件工具使用 `ContextVar` 绑定租户根目录，并发线程不会串工作区；
- 会话快照和待审批记录使用 AES-256-GCM、随机 nonce 和用途绑定 AAD；
- 服务限制请求字节、消息字符、估算输入 token、模型输出 token、并发和成本；
- 审计日志支持大小轮转、保留期清理和按会话删除；
- 指标覆盖轮次/模型/工具延迟、token、估算成本、失败、取消和引用失败；
- 故障注入点可在测试或预发布环境显式启用；
- 真实模型契约测试由手动 GitHub Actions 工作流运行；
- CI 解析 JUnit，并要求零失败且至少 150 个确定性测试实际执行。

## 4. 当前质量基线

```text
159 passed, 1 skipped
```

测试覆盖：

- 模型流式协议、工具循环和预算；
- 事务提交、取消和锁释放；
- 上下文裁剪和长期摘要；
- 参数脱敏、审批和写入冲突；
- 模型/工具指标与审计白名单；
- 快照、多会话和 dirty 恢复；
- 待审批轮次跨进程恢复；
- 工具超时和显式取消；
- 文档加载、向量检索和结果预算；
- 增量 Embedding 复用、更新和损坏缓存；
- 引用观测、强制门禁和确定性 RAG 评测。
- 风险强制注册、协作式取消、重试/退避/熔断/总预算和幂等键；
- 知识索引跨进程锁；
- 异步模型、异步工具、审批 `asend()` 和断连取消；
- SSE、独立审批 API、每会话串行和请求重放；
- 用户/工作区隔离、静态加密、限额、监控和日志生命周期；
- 故障注入、发布质量门槛和可选真实模型契约。

## 5. 当前限制

- 任意 Python 线程仍不能被安全强制终止；非协作式只读任务超时后会丢弃结果；
- HTTP 请求幂等缓存、审批 Broker 和指标当前是单进程内存实现；
- Web 入口固定使用单 Uvicorn worker；水平扩展需要共享幂等仓库、Broker 和分布式锁；
- 当前认证是部署方配置的 API Key，尚未接入 OIDC、密钥轮转或组织 RBAC；
- AES-GCM 使用单个部署密钥，尚未增加 KMS envelope key 与在线轮换；
- 成本依赖提供商 token usage 和部署方价格配置，缺失 usage 时只能执行 token 上限；
- RAG 评测侧重检索，还缺少答案正确性和引用覆盖率基准；
- 尚未迁移到 LangGraph。

## 6. 后续路线

### M6：执行可靠性深化（已完成）

- 为工具注册强制声明风险级别；
- 为长时间工具定义协作式 `CancellationToken` 注入协议；
- 只对只读或有幂等键的工具进行有限重试；
- 增加退避、熔断和总耗时预算；
- 为外部副作用引入请求幂等键；
- 为知识索引增加跨进程文件锁。

### M7：异步化和 Web 服务（已完成）

- 实现 `astream_turn()`；
- 支持异步模型和异步工具；
- 使用 FastAPI + SSE 输出 `EventEnvelope`；
- 使用独立 API 提交审批；
- 客户端断开时触发 `client_disconnect` 取消；
- 每会话保持串行事务并支持请求幂等。

### M8：多用户与生产化（已完成）

- 身份认证和会话所有权；
- 工作区、知识库、审批和审计的用户隔离；
- 待审批记录和会话快照静态加密；
- 请求大小、token、并发和成本限额；
- 日志轮转、保留和删除策略；
- 延迟、成本、失败率和引用失败率监控；
- 故障注入、真实模型集成测试和发布门槛。

### M9：LangGraph 等价迁移

- 让当前 `WorkspaceAgent` 与 LangGraph 实现共享 `contracts.py`；
- 将模型、工具、审批和提交建模为节点；
- 使用 interrupt/resume 替代双向生成器审批；
- 使用 checkpointer 替代自定义未提交状态；
- 两套实现运行相同契约测试；
- 行为完全一致后再逐步切换生产入口。

## 7. 推荐执行顺序

```text
共享幂等仓库、审批 Broker 和分布式会话锁
    ↓
OIDC / RBAC 与 KMS 密钥轮换
    ↓
答案正确性和引用覆盖率评测
    ↓
LangGraph 等价实现
```

## 8. 常用命令

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock

python main.py
python main.py --session demo
python main.py --enable-knowledge
python main.py --enable-knowledge --citation-policy require-valid

python -m pytest -q
python -m compileall -q .
python -m pytest -q --junitxml=test-results.xml
python quality_gate.py test-results.xml --min-tests 150
```

## 9. 当前结论

项目已经完成安全 Agent 内核、可靠持久化、RAG、M6 执行可靠性、M7 异步服务
和 M8 单进程多用户生产边界。下一阶段的主线是把幂等、审批和会话协调迁移到
共享基础设施，接入 OIDC/KMS，再进行 LangGraph 等价迁移；整个过程继续保持
契约测试、事务回滚和 fail-closed 原则。
