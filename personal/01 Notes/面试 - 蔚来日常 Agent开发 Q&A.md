---
title: 蔚来日常实习 - Agent开发面试 Q&A
created: 2026-05-05
updated: 2026-05-05
type: note
tags:
  - "#面试"
  - "#Agent"
  - "#RAG"
  - "#系统设计"
  - "#编程语言"
status: developing
source:
  - "[[01 Sources/面试 - 蔚来日常 agent开发 题目]]"
related:
  - "[[LLM Wiki 知识库构建法]]"
---

# 蔚来日常实习 - Agent开发面试 Q&A

## 核心摘要

- 面试岗位：蔚来日常实习 Agent 开发
- 覆盖领域：RAG 与 Code Agent、Agent 架构与上下文管理、LLM API 与微调、并发与数据库、Java/Python/C++ 语言特性
- 共 19 题，以 Agent 系统设计为主线，穿插工程基础考察

---

## 一、Agent / RAG 架构

### Q1: RAG的作用，具体在code agent中RAG的作用

**RAG（Retrieval-Augmented Generation）核心作用：**

1. **知识外挂**：LLM 训练数据有截止日期，RAG 从外部知识库检索最新信息，注入 prompt 中，让模型生成基于实时数据的回答。
2. **减少幻觉**：通过提供事实性参考文本，约束模型输出，降低编造概率。
3. **领域适配**：无需微调即可让通用模型回答垂直领域问题。

**在 Code Agent 中的具体作用：**

- **代码库检索**：Agent 需要理解项目上下文时，RAG 从代码仓库中检索相关文件、函数签名、API 文档，作为生成代码的依据。
- **依赖与版本查询**：检索最新库版本、API 变更、issue 讨论，避免生成过时代码。
- **错误诊断**：检索相似错误日志、stackOverflow/内部 wiki 的解决方案。
- **上下文补充**：当对话历史不能覆盖所需信息时，RAG 按需拉取相关代码片段或文档，弥补上下文窗口限制。

---

### Q2: Agent怎么获取的上下文

Agent 获取上下文的典型层次：

1. **系统提示（System Prompt）**：预定义的 Agent 角色、行为规则、输出格式约束。
2. **对话历史（Conversation History）**：用户与 Agent 的多轮对话消息，通常以 `[{role, content}]` 列表管理。
3. **工具调用结果（Tool Outputs）**：Agent 调用外部工具（搜索、代码执行、文件读写）的返回内容，追加到上下文。
4. **RAG 检索结果**：从向量库/知识库检索到的相关文档片段。
5. **环境状态（Environment State）**：当前工作目录、打开的文件、git 状态、终端输出等 IDE/OS 环境信息。
6. **记忆系统（Memory）**：长期记忆（跨会话持久化的用户偏好、项目信息）和短期记忆（当前会话的关键信息摘要）。

获取方式：上下文管理器按优先级和 token 预算，将以上信息拼装为 `messages` 数组发送给 LLM。

---

### Q3: 这个Agent作为插件来使用通讯协议怎么实现的

Agent 作为 IDE 插件时，通讯协议通常采用：

1. **LSP（Language Server Protocol）**：若 Agent 以语言服务器形式存在，通过 JSON-RPC 2.0 与 IDE 通信，支持代码补全、诊断、跳转等。
2. **HTTP/WebSocket**：
   - Agent 后端作为本地 HTTP 服务（如 `localhost:port`），IDE 插件通过 REST API 发送请求。
   - WebSocket 用于长连接场景（流式输出、实时同步）。
3. **stdio / Unix Socket / Named Pipe**：进程间通信，IDE 启动 Agent 子进程，通过标准输入输出或管道传递 JSON 消息。
4. **gRPC**：需要高性能流式通信时，用 protobuf 定义服务接口。

典型流程：
```
IDE Plugin → HTTP/WS → Agent Server → LLM API
                ↑                        ↓
           Tool Executor ←──────── Response
```

关键设计点：消息序列化（JSON/Protobuf）、流式传输（SSE/WebSocket）、超时与重连、取消机制（用户中断生成时发送 cancel 信号）。

---

### Q4: 长连接什么时候会关闭子连接

长连接（如 WebSocket、gRPC stream、HTTP/2 长连接）关闭子连接/子流的常见场景：

1. **空闲超时（Idle Timeout）**：超过配置的空闲时间无数据交互，服务端或负载均衡主动关闭。
2. **最大连接时长（Max Connection Duration）**：云服务（如 AWS ALB、Nginx）默认限制单连接最长存活时间（如 60s–3600s）。
3. **流式响应结束**：LLM 生成完成（`finish_reason: stop`），Agent 主动关闭该请求对应的流。
4. **用户取消请求**：前端发送 cancel 信号，Agent 中断 LLM 调用并关闭子连接。
5. **Token/速率限制**：超出速率限制时，服务端返回 429 并可能关闭连接。
6. **异常/心跳丢失**：网络波动导致心跳（ping/pong）超时，双方关闭连接。
7. **资源回收**：Agent 侧内存/文件描述符紧张时，主动清理不活跃的子连接。

**最佳实践**：实现自动重连、指数退避、连接池复用、心跳保活（如 WebSocket ping 间隔 30s）。

---

### Q5: 垂直领域业务Agent，有没有做过微调

（此题回答方向，根据实际情况调整）

**微调 vs RAG 的决策：**

| 场景 | 推荐方案 |
|------|----------|
| 领域知识频繁更新 | RAG（无需重新训练） |
| 需要特定输出格式/风格 | 微调 |
| 数据量少（<1000条） | RAG + few-shot prompt |
| 数据量充足且标注质量高 | 微调 |
| 成本敏感 | RAG（省去训练成本） |

**垂直领域 Agent 微调常见做法：**

- **指令微调（Instruction Tuning）**：用领域 QA 对、代码补全数据微调基座模型，使其遵循特定指令格式。
- **LoRA/QLoRA**：低秩适配，在消费级 GPU 上即可微调 7B–70B 模型。
- **Embedding 微调**：针对领域术语优化检索模型的嵌入表示，提升 RAG 召回率。

**若未做微调**，应说明通过 prompt engineering + RAG + 工具链同样可以达到业务目标，且维护成本更低。

---

### Q6: 现在公开的模型的API是怎么区分用户的

公开模型 API（OpenAI、Anthropic、通义千问、DeepSeek 等）的用户隔离机制：

1. **API Key**：每个用户/组织分配唯一 API Key，所有请求需携带该 Key，服务端据此识别调用方。
2. **Organization / Workspace ID**：企业版支持多组织，同一账号下不同项目用不同 Org ID 隔离。
3. **用户级速率限制（Rate Limit）**：按 API Key 维度限制 RPM（请求/分钟）和 TPM（token/分钟）。
4. **计费隔离**：每个 API Key 独立计费，通过 usage API 按 Key 查询用量。
5. **模型访问权限**：不同 Key 可授予不同模型访问权限（如 GPT-4 仅对特定 Key 开放）。
6. **内容安全策略**：不同用户可有不同内容审核规则。

**多租户 SaaS 场景的额外措施：**

- 数据库层面按 `tenant_id` 隔离用户数据。
- 推理缓存（semantic cache）按用户分片，避免跨用户数据泄露。
- 日志与监控按用户维度聚合。

---

### Q7: 工作流和Agent区别

| 维度 | 工作流（Workflow） | Agent |
|------|-------------------|-------|
| **控制模式** | 预定义 DAG / 状态机，固定执行路径 | LLM 自主决策下一步动作 |
| **灵活性** | 低——分支逻辑需事先编码 | 高——运行时动态选择工具和路径 |
| **可预测性** | 高——每一步有确定性输入输出 | 低——LLM 决策具有随机性 |
| **适用场景** | 数据管道、审批流程、CI/CD | 开放任务：代码生成、研究、对话 |
| **错误处理** | 编码好的 fallback 和重试 | 依赖 LLM 自我纠错或人工介入 |
| **工具使用** | 固定工具链，按序调用 | 按需选择工具，工具集可动态扩展 |
| **状态管理** | 显式状态机，状态转换明确 | 隐式状态存在于对话上下文中 |

**本质区别**：工作流是"人定义流程，机器执行"；Agent 是"人定义目标和工具，LLM 自主规划执行路径"。

**混合模式**：实际系统中常见两者结合——外层工作流编排多个 Agent，Agent 在工作流节点内部自主执行。

---

### Q8: 上下文存储消息怎么管理的

上下文消息管理策略：

**1. 消息结构：**
```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "...", "tool_calls": [...]},
  {"role": "tool", "tool_call_id": "...", "content": "..."}
]
```

**2. 存储方案：**
- **会话级**：内存中维护 `List<Message>`，会话结束即释放。
- **持久化**：存入数据库（PostgreSQL/Redis），支持跨会话恢复。
- **关键消息标记**：重要决策点、用户反馈标记为不可截断。

**3. 上下文窗口管理：**
- **滑动窗口**：保留最近 N 轮对话，旧消息丢弃。
- **摘要压缩**：对早期对话生成摘要，替换原文以节省 token。
- **分层存储**：热点消息在内存，冷消息在磁盘/数据库，按需加载。
- **重要性评分**：对每条消息打分，截断时优先保留高分消息。

**4. 多 Agent / 子任务场景**：每个子任务可继承父上下文或从空白上下文启动，结果回传父 Agent。

---

### Q9: 怎么依赖LLM自动处理上下文

LLM 自动管理上下文的策略：

1. **自动摘要（Auto-Summarization）**：当上下文接近 token 限制时，让 LLM 对前半段对话生成结构化摘要，替换原始消息。
2. **关键信息提取**：LLM 从对话中提取实体、决策、约束、待办事项，存为结构化记忆，后续只传记忆而不传原文。
3. **动态剪枝（Context Pruning）**：让 LLM 评估每条消息与当前问题的相关性，删除不相关消息。
4. **分层记忆架构**：
   - **Working Memory**：当前任务上下文（全量保留）。
   - **Episodic Memory**：历史对话摘要（按需检索）。
   - **Semantic Memory**：持久化知识（RAG 检索）。
5. **反思机制（Reflexion）**：LLM 在执行后自我评估，将经验教训写入长期记忆，供后续任务调用。
6. **上下文预算管理**：预设 token 分配（系统提示 20% + 工具结果 30% + 历史 30% + 输出预留 20%），超预算时自动触发压缩。

---

### Q10: Token溢出怎么办

Token 溢出（输入+输出超过模型上下文窗口限制）的处理策略：

**预防层面：**
1. **上下文预算**：预留 20–30% 给输出，输入超预算时提前截断/压缩。
2. **消息截断策略**：优先删除最早的、工具调用返回的大段内容、已被摘要覆盖的旧消息。
3. **分块处理**：大文件/长文档分块，每次只注入与当前任务相关的块。

**发生时处理：**
1. **截断旧消息**：从最早的消息开始删除，直到 token 数满足要求。
2. **压缩替代删除**：用 LLM 摘要替代原文，保留信息密度。
3. **拆分任务**：将大任务拆为子任务，每个子任务使用独立的精简上下文。
4. **错误恢复**：若 LLM 返回 `context_length_exceeded` 错误，自动触发上下文压缩后重试。
5. **降级策略**：自动切换到更大上下文窗口的模型（如 4K → 128K），或使用长上下文模型。

**工程实践**：使用 tokenizer（如 `tiktoken`）精确计算 token 数，而非按字符估算。

---

### Q11: Agent并发怎么控制

Agent 并发控制的多层策略：

**1. 请求级并发限制：**
- **信号量/令牌桶**：限制同时处理的请求数（如 `asyncio.Semaphore(N)`）。
- **队列缓冲**：超限请求进入优先级队列，FIFO 或按优先级出队。

**2. LLM API 调用并发：**
- 遵守 LLM 提供商的速率限制（RPM/TPM），本地令牌桶对齐。
- 多 Key 轮询（Key Pool），每个 Key 独立计速。
- 请求合并：多个相似请求共享一次 LLM 调用结果。

**3. 工具执行并发：**
- 独立工具（读文件、搜索）可并发执行，`Promise.all` / `asyncio.gather`。
- 有依赖的工具串行执行（前一个输出是后一个的输入）。
- 并发写操作需加锁或使用乐观锁。

**4. 数据库连接池：**
- 连接池大小限制（如 SQLAlchemy `pool_size=20, max_overflow=10`）。
- 读操作走只读副本，写操作走主库。

**5. 会话级隔离：**
- 每个用户会话独立上下文，互不干扰。
- 分布式场景用 Redis 分布式锁协调跨实例并发。

---

## 二、数据库与缓存

### Q12: 数据库挂了怎么办

**预防层：**
1. **主从复制 + 自动故障转移**：MySQL（MHA/Orchestrator）、PostgreSQL（Patroni/Repmgr），主库故障时自动提升从库。
2. **读写分离**：写走主库，读走只读副本，主库压力降低，单点故障影响面缩小。
3. **连接池健康检查**：定期探活（`SELECT 1`），剔除不健康节点。

**发生时处理：**
1. **快速失败（Fail-fast）**：设置合理超时（如 3–5s），避免请求堆积。
2. **降级策略**：
   - 读操作降级到缓存（Redis）或过期快照。
   - 写操作暂存到消息队列（Kafka/Redis Stream），等数据库恢复后回放。
   - 非关键功能直接跳过数据库写（如埋点日志）。
3. **重试策略**：指数退避 + 随机抖动（jitter），避免雪崩。
4. **熔断器（Circuit Breaker）**：连续失败 N 次后自动熔断，定期探测恢复。
5. **告警与监控**：数据库宕机实时告警，运维介入恢复。

**Agent 场景特殊处理**：Agent 调用工具时如遇数据库错误，应将错误信息返回 LLM，让 LLM 向用户解释并给出替代方案。

---

### Q13: 缓存和数据库一致性

**常见模式与一致性保证：**

| 策略 | 一致性 | 复杂度 | 适用场景 |
|------|--------|--------|----------|
| Cache-Aside | 最终一致 | 低 | 读多写少 |
| Read-Through / Write-Through | 强一致 | 中 | 缓存层封装数据访问 |
| Write-Behind | 最终一致 | 高 | 写密集型 |
| Write-Around | 最终一致 | 低 | 写入数据不常被立即读取 |
| Double-Delete | 最终一致（概率性） | 低 | 并发不高的场景 |

**Cache-Aside（最常用）流程：**
```
Read:  查缓存 → hit则返回 / miss则查DB → 写缓存 → 返回
Write: 更新DB → 删除缓存（或更新缓存）
```

**为什么写操作是删缓存而非更新？**
- 更新缓存可能写入无人读取的数据，浪费内存。
- 并发写时，更新缓存顺序难以保证，可能写入旧值。
- 删除缓存后，下次读会自然加载最新值。

**延时双删（Double-Delete）**：更新 DB 前删一次缓存，更新 DB 后延迟（如 500ms）再删一次，降低并发读写导致脏数据的概率。

**最终一致方案**：监听 MySQL binlog（Canal/Debezium）→ 异步更新/删除缓存。

**强一致方案**：分布式事务（2PC/Seata）或使用支持事务的缓存（Redis + Lua 脚本原子操作）。

---

## 三、Java / Python 语言特性

### Q14: Java的注解和Python的装饰器有什么区别

| 维度 | Java 注解（Annotation） | Python 装饰器（Decorator） |
|------|------------------------|---------------------------|
| **本质** | 元数据标记，不直接改变代码行为 | 高阶函数/类，在运行时包装并修改目标 |
| **执行时机** | 编译时（SOURCE/CLASS 保留）或运行时通过反射读取 | 定义时执行包装逻辑，调用时执行 wrapped 逻辑 |
| **行为改变方式** | 需要注解处理器（APT）或运行时反射框架解释 | 直接替换/包装原函数，可修改输入输出 |
| **参数传递** | 通过元素值对（`@Annotation(key=val)`） | 通过闭包传递参数（装饰器工厂返回装饰器） |
| **堆叠** | 可以，顺序通常无关（由处理器决定） | 可以，从下往上依次包装（就近原则） |
| **典型用途** | 标记（`@Override`）、DI（`@Autowired`）、ORM（`@Entity`） | 日志、计时、权限校验、缓存、路由注册 |
| **运行时开销** | 反射读取有开销，但注解本身无 | 每次调用都经过装饰器层，有堆叠开销 |

**一句话总结**：Java 注解是"贴在代码上的标签"，需外部解释器处理；Python 装饰器是"包在函数外的壳"，直接参与执行。

---

### Q15: Java怎么自己封装注解

**步骤：**

**1. 定义注解：**
```java
@Retention(RetentionPolicy.RUNTIME)  // 运行时可通过反射读取
@Target(ElementType.METHOD)           // 作用在方法上
public @interface LogExecution {
    String value() default "";         // 注解参数
    boolean logArgs() default false;   // 是否打印参数
}
```

**2. 编写注解处理器（通过反射/AOP）：**

方式一——反射：
```java
Method method = obj.getClass().getMethod("someMethod");
if (method.isAnnotationPresent(LogExecution.class)) {
    LogExecution ann = method.getAnnotation(LogExecution.class);
    System.out.println(ann.value());
    method.invoke(obj);
}
```

方式二——Spring AOP（最常用）：
```java
@Aspect
@Component
public class LogExecutionAspect {
    @Around("@annotation(logExecution)")
    public Object around(ProceedingJoinPoint jp, LogExecution logExecution) {
        System.out.println("Before: " + logExecution.value());
        Object result = jp.proceed();
        System.out.println("After: " + logExecution.value());
        return result;
    }
}
```

**3. 使用注解：**
```java
@LogExecution(value = "getUserById", logArgs = true)
public User getUserById(Long id) { ... }
```

**核心要素**：`@Retention`（存活到何时） + `@Target`（能贴在哪） + 处理器（AOP/APT/反射）三者缺一不可。

---

### Q16: Java的注解用Python装饰器怎么实现

将 Java 注解的功能用 Python 装饰器等价实现：

**Java 注解：**
```java
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface LogExecution {
    String value() default "";
    boolean logArgs() default false;
}
```

**Python 等价实现（装饰器工厂）：**
```python
import functools
import time

def log_execution(value="", log_args=False):
    """等价于 Java @LogExecution 注解"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if log_args:
                print(f"[LOG] {value} | args={args}, kwargs={kwargs}")
            else:
                print(f"[LOG] {value} | calling {func.__name__}")
            start = time.time()
            result = func(*args, **kwargs)
            print(f"[LOG] {value} | done in {time.time() - start:.2f}s")
            return result
        return wrapper
    return decorator

# 使用
@log_execution(value="get_user_by_id", log_args=True)
def get_user_by_id(user_id: int):
    return {"id": user_id, "name": "Alice"}
```

**对应关系：**

| Java 概念 | Python 等价 |
|-----------|------------|
| 注解定义 | 装饰器函数（或装饰器工厂） |
| 注解参数 | 装饰器工厂的参数 |
| AOP 切面 | 装饰器 wrapper 内的 before/after 逻辑 |
| 反射读取注解 | `func.__wrapped__` / `func.__name__` 等内省 |
| `@Retention(RUNTIME)` | 默认行为，Python 装饰器天然运行时 |
| `@Target(METHOD)` | 手动检查 `callable(func)` |

**本质差异**：Java 需要注解 + AOP 切面两步，Python 装饰器一步到位，因为装饰器本身就是 AOP 的实现。

---

## 四、C++ 基础

### Q17: 会不会C++，讲一下智能指针

C++ 智能指针是 RAII（资源获取即初始化）思想在指针管理上的应用，自动管理堆内存生命周期，避免内存泄漏。

**四种智能指针：**

| 类型 | 所有权 | 典型用途 |
|------|--------|----------|
| `std::unique_ptr` | 独占 | 工厂函数返回值、PIMPL、容器中存对象 |
| `std::shared_ptr` | 共享（引用计数） | 多个对象共享同一资源、观察者模式 |
| `std::weak_ptr` | 不拥有（弱引用） | 打破循环引用、缓存、观察者 |
| `std::auto_ptr` | 已废弃（C++11 起） | 不要使用 |

**1. `unique_ptr`（独占所有权）：**
```cpp
auto p = std::make_unique<Foo>(args);  // C++14 起推荐
// 不可拷贝，只能移动
auto p2 = std::move(p);  // p 变为 nullptr，所有权转移
```
- 零开销（大小等同裸指针）。
- 自定义删除器：`std::unique_ptr<FILE, decltype(&fclose)> p(fopen(...), fclose);`

**2. `shared_ptr`（共享所有权）：**
```cpp
auto p1 = std::make_shared<Foo>();
auto p2 = p1;  // 引用计数 +1
// 引用计数归零时自动释放
```
- 额外开销：控制块（引用计数 + 弱引用计数 + 删除器），大小是裸指针的 2 倍。
- 注意循环引用：A 持有 B 的 `shared_ptr`，B 持有 A 的 `shared_ptr`，永远不会释放。

**3. `weak_ptr`（打破循环引用）：**
```cpp
std::weak_ptr<Foo> wp = shared_ptr_instance;
if (auto sp = wp.lock()) {  // 尝试提升为 shared_ptr
    sp->doSomething();
}
```
- 不增加引用计数。
- 可检测对象是否存活（`expired()`）。
- 典型场景：`shared_ptr<Parent>` ↔ `weak_ptr<Child>` 打破父子循环引用。

**核心原则**：能用 `unique_ptr` 就不用 `shared_ptr`，更轻量且语义清晰。

---

### Q18: C++有遇到功能异常退出的场景吗比如coredump

**常见 Coredump 场景及排查方法：**

**1. 空指针/野指针解引用：**
```cpp
Foo* p = nullptr;
p->doSomething();  // SIGSEGV
```
- 排查：GDB 回溯 `bt full`，查看崩溃地址是否接近 0x0。

**2. 双重释放（Double Free）：**
```cpp
delete p;
delete p;  // heap corruption → SIGABRT
```
- 排查：AddressSanitizer（`-fsanitize=address`）精确定位。

**3. 栈溢出（Stack Overflow）：**
```cpp
void recurse() { recurse(); }  // 无限递归
int arr[10000000];              // 局部大数组
```
- 排查：`ulimit -c unlimited` + GDB，观察栈帧深度。

**4. 缓冲区溢出（Buffer Overflow）：**
```cpp
char buf[10];
strcpy(buf, "this string is way too long");  // 栈破坏
```
- 排查：`-fstack-protector-all` 编译选项，或使用 `std::string`。

**5. 使用已释放的内存（Use-After-Free）：**
```cpp
auto p = std::make_shared<Foo>();
// ... p 被释放
p->doSomething();  // UAF
```
- 排查：AddressSanitizer、Valgrind。

**6. 多线程数据竞争：**
```cpp
// 线程 A 写 vector，线程 B 同时读 → segfault
```
- 排查：ThreadSanitizer（`-fsanitize=thread`），或加锁/用原子操作。

**排查工具链：**
- **GDB**：`gdb ./a.out core` → `bt` / `info registers` / `frame N` / `print var`
- **AddressSanitizer**：编译加 `-fsanitize=address -g`，内存错误秒定位。
- **Valgrind**：`valgrind --leak-check=full ./a.out`，检测内存泄漏和非法访问。
- **静态分析**：`clang-tidy`、`cppcheck`。

**预防措施**：智能指针替代裸指针、`std::vector::at()` 替代 `operator[]`、`-Wall -Wextra -Werror` 编译选项、CI 中集成 sanitizer。

---

## 五、综合

### Q19: 研究生做的科研项目讲一下

（此题需根据个人实际经历回答，以下为回答框架建议）

**推荐结构（STAR 原则 + 技术深度）：**

```
项目背景：XX 领域存在 XX 问题，现有方法在 XX 方面不足。

我的工作：
1. 提出 XX 方法/模型，核心创新是 XX（如改进注意力机制/提出新的损失函数/设计新的数据增强策略）。
2. 在 XX 数据集上验证，相比 baseline 提升 XX%（具体指标）。
3. 发表论文 / 投稿至 XX 会议/期刊。

技术细节（准备被深问）：
- 模型结构、训练技巧、评估指标。
- 遇到的最大挑战及解决方案。
- 为什么选择这个方法而不是其他。
```

**面试官想听到的**：你做的是什么（一句话讲清）、你具体做了什么（不是导师做什么）、你遇到了什么问题以及怎么解决的（工程能力）、实验结果与反思（科研素养）。

---

## 知识连接

- [[LLM Wiki 知识库构建法]] — 本知识库的方法论基础
- [[03 Concepts/Agent 架构设计]] — 待创建：Agent 系统架构概念页
- [[03 Concepts/RAG 检索增强生成]] — 待创建：RAG 概念页
- [[05 MOC/Agent 开发 MOC]] — 待创建：Agent 相关笔记索引

## 后续问题

- [ ] 创建 Agent 架构设计概念页
- [ ] 创建 RAG 概念页
- [ ] 若用户有实际微调经验，补充 Q5 具体细节
- [ ] 若用户有具体科研项目，补充 Q19 个性化内容
