# 🧬 KiraAI FT  — 框架基建分支

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-brightgreen" alt="python">
</p>

> 在原版 KiraAI 之上加一层**框架级基建** 多 key 自动 fallback、可编排 system prompt、对话历史指定深度注入、异步资源预取、消息发送前钩子链全部开箱即用。
>
> **追加式改造**：所有特性默认关闭或与原行为一致，老配置零迁移继续工作。

---

## ✨ 特性一览

- 🎯 **模型组**：把多个 model entry 配成一个 group，按优先级 fallback；429 / 5xx / 超时 / 响应非法自动切换 + 分类冷却，多个白嫖 key 终于能透明轮换
- 🧱 **可编排 system prompt**：原版的"一整块字符串"切成可命名、可排序、可开关的 PromptBlock；插件按 depth 排序拼接，支持运行时条件
- 📍 **in-chat 深度注入**：让指定内容**作为消息插入到对话历史的指定深度**（不是 system 里的一段文本），适合世界书 / 旁白 / 临时语境
- 📝 **Jinja2 模板**：内置 11 个 j2 模板从 Python 字符串迁出来，换 prompt 协议只动模板文件
- ⚡ **Hints 异步流水线**：把"轮 N 吐 key → 后台准备 → 轮 N+1 注入"模式从 sticker 单点提到框架级，资源准备不阻塞用户回复
- 🎚️ **三种触发频率**：每次 / 间隔 N 次 / 概率触发，给"不想每轮都注入"的场景兜底
- 👀 **跨插件 hint 观察**：handler 产出 hint 时 fire-and-forget 触发观察者事件，emotion / 日志 / 调试插件能"听"到别人的产出
- 🚪 **输出钩子链**：LLM 输出 → XML 解析 → 实际发送 之间开放 OutputCtx 钩子链，可改消息 / 设延迟 / 拆分 / 拦截（已读不回）
- 🔄 **回迁旧增强**：之前侵入式改的 v1/chat 出图、原生多模态（图/音/视频）、`desc_video` 一并整合到新框架结构

---

## 📐 架构总览

```
用户消息 → MessageProcessor.handle_im_batch_message
            │
            ├─ 收集所有 PromptBlock（框架内置 + 插件静态 + 当轮动态）
            │     │
            │     └─ Assembler 过滤 / 排序 / 求值 → AssembledPrompt
            │              │
            │              └─ Translator (Jinja2) → 拼成 system 文本
            │
            ├─ 上一轮预备的 hints PromptBlock 自动加入
            │
            ├─ ModelGroup.call(request)  ← 这里才真正调 LLM
            │     │
            │     ├─ 按 priority 尝试 entry 1
            │     ├─ 429? → cooldown 5min, 切 entry 2
            │     └─ 5xx? → cooldown 30min, 切 entry 2
            │
            ├─ in-chat 注入：按 inject_depth 倒序 insert 进 messages
            │
            ├─ 多步 agent loop（每步都走 ModelGroup）
            │
            └─ send_xml_messages
                  │
                  ├─ XML 解析后 → AFTER_LLM_RESPONSE_PARSE
                  │       │
                  │       └─ HintsPipeline 解析 <hints_key>
                  │              │
                  │              ├─ FrequencyPolicy gate
                  │              │     ├─ every / interval / random
                  │              │     └─ skip 也触发观察者事件
                  │              │
                  │              └─ 调度 prepare() → 后台异步
                  │                     │
                  │                     └─ 完成 → 触发 ON_HINTS_PRODUCED
                  │
                  ├─ ON_OUTPUT_PIPELINE 钩子链 ← OutputCtx 在这里被传递
                  │       ├─ 修改 chains / delays / intercepted
                  │       └─ DefaultDelayHook (SYS_LOW) 填默认延迟
                  │
                  └─ adapter.send_*  → 用户收到消息
```

### 核心组件

| 组件 | 职责 |
|---|---|
| **ModelGroupManager** | 按 group_id 路由 LLM 调用，按错误类型分类冷却，AgentExecutor 多步循环每步生效 |
| **PromptBlockRegistry** | 所有插件注册的静态 PromptBlock 在此索引，按 plugin_id 隔离，热卸载干净 |
| **PromptAssembler** | 三个来源（框架 / 插件静态 / 当轮动态）合一，filter → sort → evaluate → 分桶（system / chat_injection） |
| **PromptTranslator** | Assembler 中间结构 → Jinja2 → 最终 LLM 文本，11 个内置 j2 模板可热替换 |
| **HintsPipeline** | `<hints_key>` 解析 + 后台 prepare 调度 + TTL 老化 + 200ms backpressure + 跨 session 隔离 |
| **FrequencyPolicy** | 每个 handler 一份策略对象，控制 every / interval / random 触发频率 |
| **OutputCtx** | 输出钩子链的唯一可变状态容器，含 chains / delays / intercepted / budget_ms |
| **DefaultDelayHook** | SYS_LOW 优先级内置钩子，行为与原版 `random.uniform(min,max)` 一致 |

### 数据流分层

| 层 | 内容 | 何时介入 |
|---|---|---|
| **配置层** | model_groups.json / translator.json / 插件 schema.json | 启动时一次加载，热 reload |
| **注册层** | PromptBlock / HintsHandler / OutputHook / Tag / Tool | 插件 init 时注册，terminate 时清除 |
| **调度层** | EventType 钩子链 + Pipeline + ModelGroup | 每条消息流转都跑 |
| **存储层** | data/config/*.json + 插件自有 SQLite | 持久化关键状态，进程重启可恢复 |

---

## 📁 项目结构

```
core/
├── config/
│   ├── config_loader.py             # KiraConfig.load_subconfig 接管所有独立 JSON
│   └── default.py                   # 加 default_llm_group / default_fast_llm_group
├── lifecycle.py                     # 实例化 ModelGroup / Registry / Assembler / Translator / Pipeline
├── llm_client.py                    # 加 model_group_mgr 字段
├── message_manager.py               # 主路径串起所有钩子（最重要的那个文件）
├── output/                          # ── Phase 0.5 输出钩子链
│   ├── output_ctx.py                # OutputCtx 数据结构
│   └── default_delay_hook.py        # 内置兜底延迟钩子
├── pipeline/                        # ── Phase 0.4 Hints 流水线
│   ├── __init__.py
│   └── hints_pipeline.py            # HintsKeyHandler + HintsPipeline + FrequencyPolicy + HintsResult
├── plugin/
│   ├── plugin_context.py            # 加 prompt_block_registry / hints_pipeline / model_group_mgr
│   ├── plugin_handlers.py           # EventType 加 ON_PROMPT_ASSEMBLE / AFTER_LLM_RESPONSE_PARSE
│   │                                #   / ON_OUTPUT_PIPELINE / ON_HINTS_PRODUCED
│   └── plugin_registry.py           # 装饰器集 + lifecycle 钩子注册
├── prompt_assembler.py              # ── Phase 0.2 三个来源合一,filter/sort/evaluate
├── prompt_block.py                  # ── Phase 0.2 PromptBlock + ChatInjection + AssembledPrompt
├── prompt_manager.py                # get_agent_prompt 改返回 list[PromptBlock]
├── prompt_translator.py             # ── Phase 0.3 Jinja2 + render
├── prompts/
│   ├── agent_tmpl.py                # ── DEPRECATED 保留兼容
│   └── translator/                  # ── Phase 0.3 11 个 j2 模板
│       ├── role.j2 / persona.j2 / attention.j2 / accounts.j2 /
│       ├── sessions.j2 / time.j2 / chat_env.j2 / memory.j2 /
│       └── tools.j2 / output.j2 / format.j2
└── provider/
    ├── llm_model.py                 # LLMRequest 加 multimodal list-content;旧字段标 DEPRECATED
    ├── model_group.py               # ── Phase 0.1 ModelEntry / ModelGroup / ModelGroupManager
    └── src/openai/model_clients.py  # ── 回迁: v1/chat 生图模式

data/config/                         # 项目惯例:每个新模块一个独立 JSON
├── system_config.json               # 主配置(旧)
├── bot_config.json                  # bot 行为(旧)
├── model_groups.json                # ── Phase 0.1
└── translator.json                  # ── Phase 0.3

core/plugin/builtin_plugins/sticker/
├── main.py                          # 双路径 + StickerHintsKeyHandler
└── schema.json                      # use_hints_pipeline / hints_top_k / hints_policy_*

tests/                               # 全部 stub 测试,不需要数据库 / 真实 LLM key
├── test_model_group_m2.py           # 27 项
├── test_prompt_block_m3.py          # 16 场景(含 in-chat 修订)
├── test_translator_m4.py            # 12 场景(需 Jinja2)
├── test_hints_pipeline_m5.py        # 24 场景(含 FrequencyPolicy + Observer 修订)
└── test_output_pipeline_m7.py       # M7 输出钩子
```

---

## 🚀 快速开始

### 安装

跟原版完全一致，多一个 Jinja2 依赖：

```bash
git clone <this-fork>
cd KiraAI
pip install -r requirements.txt    # 新增 Jinja2>=3.1.0
```

### 启动

```bash
.\scripts\run.bat                  # Windows
./scripts/run.sh                   # Linux/macOS
```

新增配置项默认都是空值或 `false`，**不显式启用就跟原版完全一致**。

### 验证

跑一遍自检确认全部模块就位（不需要 LLM key / 数据库）：

```bash
python tests/test_model_group_m2.py        # ModelGroup
python tests/test_prompt_block_m3.py       # PromptBlock + in-chat 注入
python tests/test_translator_m4.py         # Jinja2 翻译器
python tests/test_hints_pipeline_m5.py     # Hints 流水线 + 频率策略 + 观察者
python tests/test_output_pipeline_m7.py    # 输出钩子链
```

全 PASS 即环境 OK。

---

## ⚙️ 配置参数

<details>
<summary><b>🎯 模型组 (data/config/model_groups.json)</b></summary>

```json
{
  "groups": {
    "sidecar_group": {
      "models": [
        {
          "ref": "openai-deepseek:deepseek-chat",
          "priority": 1,
          "capabilities": ["json_output", "structured"],
          "max_timeout": 30
        },
        {
          "ref": "modelscope-qwen:qwen2.5-72b",
          "priority": 2,
          "capabilities": ["long_context", "creative"],
          "max_timeout": 45
        }
      ],
      "cooldown_429": 300,
      "cooldown_503": 1800,
      "validate": { "min_chars": 1, "require_xml_root": false }
    }
  }
}
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `ref` | string | — | `<provider_id>:<model_id>` 复用现有命名 |
| `priority` | number | 1 | 数字小 = 越早尝试 |
| `capabilities` | string[] | [] | 标签：调用方可 prefer 但不强求 |
| `max_timeout` | number | 30 | 单次请求秒数上限；超过视同失败 |
| `cooldown_429` | number | 300 | 429 错误冷却秒数 |
| `cooldown_503` | number | 1800 | 5xx 错误冷却秒数 |
| `validate.min_chars` | number | 1 | 响应文本最短字符数 |
| `validate.require_xml_root` | boolean | false | 响应必须含合法 XML 根 |
| `validate.require_json` | boolean | false | 响应必须可解析为 JSON |

主对话走模型组：在主配置里设 `"models.default_llm_group": "sidecar_group"`。不配则回退到原 `models.default_llm` 单模型行为。

</details>

<details>
<summary><b>📝 翻译器 (data/config/translator.json)</b></summary>

```json
{
  "template_dir": "",
  "debug_markers": false
}
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `template_dir` | string | "" | 自定义模板目录（相对仓库根或绝对路径）；空 = 用打包的 `core/prompts/translator/` |
| `debug_markers` | boolean | false | true 时每块前后包 `<!-- block:name depth=N source=... -->` 注释；调试用，**生产关掉**（增加 token） |

</details>

<details>
<summary><b>⚡ 跨回复优化 (插件 schema.json)</b></summary>

```json
{
  "use_hints_pipeline": false,
  "hints_top_k": 12,
  "hints_policy_mode": "every",
  "hints_policy_interval": 1,
  "hints_policy_probability": 1.0
}
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `use_hints_pipeline` | boolean | false | true = 走新流水线（prompt 体积小），false = 老路径（prompt 列出全部 sticker） |
| `hints_top_k` | number | 12 | 每次 LLM 吐 `<hints_key>` 时返回多少候选 |
| `hints_policy_mode` | string | "every" | 触发频率模式：`every` / `interval` / `random` |
| `hints_policy_interval` | number | 1 | mode=interval 时的间隔 N |
| `hints_policy_probability` | number | 1.0 | mode=random 时的概率 [0,1] |

**何时启用 pipeline**：sticker 数量 ≥ 10 个时显著节省 prompt token。少于 10 个用老路径更直接。

</details>

<details>
<summary><b>🚪 消息延迟 (bot_config.json)</b></summary>

```json
{
  "bot": {
    "min_message_delay": 0.8,
    "max_message_delay": 1.5,
    "use_native_multimodal": false,
    "use_native_audio_multimodal": false,
    "use_native_video_multimodal": false
  }
}
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `min_message_delay` | number | 0.8 | DefaultDelayHook 最小延迟秒数 |
| `max_message_delay` | number | 1.5 | DefaultDelayHook 最大延迟秒数 |
| `use_native_multimodal` | boolean | false | true = 图片直接 base64 喂给视觉 LLM；false = 走 VLM 描述+缓存 |
| `use_native_audio_multimodal` | boolean | false | 音频原生多模态 |
| `use_native_video_multimodal` | boolean | false | 视频原生多模态 |

延迟范围被框架的 DefaultDelayHook 接管（SYS_LOW 优先级），可被插件钩子在更高优先级覆盖。

</details>

---

## 🎯 模型组（Model Group）

把多个 model entry 配成一个 group，调用方只持 group_id，不持 key/url。框架按 priority 顺序尝试，遇到错误自动切换 + 冷却。

### 错误处理与冷却

| 触发 | 分类 | 冷却时长 | 行为 |
|---|---|---|---|
| HTTP 429 / "rate_limit" | rate_limit | 5 min | 切下一个 |
| HTTP 401/403 / "auth" | auth | 30 min | 切下一个 |
| HTTP 5xx / 连接错误 | unavailable | 30 min | 切下一个 |
| HTTP 400 / "invalid_request" | invalid_request | 不冷却 | 切下一个（可能是 prompt 问题） |
| 超时 | timeout | 5 min（连续 3 次升级 30 min） | 切下一个 |
| 响应校验失败 | validate_fail | 不冷却（连续 3 次同模型升级 30 min） | 切下一个 |
| 全部冷却 | — | 等最早过期的，超 60s 抛异常 | — |

### 切换日志

每次切都打一行 WARN，便于 grep 出"为什么这次走了备用 key"：

```
[ModelGroup] sidecar_group: openai-deepseek:deepseek-chat → modelscope-qwen:qwen2.5-72b (reason: 429)
[ModelGroup] sidecar_group: modelscope-qwen:qwen2.5-72b cooling 1800s (reason: unavailable, status=503)
```

### 默认模式兼容

旧 `models.default_llm` 配置完全不动也能用。框架启动时把它包成虚拟 `__default__` 单成员组，所有调用透明走过这层——升级零迁移。

---

## 🧱 可编排 system prompt

把 system prompt 切成可命名、可排序、可开关、可缓存的 PromptBlock。三个来源在同一管线汇合：

```
框架内置 11 块（role/persona/attention/accounts/sessions/time/chat_env/memory/tools/output/format）
       │
插件静态注册（@register.prompt_block，启动时一次）
       │
插件运行时贡献（@on.prompt_assemble，每轮重新求值）
       │
       ▼
   Assembler  → filter (enabled & condition) → sort by depth → evaluate
       │
       ▼
   AssembledPrompt → Translator (Jinja2) → 最终 system 文本
```

### 内置块布局

11 个内置块占了 `depth=10/20/30/.../110`，**插件用奇数槽位最不容易撞**（5/15/25/.../85）。`95` 已被 skills 占用。

| depth | 块名 | 内容 |
|---|---|---|
| 10 | role | 角色定位 |
| 20 | persona | 人设内容 |
| 30 | attention | 注意事项 |
| 40 | accounts | 账号列表 |
| 50 | sessions | 当前会话 |
| 60 | time | 当前时间 |
| 70 | chat_env | 聊天环境（platform/adapter/chat_type） |
| 80 | memory | 记忆段 |
| 90 | tools | 工具说明 |
| 95 | skills | 技能（已占用） |
| 100 | output | 输出规约 |
| 110 | format | XML 标签规范 + hints_key 用法 |

---

## 📍 in-chat 深度注入

PromptBlock 默认走 system 拖拽

设 `position="in_chat"` 即切换通道：内容不进 system，而是作为独立 message 插入到 `request.messages` 的指定深度（从末尾倒数）。

### 深度语义

假设当前 `request.messages` 长度是 6（含 system + 历史 + 最新 user 消息）：

| `inject_depth` | 落点 | 适合 |
|---|---|---|
| 0 | `messages[-1]` 之后（末尾） | 紧贴生成位的 reminder |
| 1 | 最新 user 消息之前 | **最常用**——LLM 看到注入后立即看到用户输入 |
| 4 | 倒数第 4 条之前 | World Info / 长生命周期语境 |

### 注入消息的 role

可选 `system` / `user` / `assistant`，默认 `system`（注入读起来像框架指令而非对话）。`user` / `assistant` 适合特殊场景（如把 OOC 注入伪装成用户补充输入）。

### 跟 system 拖拽的对比

| 维度 | system 拖拽（默认） | in-chat 注入 |
|---|---|---|
| 用 | 系统级条目（人设、记忆摘要、世界书 constant） | 临时性 / 带位置语义的注入（Author's Note、World Info at depth） |
| 排序键 | `depth`（小 = 靠前），多块 join 成一条 system | `inject_depth`（从末尾倒数），多块独立成 message |
| 落点 | `messages[0]` 那条 system | `messages[len - inject_depth]` |
| role | 始终 system | 可选 system / user / assistant |

---

## 📝 Jinja2 模板

11 个模板从 Python 字符串迁出来，独立文件 + Jinja2 语法。换 prompt 协议（XML → JSON / Markdown）只动模板，不改插件和 Assembler。

### 模板示例（截选 `format.j2`）

```jinja2
## 输出格式（Format）
你的回复需要使用如下 xml tag 结构（非标准 xml，没有 <root> 标签）：

<msg>
    ...
</msg>

其中可以有多个 <msg>，代表发送多条消息。<msg> 中可以使用的标签：

{{ message_types }}

## 资源预取（hints_key）

除了 <msg> 标签外，你还可以输出 0 或多个**顶层** <hints_key> 标签来"预订"
下一轮对话需要的资源：

<hints_key type="资源类型">关键词1,关键词2,关键词3</hints_key>
```

### 自定义模板

把整个 `core/prompts/translator/` 目录复制到 `data/my_templates/`，改你想改的那几个，然后在 `translator.json` 设：

```json
{ "template_dir": "data/my_templates" }
```

缺失文件 / 缺失变量会**静默渲染成空**（不会让框架崩），单个模板出错不影响其他块。

### debug markers

`translator.json` 设 `"debug_markers": true`，LLM 看到的 system prompt 会带块边界注释：

```xml
<!-- block:role depth=10 source=framework -->
你是 ...
<!-- /block:role -->
<!-- block:emotion_snapshot depth=70 source=plugin:emotion -->
## 角色当前情绪
...
<!-- /block:emotion_snapshot -->
```

排查"为什么块顺序不对 / 为什么我的块没出现"很方便。**生产环境关掉**，会增加 token 数。

---

## ⚡ Hints 异步流水线

很多 prompt 内容需要现拉（表情包候选、记忆检索、世界书查询）。但拉的那一秒不能让用户等——主 LLM 已经在生成回复了。

把这件事泛化成两轮流水线：

```
轮 N:    LLM 输出 ... <hints_key type="memory">小学,游泳,刘老师</hints_key> ...
   │
   │ (后台异步)
   ├─ Pipeline 解析 <hints_key>
   ├─ FrequencyPolicy 决定要不要触发本次
   ├─ 如触发 → 调你的 prepare(keys, sid, ctx) → 返回 PromptBlock
   └─ 存入 pending[sid][type]，等下一轮
   
轮 N+1:  Pipeline 自动把准备好的 PromptBlock 注入下一次 prompt
        LLM 这次能看到 "## 候选记忆：..."
```

### 第一个使用者：表情包

原本 sticker 插件在每轮都把全部表情包列在 prompt 里，10 个以上 prompt 体积明显膨胀。pipeline 模式只在 prompt 里描述用法 + `<hints_key>` 协议；LLM 想用时主动吐 key，handler 用关键词子串匹配 top-k，下一轮注入候选列表，LLM 用 `<sticker>id</sticker>` 实际发送。

**双路径配置** (`use_hints_pipeline`)：

| 模式 | tag description | prompt 体积 |
|---|---|---|
| `false`（默认） | 全部 sticker 列在 prompt 里 | O(N)，N = 表情包数量 |
| `true` | 只描述用法 + `<hints_key>` 协议 | O(K)，K = top_k 候选数（默认 12） |

出问题就关掉，零回滚成本。

### 关键约束

| 约束 | 作用 |
|---|---|
| **prepare 必须不阻塞** | 用 `asyncio.create_task` 派发，立即返回，不阻塞用户回复 |
| **TTL 老化** | 默认 600s 未消费的 pending 自动丢弃 |
| **200ms backpressure** | 下一轮等最多 200ms，超时就跳过这一轮（不让流水线拖累主路径） |
| **覆盖式更新** | 同 session 同 type 重发 → 取消旧 in-flight，最后一次胜出 |
| **跨 session 隔离** | sid_A 的 pending 不会被 sid_B 看到 |
| **handler 异常隔离** | prepare 抛异常 → 一行 warn，pending 不写入，不影响主路径 |

---

## 🎚️ 三种触发频率

不是每次 LLM 吐 `<hints_key>` 都要触发 prepare。三种模式可选：

| 模式 | 行为 | 适合 |
|---|---|---|
| `every`（默认） | 每次 emission 都触发 | M5 默认，跟原行为一致 |
| `interval` | 仅在第 N、2N、3N… 次触发，其余跳过 | "内容刷新慢"的资源，同主题反复检索没意义 |
| `random` | 每次以概率 `p` 触发，可设 seed 复现 | "人类有时不查"的非确定性，emergent variety |

策略在**调度入口**生效——跳过的 emission 不调度 prepare（省 cost），但仍触发观察者事件以便 /debug 看 skip 速率。

### 配置示例

```json
{
  "hints_policy_mode": "interval",
  "hints_policy_interval": 3
}
```

效果：sticker 候选只在 LLM 第 3、6、9… 次吐 `<hints_key type="sticker">` 时准备；其余轮次空气。

```json
{
  "hints_policy_mode": "random",
  "hints_policy_probability": 0.3
}
```

效果：每次 LLM 吐 key 都有 30% 概率触发。70% 轮次空气，让 LLM 偶尔"自由发挥"。

---

## 👀 跨插件 hint 观察

handler 产出 hint 时（成功 / None / 异常 / policy skip 都触发）会 fire-and-forget 派发 `ON_HINTS_PRODUCED` 事件。其他插件能"听"到这个事件，做自己的反应：

- **logging 插件**：记录每个 hint 的产出 / 失败率 / 平均耗时
- **emotion 插件**：观察 memory_recall 的 hint 内容，相应调情绪
- **debug 面板**：实时看 in-flight pending 状态

观察者用 `asyncio.create_task` 启动，慢观察者**不阻塞**用户回复路径；多个观察者间无序。事件携带的 `HintsResult` 有：

| 字段 | 含义 |
|---|---|
| `session_id` | 哪个 session |
| `key_type` | 哪个 type 的 hint |
| `keys` | 解析出来的关键词列表 |
| `block` | 产出的 PromptBlock；None 表示 skip / 异常 / handler 返回 None |
| `error` | 异常信息字符串（handler 抛异常时） |
| `elapsed_ms` | prepare 耗时 |
| `skipped_by_policy` | True 表示被 FrequencyPolicy 跳过 |
| `plugin_id` | handler 所属插件（识别来源） |

---

## 🚪 输出钩子链

LLM 输出 → XML 解析 → 真正发送 之间，开放一组可注册钩子。每个钩子拿到同一个 `OutputCtx`，可以：

| 操作 | 字段 | 效果 |
|---|---|---|
| 改消息 | `ctx.chains` | 拆 / 合并 / 编辑 / 删除发出去的消息 |
| 设延迟 | `ctx.delays` | 每条消息发送前等待秒数（None = DefaultDelayHook 填默认） |
| 拦截 | `ctx.intercepted=True` | 已读不回；框架不发送，但仍触发 ON_STEP_RESULT |
| 跨钩子通信 | `ctx.meta` | 钩子之间共享笔记 |
| 看预算 | `ctx.budget_ms` / `ctx.api_elapsed_ms` | 从用户消息时刻起算的延迟预算余量 + 本轮 LLM 耗时 |

### 内置 DefaultDelayHook

框架自带一个 `SYS_LOW` 优先级的钩子，在 `delays` 还有 None 槽位时填 `random.uniform(min, max)`——行为跟原版一致。更高优先级的钩子设过的 delay **不会被覆盖**（DefaultDelayHook 只填 None）。

### 关键约束

| 约束 | 作用 |
|---|---|
| **delay > 5s 不要在这做** | session_lock 整体在锁内，长 sleep 阻塞同 session 后续消息 |
| **重型副作用要异步** | 写日志 / 推 memory 用 `asyncio.create_task`，不阻塞链 |
| **拦截后 ON_STEP_RESULT 仍触发** | `message_results=[]`，下游 handler 要容忍 |

### 拦截语义（已读不回的基础）

`ctx.intercepted=True` 时框架不发送任何 chain，但 `ON_STEP_RESULT` 仍触发——下游插件能知道"这一轮想说但没说"，便于把内容存入 unsent 表（具体 unsent 机制留给后续 phase）。

---

## 🛡️ 兼容性

每个特性都是**追加式**的，老用法继续工作：

| 旧用法 | 是否还能用 | 备注 |
|---|---|---|
| `@on.llm_request` 改 `request.system_prompt` / `user_prompt` | ✅ | 标了 `# DEPRECATED`，Phase 1 才删；Assembler 在它之后跑，老钩子加的 Prompt 落在后面 |
| `models.default_llm` 直接配单模型 | ✅ | 内部包成虚拟 `__default__` 单成员组 |
| 表情包"在 prompt 里全量列出"路径 | ✅ | sticker 插件 `use_hints_pipeline=false`（默认）走老路径 |
| `random.uniform` 风格的消息延迟 | ✅ | 由 `DefaultDelayHook` 在 SYS_LOW 接管，行为不变 |
| 老的 `agent_tmpl.py` 字符串模板 | ✅ | 标了 `# DEPRECATED`，Translator 已平迁到 j2 |
| `LLMRequest.assemble_prompt()` | ✅ | 标了 `# DEPRECATED`，message_manager 仍调它构造 messages |

---

## 🔄 旧侵入式修改回迁

之前在原仓库基础上手改过的几个增强，在 M7 里一并回迁到新框架：

### OpenAI Image Client v1/chat 模式

类属性 `USE_CHAT_COMPLETIONS=True` 切换到 `/v1/chat/completions` 多模态聊天接口出图。从三种形式提取图片：

- Markdown `![alt](url)` 链接
- `data:image/...` URI
- `output_image` 内容部件

**明确拒绝从纯文本里抽 URL**——防 prompt injection。

### 原生多模态

图 / 音 / 视频 base64 直喂支持视觉的 LLM，跳过 VLM 描述+缓存路径。三个开关独立：

| 配置 | 启用范围 |
|---|---|
| `bot_config.bot.use_native_multimodal` | 图片 |
| `bot_config.bot.use_native_audio_multimodal` | 音频 |
| `bot_config.bot.use_native_video_multimodal` | 视频 |

默认全 `false`，走原 VLM 描述路径。打开后 `LLMRequest` 的 user message 自动从字符串 content 切换到 OpenAI vision 的 list-content 形态。

### `desc_video` 工具函数

`core/utils/common_utils.py` 新增，跟 `desc_img` 同位置，给"视频不走原生多模态但仍想要描述"的场景。

---

## 📊 注入效果示例

启用 `debug_markers=true` 后，主 LLM 收到的 system prompt 形如（截选）：

```xml
<!-- block:role depth=10 source=framework -->
你是一个独立的人，不要询问"能为你做什么"...
<!-- /block:role -->

<!-- block:persona depth=20 source=framework -->
## 角色信息
你叫 Kira，一名喜欢科技与文学的程序员...
<!-- /block:persona -->

<!-- block:emotion_snapshot depth=70 source=plugin:emotion -->
## 角色当前情绪
- 主导：好奇（intensity=0.6, topic=新出的本地大模型）
- 基线 mood：valence=+0.2, arousal=+0.1
<!-- /block:emotion_snapshot -->

<!-- block:memory depth=80 source=framework -->
**近期要点**
- [01-15 14:30~15:00] 用户提到周末计划去爬山...

**相关回忆**
- [01-10 20:15] 上次爬山用户B迟到了1小时
<!-- /block:memory -->

<!-- block:sticker_hints_candidates depth=92 source=plugin:sticker -->
## 本轮可用的 sticker 候选
（基于上一轮你预订的关键词：开心,大笑,赞同；共 5 个）
[sticker_042] 登山小人加油
[sticker_015] 猫猫歪头疑惑
[sticker_088] 程序员大笑
...
<!-- /block:sticker_hints_candidates -->

<!-- block:format depth=110 source=framework -->
## 输出格式
你的回复需要使用如下 xml tag 结构...
## 资源预取（hints_key）
除了 <msg> 标签外，你还可以输出 0 或多个**顶层** <hints_key> 标签...
<!-- /block:format -->
```

各块按 depth 排序拼接，插件内容自然嵌入框架内置块之间。in-chat 注入的内容则不在这里——它作为独立 message 出现在 `messages` 中段。

---

## 🤝 与原版 KiraAI 的关系

这个 fork **不替换**原版的任何核心模块，只是**追加**框架级能力。原版的：

- 适配器系统（QQ / 微信 / 等）
- 会话管理
- 消息处理流程
- 内置 chat / kira-ai 插件
- WebUI

全部不动。Phase 0 在它们之间和之上加了模型组、PromptBlock 管线、Hints 流水线、输出钩子。

老插件零迁移继续工作；想用新特性的插件按需开关或注册新装饰器即可。

---


---

## 📄 License

跟随上游 KiraAI。

---

> 有问题或想聊聊设计思路，欢迎开 issue。
