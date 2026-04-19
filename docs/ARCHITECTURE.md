# 架构与实现报告

> 本文档基于代码静态阅读生成，描述 `wechat-game-automation-agent` 的整体设计、关键模块实现、踩坑点以及扩展方法。阅读对象：新加入的开发者 / 未来的自己 / Claude Code 新会话。

---

## 1. 一句话定位

**基于 VLM（视觉语言模型）的 macOS 微信小游戏自动化框架**。通用 `GameLoop` 负责"截图 → 推理 → 执行"的主循环，每款游戏通过实现 `BaseGame` 适配器接入。当前落地一款：

| 游戏 | 类型 | 动作空间 | AI 后端 | 关键机制 |
|------|------|----------|---------|----------|
| `endless_winter`（无尽冬日） | 开放 UI 操作 | `click` / `drag` / `long_press` | **双系统**：Qwen VL + Gemini | 场景分类、变化检测、A-B 循环防死锁、CV 对齐 |

---

## 2. 运行入口与命令

### 2.1 CLI 总览（`main.py`）

```bash
python3 main.py --game endless_winter [options]
```

关键开关：

| 开关 | 作用 | 默认 |
|------|------|------|
| `--max-rounds N` | 最大轮数（覆盖 yaml） | 200 |
| `--interval S` | 每轮间隔（秒） | 1.5 |
| `--dry-run` | 只分析、不下发点击 | False |
| `--log-level DEBUG` | 详细日志 | INFO |
| `--use-gemini` | 强制使用 Gemini 云端 | False |
| `--use-local` | 使用本地 Qwen VL（`--local-api-url`） | True |

### 2.2 启动流程

`main.py`:

1. `load_dotenv` → `.env` 注入 `GEMINI_API_KEY`
2. `load_global_config()` + `load_game_config(name)` → 合并两级 YAML（pydantic 校验）
3. 依据 `--use-local/--use-gemini` 构造 `LocalVisionClient` 或 `GeminiVisionClient`
4. **动态导入** `games.<name>.game` 模块，约定类名为 `<CamelCase>Game`（`endless_winter` → `EndlessWinterGame`）
5. 尝试 `GameClass(config, ai_client=...)`，若不支持 `ai_client` 形参则回退为 `GameClass(config)`
6. `GameLoop(game, ai_client, ...).run()`

> 没有测试套件、没有 lint、没有 build。唯一的校验手段是跑真实游戏或 `--dry-run`。

---

## 3. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                       main.py                           │
│  CLI → 配置加载 → 客户端装配 → GameLoop.run()             │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  core/game_loop.py                      │
│                    GameLoop                             │
│   ┌──────────────────────────────────────────────┐      │
│   │  while round < max:                          │      │
│   │    1. ScreenCapture.capture_game_area()      │      │
│   │    2. is_screenshot_valid()                  │      │
│   │    3. game.identify_scene() → 分支处理        │      │
│   │    4. game.is_game_over() / on_game_over()   │      │
│   │    5. game.analyze_with_retry() → AI 推理    │      │
│   │    6. game.parse_ai_response() → action dict │      │
│   │    7. game.execute_action(action)            │      │
│   │    8. 记录成功/失败                            │      │
│   └──────────────────────────────────────────────┘      │
└──────┬─────────────────────────────────────────────┬────┘
       │                                             │
       ▼                                             ▼
┌──────────────────┐                        ┌─────────────────────┐
│  core/ (基础设施) │                        │  games/<name>/ (适配) │
│  ─ screen.py     │                        │  ─ game.py (BaseGame) │
│  ─ input_ctrl.py │                        │  ─ prompts.py         │
│  ─ ai_client.py  │                        │  ─ change_detector    │
│  ─ base_game.py  │                        │  ─ scene_classifier   │
│  ─ quartz_window │                        │  ─ stuck_monitor      │
└──────────────────┘                        └─────────────────────┘
```

核心分层：

- **基础设施层（`core/`）**：窗口检测、截图、点击、AI 客户端、抽象基类。游戏无关。
- **游戏适配层（`games/<name>/`）**：实现 `BaseGame` 协议，内含专用 prompt / 解析 / 执行 / 启发式辅助。
- **配置层（`config/`）**：全局 `settings.yaml` + 每游戏 `games/<name>.yaml`，pydantic 解析。

---

## 4. 核心基础设施模块

### 4.1 `core/base_game.py` — 游戏适配协议（93 行）

抽象基类，定义游戏必须提供的接口：

**必须实现（3 个）**：
- `build_prompt(screenshot) -> str`
- `parse_ai_response(response) -> dict`
- `execute_action(action) -> None`

**可选钩子**：`on_round_start` / `on_round_end` / `is_game_over` / `on_game_over` / `get_game_name` / `identify_scene` / `analyze_with_retry`

额外约定（非协议的隐式契约，`GameLoop` 直接读属性）：`top_offset` / `bottom_offset` / `resize_width` / `use_resize`

### 4.2 `core/game_loop.py` — 通用主循环

最关键的文件。除了"截图 → 推理 → 执行"三步，还附带大量保护与观测逻辑：

**PerformanceStats 类**：记录每帧的 screenshot/ai/parse/exec 耗时、confidence、action 类型、error 类型，结束时打印表格。

**主循环状态机**（`run()`）：

```
每轮开始
 ├─ skip_next_screenshot?  → 跳过本轮（动作后让游戏渲染）
 ├─ capture_game_area      → screenshot
 ├─ is_screenshot_valid    → 黑屏/单色则跳过，≥5 次加长等待
 ├─ identify_scene (可选)
 │    ├─ game_over  → on_game_over(), continue
 │    ├─ loading   → sleep 1s, continue
 │    ├─ menu      → TODO
 │    └─ error     → on_game_over() / break
 ├─ is_game_over (仅 round>5 且 执行过 >3 次)
 │    └─ on_game_over() → 继续/停止
 ├─ analyze_with_retry or (build_prompt + ai_client.analyze)
 │    └─ save_debug_screenshot(ss + response)
 ├─ parse_ai_response → action dict
 ├─ execute_action(action, screenshot=...)
 │    ├─ 成功 → total_jumps_executed++, skip_next_screenshot=True
 │    └─ skip 累计 ≥5 → on_game_over() 强制重启
 └─ sleep(loop_interval)
```

**连续失败保护**：`consecutive_failures` 达 3 次或 `consecutive_skip_actions` 达 5 次自动触发重启/停止。

### 4.3 `core/screen.py` + `core/quartz_window.py` — macOS 截图

- 通过 `CGWindowListCopyWindowInfo` 枚举所有窗口，匹配 `owner_name in ("WeChat", "微信")`。
- 评分规则：`width < 600 且 0.3 < aspect < 1.5` 打满分（`area`），否则 `area * 0.1`，防止误选聊天主窗口。
- 关键方法：`_capture_window_direct()` 用 `CGWindowListCreateImage` **直接抓窗口内容**（`kCGWindowListOptionIncludingWindow`），被其它窗口遮挡也能正常获取 —— 这是 macOS 自动化的关键，不依赖前台可见性。
- 路径：`CGImage → NSImage → TIFF → PIL.Image`（通过 `BytesIO`）。
- `top_offset` / `bottom_offset` 在捕获阶段裁剪 + 缩放时挂 `screenshot.resize_scale` 属性供游戏还原真实坐标。
- `is_screenshot_valid()`：三重校验（mean < 20 黑屏、std < 5 单色、unique colors < 20 过少颜色）。
- `enhance_image()`：PIL 对比度+亮度增强，低 confidence 时可用于二次推理。
- `save_debug_screenshot()`：每轮写 `debug/screenshots/<game>/round_NNN_HHMMSS.{png,txt}`，png 是截图，txt 是 AI 原始响应。

### 4.4 `core/input_controller.py` — macOS 输入

**只有 94 行但全是踩坑换来的**：

- **放弃 `pyautogui`**，走 `subprocess.run(["cliclick", ...])`。原因：pyautogui 的 CGEvent 会被微信静默丢弃（经验证：pyautogui 点击后像素差约 1.8 纯噪声，cliclick 后像素差 25.8 真实 UI 变化）。
- API：`press_and_hold(dd+sleep+du)`、`click(c 或 dd+du)`、`drag(dd → 多次 dm → du，内插步数 ≥4)`、`key_press(kp:<key>)`、`move_to(m)`。
- 没有任何防护 / 坐标校验 —— 上游保证。

### 4.5 `core/ai_client.py` — 双模型客户端

两个并列类，接口兼容：`analyze(image, prompt, max_retries)` + `analyze_text(prompt, ...)`。

**GeminiVisionClient**（直接 HTTP，不依赖 google 官方 SDK）：
- endpoint：`https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent`
- 使用 `X-goog-api-key` header
- base64 JPEG（质量 85）
- 429 / 其他错误统一 **指数退避**（`2^attempt` 秒），默认 3 次重试
- 默认模型：`gemini-flash-latest`（`config/settings.yaml`）

**LocalVisionClient**（OpenAI 兼容协议，对接 LM Studio）：
- 默认地址 `http://192.168.1.156:1234`
- 默认模型 `qwen/qwen3-vl-8b`
- `max_tokens=2000`, `temperature=0.3`
- 支持 `image_format` = `jpeg` 或 `webp`（endless_winter 用 webp quality=70 降带宽）
- timeout=120s（本地推理慢）
- 重试时固定 sleep 5 秒，不指数退避
- **没有 429 处理**（本地模型不会限流）

### 4.6 `utils/`

- `config_loader.py`：pydantic 模型（`GlobalConfig`、`GameConfig`），全部 `extra="allow"` 以便各游戏自由扩展字段。环境变量 `GEMINI_API_KEY` 覆盖 yaml 值。首次运行会生成默认 `settings.yaml`（但默认模型字符串是过时的 `gemini-2.0-flash`，当前实际用 `gemini-flash-latest`）。
- `logger.py`：标准控制台 + `debug/autoplay.log` 文件双路输出。
- `image_utils.py`：`save_screenshot` / `draw_crosshair` / `resize_image`，目前只在少量调试场景使用。

---

## 5. 游戏实现：`endless_winter`

### 5.1 双系统架构（Dual-System）

灵感源自 Kahneman 的 System 1 / System 2：

| | System 1 | System 2 |
|---|---|---|
| 模型 | 本地 Qwen VL (Ref 格式) | Gemini 云端 (JSON) |
| 返回 | `目标=xxx\n(xmin,ymin,xmax,ymax)`（0–1000 归一化） | 结构化 JSON（像素坐标） |
| 延迟 | 快 | 慢（+网络） |
| 限流 | 无 | `min_interval_s=5.0` |
| 动作 | 仅 click | click / drag / long_press |

**触发 S2 的条件**（任一）：
- S1 连续失败 ≥ `system1.fail_threshold`（默认 3）
- `_force_system2` 标志位被 StuckMonitor 或"同名目标 3 次"检测置位
- 连续 `NO_CHANGE` ≥ 5 次

### 5.2 `analyze_with_retry` 控制流

```
┌── 黑名单衰减（每轮 remaining--，0 则移除）
├── ChangeDetector 闸门
│    └── 上次已成功执行 且 MSE 未超阈值 → return "NO_CHANGE"（_no_change_count++）
│        （连续 5 次则强制绕过此闸门）
├── SceneClassifier.get_scene() → main_city/wilderness/battle/loading/dialog/ad/unknown
├── 场景级 action 缓存（TTL=2s）命中 → 返回缓存
├── 记录本次 screenshot 的时间戳 + 灰度副本（供新鲜度校验）
├── System 1（非 loading 场景且未 _force_system2）
│    ├── build_system1_prompt_for_scene（含场景优先级 + 短期记忆 + 主城禁区提醒）
│    ├── Qwen → parse_qwen_ref_bbox → (cx, cy) 像素
│    ├── 黑名单半径 60px 命中 → 丢弃
│    ├── 近 3 次目标描述相同 → set _force_system2 = True, 进入 S2
│    ├── 命中"前往"/"任务栏"/"关闭/返回"/通用 按钮 → CV snap 对齐
│    ├── 缓存 _last_parsed_action, 返回 "S1_OK"
│    └── 失败 → _s1_fail_count++；达阈值则进入 S2
├── System 2（达阈值或被强制）
│    ├── 距上次调用 < min_interval_s → return "SKIP"
│    ├── build_system2_prompt_with_drag（历史 5 步 + 强制关闭/返回规则）
│    ├── Gemini → parse_gemini_json
│    ├── 底栏按钮（"一键已读"等）snap 到 y=95.5%
│    ├── 关闭按钮（X/返回）若不在顶部带则 snap 到 (93%, 7%)
│    ├── 缓存，返回 "S2_OK"
│    └── 失败 → return "SKIP"
└── 所有系统失败 → "SKIP"
```

`parse_ai_response` 的作用实际上是**消费 `_last_parsed_action` 缓存**：`analyze_with_retry` 已经把结构化 action 算好了，parse 只是按 sentinel 字符串（`NO_CHANGE` / `S1_OK` / `S2_OK` / `SKIP`）取出对应数据。

### 5.3 辅助组件

**`ChangeDetector`（85 行）**：
- 输入 PIL 图 → 灰度 → GaussianBlur(5×5) → 跳过顶部 `roi_skip_top` (10–15%) → 与上一帧做 `absdiff` → 均值 > threshold(30–40) 视为变化
- 目的：静态 UI（等待状态）下直接跳过 AI 推理，节省成本

**`SceneClassifier`（114 行）**：
- 调 Qwen `build_scene_classification_prompt`，返回 7 类场景之一
- 内置缓存 `_classify_interval=5s`，避免每帧分类
- 关键字回退：若模型返回不规范，按中文关键字匹配（"弹窗"→dialog、"广告"→ad…）

**`StuckMonitor`（114 行）**：
- 维护 `deque(maxlen=click_threshold)`，每次 click 坐标按 `grid_size` 栅格化后入队
- `is_stuck()` 判定：
  - 所有点相同 → stuck（同一位置反复点击）
  - **只有 2 个不同点 且 threshold ≥ 3 → stuck**（A-B 循环，如 S1 在两个错误位置来回横跳）
- `recover()`：先 `Escape`，再尝试点 (93%, 5%) 关闭按钮、(7%, 5%) 返回按钮；并要求 SceneClassifier 重新分类

**CV Snap（`game.py` 中的多个 `_snap_*` 方法）**：
- HSV 阈值找蓝色/深色 blob → cv2 轮廓 → 按面积 / 纵横比 / 位置过滤
- 专用 snap：`前往` 按钮（右侧窄蓝色 300–1500 px²，aspect 2–5）、任务栏（底部最宽蓝色带）、关闭按钮（顶部角落）
- 通用 snap：若 blob 中心距 AI 坐标 ≤ 45px 则用 blob 中心替换，否则保留 AI 坐标
- **保守策略**：不用 MORPH_CLOSE（会把相邻横幅吸进来），距离不够就 return None

### 5.4 关键状态字段

| 字段 | 含义 |
|------|------|
| `_s1_fail_count` / `_fail_threshold`(3) | S1 连续失败计数 |
| `_action_history` (deque 5) | 近期动作（供 S1 prompt 作短期记忆，包含 target/x/y/outcome） |
| `_last_parsed_action` | `parse_ai_response` 消费的缓存 |
| `_s2_last_call_time` / `_s2_min_interval`(5s) | Gemini 速率限制 |
| `_scene_classifier` / `_stuck_monitor` / `_change_detector` | 三个辅助单例 |
| `_force_system2` | stuck 检测/同名目标检测置位 |
| `_stuck_blacklist`: `[(gx, gy, remaining_rounds), ...]` | 黑名单，半径 60px，默认 5 轮内拒绝该区 |
| `_recent_targets` (deque 3) | 同名目标检测 |
| `_no_change_count` / `_no_change_force_s2_threshold`(5) | 连续无变化帧计数 |
| `_scene_action_cache` / `_cache_ttl`(2s) | 场景级动作缓存 |
| `_stale_threshold`(1.5s) / `_stale_mse_threshold`(500) | 新鲜度校验（执行前再查一次） |

### 5.5 高阶能力层（planner / memory / verifier / learning / model_router / completion_checks / research）

在双系统之外，`endless_winter` 通过 7 个可独立开关的模块提升"像商业 agent"的能力，全部通过 yaml 配置开启（多数默认开启，`research` 默认关）：

| 模块 | 文件 | 作用 |
|------|------|------|
| **Planner（任务树）** | `core/planner.py` + `config/plans/endless_winter.yaml` | DFS 取第一个 pending 叶任务，把"当前子目标+描述+预期场景+成功关键字"通过 `active_task_context()` 注入 S1 prompt 首行 |
| **MemoryStore（跨会话记忆）** | `core/memory_store.py` (SQLite `data/memory.db`) | 3 张表：`trajectories`(所有动作+结果) / `skills`(scene+target → 稳定成功坐标) / `blacklist`(带 TTL 的持久失败区域) |
| **Verifier（后置校验）** | `core/verifier.py` | 动作前 `record(expected_scene, scene_before)`；下一帧 `verify(scene_after, changed)` 产出 `{ok, reason, scene_matched}`；驱动 planner 成败、memory 黑名单 |
| **TrajectoryLogger（学习数据）** | `core/trajectory_logger.py` + `scripts/export_dpo.py` | 每次 infer+action+outcome 写一行 JSONL；`export_dpo.py` 把同场景下的成功/失败回合配对成 DPO `{prompt, chosen, rejected}` |
| **ModelRouter（多家 GUI VLM）** | `core/model_router.py` + `scripts/run_benchmark.py` + `tests/benchmark/scenarios.yaml` | `system1.model_family` 选 `qwen_vl`（默认）/`ui_tars`/`cogagent`；benchmark 跑预标注场景算 grounding 准确率 |
| **CompletionChecks（完成判定）** | `core/completion_checks.py` | 可插拔的任务完成判定注册表，`Planner.tick(ctx, idle)` 内按 yaml 中的 `completion_check` 字段路由；默认走 **稳定性回退**（expected_scene 下连续 N 轮 idle 视为完成） |
| **Research（网上攻略）** | `core/research.py` + `scripts/fetch_research.py` + `data/research/<game>.md` | Gemini 拉取中文攻略要点，带 frontmatter 缓存（`refresh_days` 控制新鲜度），在 S1 prompt 开头注入【游戏攻略参考】 |

### 5.6 控制流新增关节

```
analyze_with_retry
 ├─ purge_expired_blacklist
 ├─ ChangeDetector → 若 no_change：verifier.verify(scene_after=None, changed=False) → return "NO_CHANGE"
 │                   → _tick_planner(idle=True, screenshot)  ← 稳定性累计
 ├─ SceneClassifier.get_scene() → 同时 verifier.verify(scene_after, changed=True)
 │    └─ verify 结果 → memory.record_action + memory.add_blacklist（失败时，TTL 600s）
 │                    + planner.mark_failure（失败时）
 │                    + trajectory_logger.log
 │    → _tick_planner(idle=not changed, screenshot)  ← 含 completion_check + 稳定性回退
 ├─ System 1
 │    ├─ model_family != qwen_vl → _run_system1_routed（经 model_router）
 │    └─ prompt 首行依次注入：【游戏攻略参考】(research) + planner.active_task_context()
 ├─ S1/S2 成功 → verifier.record(action, expected_scene=active_task.expected_scene)
 └─ _do_click 额外闸门：memory.is_blacklisted(scene, x, y) 命中则拒绝
```

### 5.7 任务完成判定（C 框架 + B 回退）

之前的实现用"成功关键字命中 + verifier 通过"触发 `mark_success`，踩了两个坑：关键字假阳性 + 一次成功就永久标记完成。现在分两层：

**B 层（稳定性回退，默认）**：`Planner.tick_stability(scene_matches, idle)` —— 处于 `expected_scene` 且连续 `stability_rounds` 轮无动作或 NO_CHANGE 视为完成。对应 yaml：

```yaml
tasks:
  - name: dismiss_popups
    stability_rounds: 2
    repeat_until_stable: true   # 完成后立刻回到 pending，下次弹窗出现再次触发
```

`repeat_until_stable` 用于周期性任务（弹窗、任务栏），`mark_success` 里若见到该标志会把 status 直接复位为 `STATUS_PENDING`。

**C 层（可插拔检查，可选）**：`core/completion_checks.py` 暴露 `@register(name)` 装饰器 + `CheckContext` dataclass（screenshot/scene/active_task/memory/qwen_client）。yaml 里通过 `completion_check: qwen_yesno` 或 `completion_check: {type: ..., prompt: ...}` 指定；`Planner.tick` 优先调 C，返回 True 才 mark_success，否则回退到 B。当前 `qwen_yesno` 是 stub（抛 `NotImplementedError`，被 `tick` 捕获并仅 debug 日志，不影响运行）。这是为未来"主动用 VLM 问一句'任务完成了吗'"留的接入点。

### 5.8 研究（Research）层

`core/research.py` 实现了一个**带 frontmatter 的 markdown 缓存**：

```markdown
---
game: endless_winter
fetched_at: 1744800000
source: gemini | cold_start_placeholder
queries: [...]
---
# 正文
```

- `ResearchStore(game)`：`load()` / `save(body, queries, source)` / `age_days()` / `is_stale(days)` / `load_for_prompt(max_chars)`。
- `fetch_with_gemini(client, game, queries)`：组一段中文 prompt 让 Gemini 按四个主题（日常/建筑/避坑/技巧）写要点，返回纯文本。**当前没走 Google Search grounding tool**，就是纯文本让 Gemini 依靠训练知识答；未来想真正联网可在 ai_client 里加 `tools: [{google_search: {}}]`。
- `_format_for_prompt`：strip URL、截到最后一个完整换行、卡在 `prompt_budget_chars`（默认 600）。

默认 `research.enabled: false`，用户需主动跑 `python3 scripts/fetch_research.py --game endless_winter` 覆盖 `data/research/endless_winter.md`（当前是 cold-start 占位，手写了 4X 手游的通用套路），然后把 yaml 开关打开。

### 5.9 Prompt 设计要点

**System 1 Scene-Adapted Prompt**：
- 7 个场景各有不同的"次优先级"提示
- 内嵌"**主城画面禁区**"：头像 / 右下邮件 / 野外按钮 明确列入黑名单（修复头像被误认为返回箭头的语义错误）
- 注入"最近失败坐标"短期记忆，若同名按钮在多处，要求选**位置不同于历史坐标**的那个
- 格式要求"两行输出"：先中文目标名，再坐标框 —— 强制模型先描述再定位，显著提升准确率
- 0–1000 归一化坐标（Qwen VL 的固定格式）

**System 2 Drag-Enabled Prompt**：
- 显式告知"返回像素坐标"（不是 0–1000）
- **强制规则**：看到关闭/返回按钮必须返回它的坐标，不允许选 tab/一键已读/领取/前往
- 说明 click/drag/long_press 三种动作的使用场景
- 每条动作类型给出独立 JSON schema 示例
- 历史 5 步动作注入上下文，"如果最近 3 步同目标 → 立刻切换关闭/返回或 drag"

**Parser**：
- `parse_qwen_ref_bbox`：支持 3 种正则（`<|box_start|>` / `<|im_start|>` / 裸括号），取 bbox 中心 → 反归一化 → clamp
- `parse_gemini_json`：从响应里找 `{...}` 直接 `json.loads`

---

## 6. 配置系统

### 6.1 两级 YAML

```
config/
├── settings.yaml              # 全局：gemini/wechat/loop/logging
└── games/
    └── endless_winter.yaml    # system1/system2/scene/stuck/change_detection/inference/vision
```

### 6.2 endless_winter 配置要点

- `vision.top_offset = 40`（不是 88，否则裁掉 ← 和 ✕）
- `vision.resize_width = 720`（省钱模式；若 Qwen bbox 开始漂回 null）
- `inference.image_format = webp, quality = 50`（省钱模式；精度不够回 70+）
- `change_detection.threshold = 25, roi_skip_top = 0.15`（激进 NO_CHANGE；慢动画被误判为静止时回 30–40）
- `scene_classifier.classify_interval_s = 5.0`
- `stuck_monitor.click_threshold = 3, grid_size = 30`
- `memory.skill_min_success = 1`（1 次成功即缓存 (scene,target)→坐标；verifier + blacklist 做错误回退）

### 6.3 pydantic 扩展性

所有配置类都设 `extra = "allow"`，意味着**游戏随便加 yaml 字段都可以自动穿透到 `config.model_dump()`**，无需同步修改 pydantic 模型。这是刻意的，方便快速迭代。

---

## 7. 关键硬约束与踩坑

### 7.1 macOS 输入

- **必须用 cliclick，不能用 pyautogui**。WeChat 静默丢弃 pyautogui 的 CGEvent。
- 点击前应 `ScreenCapture.activate_wechat()` 把微信置前（`NSApplicationActivateIgnoringOtherApps`）。
- Quartz `CGWindowListCreateImage` 是唯一可靠的截图路径，`ImageGrab` 会被遮挡。

### 7.2 坐标系

- **窗口坐标 vs 屏幕坐标**：AI 返回的是截图内坐标，点击前需加 `window.x` 和 `top_offset`。
- **resize 会影响坐标**：`screenshot.resize_scale` 属性保留原始→缩放的比例，`execute_action` 要乘回去。
- **Qwen 0–1000 归一化**：`parse_qwen_ref_bbox` 内部除以 1000 再乘 `img_width/height`。
- **`top_offset` 自洽**：既作用于截图裁剪，也作用于点击 Y 偏移（`screen_y = win_y + top_offset + y`）。改它不会错位。

### 7.3 Prompt 设计

- Qwen 的 bbox 必须配合"先描述后坐标"的两行输出，否则会乱给。
- 主城画面**没有**返回/关闭按钮 —— prompt 要显式告诉模型，否则头像会被误识别成返回箭头。
- 红色资源数字表示"不够"但 **仍应点升级按钮**（游戏会引导资源获取）—— prompt 和 CV 都不做拦截。
- 同名按钮出现多处时，要把"最近失败坐标"列入 prompt，引导模型选 **位置不同的** 那个。

### 7.4 时序陷阱

- **动作后故意跳过下一帧**（`skip_next_screenshot`）：给游戏渲染时间，否则截到过渡动画，所有判定都错。
- **game over 检测有延迟阈值**：必须 `round > 5 且 total_jumps > 3`，防止本地模型在前几帧误判。

### 7.5 S1/S2 平衡

- 不要贸然减小 `system2.min_interval_s`：Gemini 实际是慢且贵的。
- 不要删除 ChangeDetector：无变化时跳推理是主要成本优化。
- 不要让 `NO_CHANGE` 无限循环：连续 5 次强制进 S2 是死循环保护。
- StuckMonitor 的 **A-B 循环检测** 不能去掉：S1 误判时经常在两个坐标往复横跳，单纯"全相同"判不出来。

---

## 8. 观测性

三个主要输出：

1. **`debug/autoplay.log`**：全局日志，setup_logger 自动配置文件 handler。
2. **`debug/screenshots/<game>/round_NNN_HHMMSS.{png,txt}`**：每轮截图 + AI 原始响应（`ScreenCapture.save_debug_screenshot`）。
3. **`PerformanceStats` 终局报表**：总帧数、avg screenshot/AI ms、平均 confidence、动作类型分布、错误类型分布。

---

## 9. 添加新游戏的 checklist

1. `games/<new_game>/` 下建：
   - `game.py`：一个继承 `BaseGame` 的 `<NewGame>Game` 类
   - `prompts.py`：至少 `get_analysis_prompt`
   - `__init__.py`（可空）
2. `config/games/<new_game>.yaml`：至少包含 `name` / `vision.top_offset/bottom_offset`
3. `game.py.__init__` 必须设：`self.top_offset`, `self.bottom_offset`, `self.resize_width`, `self.use_resize`
4. 实现三个必要方法（build_prompt / parse_ai_response / execute_action）
5. 若需重启，实现 `is_game_over` + `on_game_over`
6. 若需场景分支，实现 `identify_scene`
7. `--dry-run` 先跑通推理和坐标，再开真实点击
8. 复杂游戏参考 endless_winter：加 ChangeDetector（成本）、SceneClassifier（分支）、StuckMonitor（容错），配合 `analyze_with_retry` + `_last_parsed_action` 缓存模式

---

## 10. 技术债与改进建议

（仅基于代码观察，未经立项）

1. **Gemini SDK**：当前直接 HTTP，手动处理 429 和重试。迁到官方 `google.genai` 可省代码。
2. **默认模型版本字符串**：`config_loader.py` 默认 `gemini-2.0-flash`，`settings.yaml` 写 `gemini-flash-latest`。文件系统里会生成过时默认，读者容易困惑。
3. **`pyautogui` 依赖仍在 `requirements.txt`**：实际已全切 cliclick，可以移除。
4. **`local_api_url` 默认硬编码 IP**（`http://192.168.1.156:1234`）：应走环境变量 / yaml。
5. **endless_winter game.py 1000+ 行**：System 1 / System 2 / snap 启发式 / 状态管理混在一起，可拆分到 `system1.py`, `system2.py`, `cv_snap.py`。
6. **prompts 文件 500 行**：包含大量中文长 prompt 字符串 + 解析器，可拆成 `prompts/` 子包。
7. **没有单元测试**：`ChangeDetector.compute_mse`、`parse_qwen_ref_bbox`、`StuckMonitor.is_stuck` 都是纯函数，写 pytest 容易且有价值。
8. **首次启动自动创建 `settings.yaml` 但只建基础结构**，随后 `load_game_config` 又会抛错，初学者体验不统一。
