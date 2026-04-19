# 微信小游戏 AI 自动化框架

基于视觉语言模型（VLM）的 macOS 微信小游戏自动化框架。通用 `GameLoop` 驱动"截图 → 推理 → 执行"主循环，每款游戏通过实现 `BaseGame` 适配器接入。

当前支持：

| 游戏 | 类型 | AI 后端 |
|------|------|---------|
| 无尽冬日 (`endless_winter`) | 多动作 UI 操作 | 双系统（本地 Qwen + Gemini 升级） |

## 快速开始

```bash
# 1. 环境
python3 -m venv venv && source venv/bin/activate
pip3 install -r requirements.txt
brew install cliclick          # 必需：pyautogui 会被微信静默丢弃

# 2. API Key（使用云端模型时需要）
echo "GEMINI_API_KEY=你的key" > .env

# 3. 运行
python3 main.py --game endless_winter --dry-run    # 只分析不操作
python3 main.py --game endless_winter              # 正式运行
```

准备步骤：打开微信 → 进入对应小游戏 → 运行脚本。

## 常用命令

```bash
python3 main.py --game endless_winter --max-rounds 50     # 最多 50 轮
python3 main.py --game endless_winter --log-level DEBUG   # 详细日志
python3 main.py --game endless_winter --use-gemini        # 用 Gemini 云端（默认走本地 Qwen）
```

停止：`Ctrl+C`。

## 调试

- 日志：`tail -f debug/autoplay.log`
- 每轮截图 + AI 响应：`debug/screenshots/<game>/round_NNN_HHMMSS.{png,txt}`

## 深入阅读

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 架构全貌、模块详解、踩坑总结、扩展新游戏 checklist
- [`CLAUDE.md`](CLAUDE.md) — Claude Code 使用时的上下文指引

## 添加新游戏

简要步骤（细节见 ARCHITECTURE.md §10）：

1. `games/<new_game>/game.py` 继承 `BaseGame`，实现 `build_prompt` / `parse_ai_response` / `execute_action`
2. `config/games/<new_game>.yaml` 至少包含 `name` 和 `vision` 字段
3. `python3 main.py --game <new_game> --dry-run` 先验证推理
4. 开启真实点击

## 许可证

MIT
