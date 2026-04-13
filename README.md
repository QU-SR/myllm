# myllm

便携式本地大模型命令行工具，基于 Go + llama.cpp，支持文本与视觉模型的交互式聊天。

## 功能特性

- **便携布局** — 所有文件通过相对路径组织，可放在 U 盘或任意目录运行
- **双击即用** — 无参数启动时自动进入交互式聊天界面，Windows 下双击 `myllm.exe` 即可
- **文本 + 图片** — 支持纯文本对话和图片理解（需视觉模型 + mmproj）
- **可编辑系统提示词** — 直接编辑 `config/system_prompt.txt` 自定义助手行为
- **CPU 优先** — 默认使用 CPU 推理，无需 GPU；也可通过外部 `llama-server` 接入 GPU
- **模型常驻** — 可选将模型常驻内存，空闲后自动释放，兼顾速度与资源
- **Shell 工具** — 可选启用，让模型通过 `<shell>...</shell>` 标签执行本地命令
- **流式输出** — 实时逐 token 显示生成内容
- **Ollama 兼容** — 支持 `myllm pull <名称>` 从 Ollama 注册表下载模型

## 快速开始

```bash
# 检查环境状态
myllm doctor

# 下载模型（支持 Ollama 名称、GGUF 直链、本地文件）
myllm pull tiny --path ./gemma-3-1b-it-q4.gguf
myllm pull vision --path ./qwen2.5-vl-3b.gguf --mmproj ./mmproj.gguf --vision
myllm pull gemma4:e2b

# 查看已安装模型
myllm ls

# 启动交互式聊天
myllm

# 单次提问
myllm run tiny "你好"
myllm run vision --image ./cat.jpg "描述这张图片"
```

## 命令一览

| 命令 | 说明 |
|------|------|
| `myllm` | 启动交互式聊天界面 |
| `myllm chat [model]` | 启动交互式聊天（可指定模型） |
| `myllm pull <名称或URL>` | 下载模型到 `models/` 目录 |
| `myllm ls` | 列出已安装的模型 |
| `myllm info <名称>` | 查看模型详细信息 |
| `myllm rm <名称> [--delete-files]` | 移除模型（加 `--delete-files` 同时删除文件） |
| `myllm run <名称> [提示文本]` | 单次运行模型 |
| `myllm doctor` | 检查运行环境（后端、模型数量等） |
| `myllm bench` | 输出推荐的 CPU 线程数 |
| `myllm help` | 显示帮助信息 |

### run 命令参数

```
myllm run <名称> [提示文本] [选项]
  --image FILE         附加图片文件（可重复使用）
  --system-file FILE   指定系统提示词文件
  --interactive        交互式聊天模式
  --threads N          CPU 线程数
  --context N          上下文长度
  --temp F             温度（默认 0.7）
  --top-p F            Top-p 采样（默认 0.95）
```

## 交互式界面

运行 `myllm`（无参数）或双击 `myllm.exe` 进入交互式聊天。

### 斜杠命令

| 命令 | 说明 |
|------|------|
| `/help` | 查看全部命令 |
| `/model` | 选择当前模型 |
| `/pull` | 下载模型 |
| `/performance` | 配置性能预设、常驻策略和释放时间 |
| `/shell` | 切换 Shell 工具开关 |
| `/clear` | 清空当前对话 |
| `/system` | 切换系统提示词文件 |
| `/threads` | 设置 CPU 线程数 |
| `/context` | 设置上下文长度 |
| `/temp` | 设置温度 |
| `/top-p` | 设置 Top-p |
| `/stats` | 查看当前运行状态 |
| `/doctor` | 检查环境状态 |
| `/exit` | 退出程序 |

### 性能优化预设

通过 `/performance` 命令可选择：

- **极速响应** — 更多线程，上下文 4096，偏重速度
- **均衡** — 日常推荐配置
- **省内存** — 较少线程，上下文 2048

还可配置模型常驻策略（按需加载 / 常驻内存）和空闲释放时间（5/15/30/60 分钟）。

## 自定义系统提示词

直接编辑 `config/system_prompt.txt`，或通过命令行指定：

```bash
myllm run tiny --system-file ./prompts/code-review.txt "审查这段代码"
```

在交互模式中使用 `/system` 命令切换提示词文件。

## 目录结构

```
myllm/
├── myllm.exe            # 主程序
├── main.go              # 主源码
├── tui.go               # 交互式 TUI 源码
├── go.mod / go.sum      # Go 依赖
├── README.md
├── backends/            # llama.cpp 后端
│   ├── llama-server.exe
│   ├── llama.dll
│   └── ggml-*.dll
├── models/              # 模型文件（.gguf）
├── config/              # 配置文件
│   ├── config.json
│   └── system_prompt.txt
├── cache/               # 下载缓存
├── logs/                # 运行日志
└── tmp/                 # 临时文件
```

## 运行时后端

将 `llama-server`（Linux/macOS）或 `llama-server.exe`（Windows）放入 `backends/` 目录。

程序会在每次提问时自动启动后端进程、完成推理后关闭，不会留下常驻后台服务（除非启用了模型常驻模式）。

## 技术栈

- **语言** — Go 1.24
- **TUI 框架** — [Bubble Tea](https://github.com/charmbracelet/bubbletea) + [Bubbles](https://github.com/charmbracelet/bubbles) + [Lipgloss](https://github.com/charmbracelet/lipgloss)
- **推理后端** — [llama.cpp](https://github.com/ggml-org/llama.cpp)（通过 `llama-server` OpenAI 兼容 API）
- **模型格式** — GGUF

## 许可证

MIT
