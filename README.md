# DataDescribeAgent - 智能数据分析助手

> 基于 HelloAgents 框架（`SimpleAgent`）的智能数据分析工具：读取表格 → LLM 清洗 / 统计 / 可视化 / 报告。

## 📝 项目简介

DataDescribeAgent 是一个智能数据分析助手，能够自动分析数据、生成可视化图表（ECharts）、撰写分析报告（Markdown）。
本项目由 Datawhale `hello-agents` 仓库中的 `Co-creation-projects/1zrj-DataAnalysisAgent` 改造而来：

- 把 `main.ipynb` 改造成可直接运行的 `main.py`
- 适配 `hello-agents` 1.0.0 的 `ToolResponse` API
- 修复 `DataStatisticsTool` 漏写 `import numpy` 的隐藏 bug 并补齐工具注册
- **默认使用 DeepSeek** 作为 LLM（OpenAI 兼容接口，支持 function calling）

### 核心功能

- ✅ 数据清洗：调用 `data_cleaner` 工具按规则清洗表格数据（列筛选、去空、补零）
- ✅ 数据统计：调用 `data_statistics` 工具做描述性统计（均值 / 中位数 / 标准差 / 分位数 / 类别分布）
- ✅ 可视化：LLM 生成 ECharts 代码，抽取并渲染全部图表到 `output/echarts.html`
- ✅ 报告生成：输出 Markdown 格式的 `output/report.md`

## 🛠️ 技术栈

- HelloAgents 框架（`SimpleAgent` + 自定义 `Tool`）
- DeepSeek（`deepseek-chat`，OpenAI 兼容接口）
- pandas / numpy（数据处理）
- ECharts（前端可视化）

## 🚀 快速开始

### 1. 安装依赖（建议使用虚拟环境）

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. 配置 LLM（推荐用 .env）

```bash
cp .env.example .env          # Windows:  copy .env.example .env
# 然后编辑 .env，填入你的 DeepSeek Key
```

`.env` 示例（默认即 DeepSeek）：

```ini
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=你的_deepseek_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TIMEOUT=60
```

> DeepSeek Key 申请：https://platform.deepseek.com
> 想换其它 OpenAI 兼容服务（OpenAI / 硅基流动 / 本地 Ollama 等）？只改 `.env` 里的三个值即可，`main.py` 无需改动。

### 3. 运行

```bash
python main.py
```

运行后会在 `output/` 下生成 `echarts.html` 与 `report.md`。

## 📂 项目结构

```
DataDescribeAgent/
├── README.md           # 项目说明
├── requirements.txt    # 依赖列表
├── .gitignore          # 忽略 .env / .venv
├── .env.example        # 环境变量示例（占位符）
├── .env                # 你的真实密钥（本地用，不提交）
├── main.py             # 主程序（由 main.ipynb 改造）
├── data/
│   └── simple_data.xls # 示例数据
└── output/
    ├── echarts.html    # 生成的图表
    └── report.md       # 生成的报告
```

## 🔧 技术实现

### 工具系统

1. **DataCleaningTool**：数据清洗工具 —— 基于规则清洗表格数据
2. **DataStatisticsTool**：数据统计工具 —— 提供描述性统计分析（容错解析多种输入结构）

### 智能体设计

使用 HelloAgents 的 `SimpleAgent`，配合自定义工具与 system prompt 完成「清洗 → 统计 → 可视化 → 报告」的流水线。

## 🙏 致谢

感谢 Datawhale 社区与 Hello-Agents 项目！
