"""
main.py —— 由 main.ipynb 改造而来的可直接运行脚本
============================================================
DataDescribeAgent: 基于 HelloAgents 框架的智能数据分析助手
  - 读取 data/simple_data.xls
  - 调用 LLM（默认 DeepSeek，可改 .env 切换其它 OpenAI 兼容模型）做清洗 / 统计 / 可视化 / 报告
  - 输出 output/echarts.html（图表）与 output/report.md（分析报告）

相对原 notebook 的改进:
  1. 配置从 .env 读取（参考 .env.example），不再把 API Key 写死在代码里
  2. 修复了 DataStatisticsTool 漏写 `import numpy as np` 的隐藏 NameError bug
  3. 路径基于脚本所在目录计算，VSCode / 任意 cwd 下都能正确找到 data 与 output
  4. 增加了缺失匹配时的兜底处理，避免正则抽不到内容直接崩溃

运行:
    python main.py
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from dotenv import load_dotenv

from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse


# ==============================
# 0. 路径与 LLM 配置
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "simple_data.xls")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 加载 .env（若存在）。LLM_API_KEY 必须来自这里或真实环境变量
load_dotenv(os.path.join(BASE_DIR, ".env"))

# LLM 参数：环境变量优先；缺失时用默认值（base_url / model 可改，key 必须提供）
# 默认使用 DeepSeek（OpenAI 兼容接口）；可在 .env 中改为任意 OpenAI 兼容服务
os.environ.setdefault("LLM_MODEL_ID", "deepseek-chat")
os.environ.setdefault("LLM_BASE_URL", "https://api.deepseek.com/v1")
os.environ.setdefault("LLM_TIMEOUT", "60")

if not os.environ.get("LLM_API_KEY"):
    raise SystemExit(
        "❌ 未检测到 LLM_API_KEY。请在项目根目录创建 .env 文件并填入你的 API Key（默认 DeepSeek），"
        "参考 .env.example。\n"
        "   复制命令:  cp .env.example .env   # Windows:  copy .env.example .env"
    )

print("✅ 库导入和配置完成")


# ==============================
# 1. 数据分析工具
# ==============================
class DataCleaningTool(Tool):
    """数据清洗工具 - 基于用户指定规则清洗表格数据"""

    def __init__(self):
        super().__init__(
            name="data_cleaner",
            description="对传入的表格数据执行清洗操作，包括去空值、列筛选等",
        )

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        data_json = parameters.get("data_json")
        if not data_json:
            return ToolResponse.success(text="错误：缺少原始数据（data_json 不能为空）")

        try:
            raw_data = json.loads(data_json)
            records = raw_data.get("完整数据", [])
            if not records:
                return ToolResponse.success(text="警告：原始数据为空，无法清洗")

            df = pd.DataFrame(records)

            # 1. 列筛选
            columns_to_keep = parameters.get("columns_to_keep")
            if columns_to_keep:
                missing_cols = [col for col in columns_to_keep if col not in df.columns]
                if missing_cols:
                    return ToolResponse.success(
                        text=f"错误：指定保留的列不存在：{missing_cols}"
                    )
                df = df[columns_to_keep]

            # 2. 删除空值行 + 补零
            if parameters.get("drop_na", False):
                df = df.dropna()
            df = df.fillna(0)

            cleaned_records = df.where(pd.notnull(df), None).to_dict(orient="records")
            result = {"clean_data": cleaned_records}
            return ToolResponse.success(
                text=json.dumps(result, ensure_ascii=False, indent=2),
                data=result,
            )

        except json.JSONDecodeError:
            return ToolResponse.success(text="错误：data_json 不是有效的 JSON 格式")
        except Exception as e:
            return ToolResponse.error(code="CLEAN_ERROR", message=f"清洗过程中出错：{str(e)}")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="data_json", type="string", description="原始数据的 JSON 字符串", required=True),
            ToolParameter(name="drop_na", type="boolean", description="是否删除包含空值的行", required=False),
            ToolParameter(name="columns_to_keep", type="array", description="要保留的列名列表", required=False),
        ]


def _extract_records(raw: Any) -> List[dict]:
    """从多种可能的 JSON 结构里容错地提取记录列表。

    - 本身就是 list -> 直接返回
    - dict 含 clean_data / 完整数据 / data 等键 -> 取其中的 list
    - 其它 dict -> 视为单条记录返回
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("clean_data", "完整数据", "data", "records", "rows"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        return [raw]
    return []


class DataStatisticsTool(Tool):
    """数据统计工具 - 提供描述性统计分析（注意：依赖 numpy，已在文件顶部 import）"""

    def __init__(self):
        super().__init__(
            name="data_statistics",
            description="对数据进行描述性统计分析，包括均值、中位数、标准差等",
        )

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        data_json = parameters.get("data_json")
        if not data_json:
            return ToolResponse.success(text="错误：缺少数据（data_json 不能为空）")

        # 1) 解析输入（JSON 解析失败单独友好返回，不进下面的 except）
        try:
            raw_data = json.loads(data_json)
        except (json.JSONDecodeError, TypeError):
            return ToolResponse.success(text="错误：data_json 不是有效的 JSON 格式")

        # 2) 容错提取记录：兼容 data_cleaner 的 {"clean_data":[...]}
        #    与原始 {"完整数据":[...]}，以及裸露的数组 / 单条记录
        records = _extract_records(raw_data)
        if not records:
            return ToolResponse.success(
                text="警告：未能从输入中解析出数据记录。请确认 data_json 传入的是 "
                     "data_cleaner 工具返回的完整 JSON 字符串（其内部含 clean_data 字段）。"
            )

        try:
            df = pd.DataFrame(records)

            numeric_stats = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                numeric_stats[col] = {
                    "count": int(df[col].count()),
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75)),
                }

            categorical_stats = {}
            for col in df.select_dtypes(include=["object", "string"]).columns:
                value_counts = df[col].value_counts().head(10).to_dict()
                categorical_stats[col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": value_counts,
                }

            result = {
                "shape": f"{len(df)} 行, {len(df.columns)} 列",
                "numeric_stats": numeric_stats,
                "categorical_stats": categorical_stats,
            }
            print(f"📊 data_statistics 已执行: {result['shape']}（数值列 {len(numeric_stats)} 个，类别列 {len(categorical_stats)} 个）")
            return ToolResponse.success(
                text=json.dumps(result, ensure_ascii=False, indent=2),
                data=result,
            )

        except Exception as e:
            return ToolResponse.error(code="STAT_ERROR", message=f"统计分析出错：{str(e)}")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="data_json", type="string", description="数据的 JSON 字符串", required=True),
        ]


# ==============================
# 2. 创建智能体
# ==============================
SYSTEM_PROMPT = """你是一名专业数据分析师，请严格按以下步骤完成任务：

1. 调用 data_cleaner 工具清洗数据，得到清洗结果（JSON 结构，包含 clean_data 字段）；
2. 调用 data_statistics 工具做统计，其 data_json 参数**必须**传入上一步 data_cleaner 返回的完整结果字符串；
3. 根据数据选择合适的图表，用 ECharts 代码进行可视化。每个图表用独立的代码块包裹，格式为：
```echarts
option = {
    // 这里写 ECharts 配置
};
```
你可以输出多个图表（如趋势折线图、对比柱状图、热力图等）；
4. 最后输出一份数据分析报告，报告**必须**以一级标题 `# 数据分析报告` 开头，并包含以下小节：
   - 分析背景与目标
   - 关键发现
   - 统计计算、趋势识别、异常检测或对比分析（结论必须基于数据，避免主观臆断）
   - 总结与建议

要求：
- 报告使用 Markdown 格式；
- 不要输出你的思考过程、不要调用额外的笔记类工具、不要输出独立 HTML 文件；
- 只输出 ECharts 代码块（可多个）和最后的 Markdown 报告。
"""


def build_agent() -> SimpleAgent:
    tool_registry = ToolRegistry()
    tool_registry.register_tool(DataCleaningTool())
    tool_registry.register_tool(DataStatisticsTool())

    agent = SimpleAgent(
        name="数据分析助手",
        llm=HelloAgentsLLM(),
        system_prompt=SYSTEM_PROMPT,
        tool_registry=tool_registry,
    )
    print("✅ 智能体创建完成")
    print(f"✅ 可用工具: {list(tool_registry._tools.keys())}")
    return agent


# ==============================
# 3. 读取示例数据
# ==============================
def read_data() -> str:
    try:
        df = pd.read_excel(DATA_PATH, engine="xlrd")
        # 不做清洗！保留原始 NaN（pandas 会将 Excel 空单元格转为 NaN）
        data_records = df.to_dict(orient="records")
        clean_input = {"完整数据": data_records}
        sample_data = json.dumps(clean_input, ensure_ascii=False, indent=2)
        print(f"✅ 已读取数据: {DATA_PATH}（{len(data_records)} 行）")
        return sample_data
    except FileNotFoundError:
        return json.dumps({"error": f"Excel 文件不存在: {DATA_PATH}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"读取 Excel 文件失败: {str(e)}"}, ensure_ascii=False)


# ==============================
# 4. 保存报告与图表
# ==============================
def save_outputs(result: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4.1 抽取所有 ECharts 代码块（支持多图）
    echarts_blocks = re.findall(r"option\s*=\s*(\{[\s\S]*?\});", result)
    if echarts_blocks:
        print(f"✅ 提取到 {len(echarts_blocks)} 个 ECharts 图表")
    else:
        echarts_blocks = ["{}"]
        print("⚠️ 未从结果中找到 ECharts 代码，html 将使用空配置")

    # 4.2 抽取 Markdown 报告
    report_match = re.search(r"(# 数据分析报告[\s\S]*)", result)
    if report_match:
        markdown_report = report_match.group(1).strip()
        print("✅ 提取到 Markdown 报告")
    else:
        markdown_report = result.strip()
        print("⚠️ 未找到以 '# 数据分析报告' 开头的报告，已保存完整返回内容作为兜底")

    # 4.3 保存 Markdown
    md_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"📄 Markdown 报告已保存: {md_path}")

    # 4.4 保存 ECharts HTML（渲染全部图表）
    HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DataDescribeAgent - ECharts</title>
    <script src="https://cdn.staticfile.org/echarts/4.3.0/echarts.min.js"></script>
    <style>body{{font-family:-apple-system,Segoe UI,sans-serif;background:#fafafa;padding:16px;}}</style>
</head>
<body>
<div style="text-align:center;color:#555;margin-bottom:12px;">数据分析结果（共 {count} 张图表）</div>
{divs}
    <script type="text/javascript">
{inits}
    </script>
</body>
</html>
"""
    divs = []
    inits = []
    for i, code in enumerate(echarts_blocks):
        div_id = f"chart_{i}"
        divs.append(
            f'    <div id="{div_id}" style="width:920px;height:520px;'
            f'margin:0 auto 28px;"></div>'
        )
        chart_js = (
            "    var c" + str(i) + " = echarts.init(document.getElementById('" + div_id + "'));\n"
            "    var option" + str(i) + " = " + code + ";\n"
            "    c" + str(i) + ".setOption(option" + str(i) + ");\n"
        )
        inits.append(chart_js)
    html_code = HTML_TEMPLATE.format(
        count=len(echarts_blocks),
        divs="\n".join(divs),
        inits="\n".join(inits),
    )
    html_path = os.path.join(OUTPUT_DIR, "echarts.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    print(f"📊 ECharts 图表已保存: {html_path}")


# ==============================
# 主流程
# ==============================
def main() -> None:
    agent = build_agent()
    sample_data = read_data()

    print("\n=== 开始数据分析 ===")
    result = agent.run(f"对以下数据绘制图表和数据分析\n\n{sample_data}\n")
    print("\n=== LLM 返回结果 ===\n")
    print(result)

    save_outputs(result)
    print("\n✅ 全部完成。打开 output/echarts.html 查看图表，output/report.md 查看报告。")


if __name__ == "__main__":
    main()
