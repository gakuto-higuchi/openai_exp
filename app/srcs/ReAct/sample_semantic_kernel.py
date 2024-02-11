import os
import asyncio

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    AzureChatCompletion,
)


# 初期化
kernel = sk.Kernel()

api_key = os.getenv("OPENAI_API_KEY")

kernel.add_chat_service(
        "openai",
        OpenAIChatCompletion(ai_model_id="gpt-4", api_key=api_key),
    )

ask = """
以下の内容は、サプライヤーの供給計画データの修正前データ、参考データである。参考データを参考に、修正しなければならない箇所を検出し、修正案を提案してください
---
修正前データ：
Date,Product,Quantity,Region,Cost,Supplier
      2024-01-01,Product A,100,North,500,Supplier X
      2024-01-01,Product B,150,South,750,Supplier Y
      2024-01-01,Product C,200,West,900,Supplier Z
      2024-01-02,Product A,120,East,600,Supplier Z
      2024-01-02,Product B,130,West,650,Supplier X
      2024-01-02,Product C,210,North,920,Supplier Y
      2024-01-03,Product A,140,South,700,Supplier Y
      2024-01-03,Product B,120,East,610,Supplier Z
      2024-01-03,Product C,200,North,890,Supplier X
      2024-01-04,Product A,130,West,650,Supplier X
      2024-01-04,Product B,140,North,720,Supplier Y
      2024-01-04,Product C,190,East,850,Supplier Z

---
参考データ：
Date,Product,Original_Quantity,Modified_Quantity,Reason,Market_Trend,Competitor_Activity
      2023-12-01,Product A,100,110,Demand Increase,Upward,New Competitor Entry
      2023-12-02,Product B,150,140,Supply Constraint,Downward,Competitor Discount
      2023-12-03,Product C,200,190,Overstock,Downward,Competitor Launch
      2023-12-04,Product A,120,130,New Market,Stable,Competitor Stock Shortage
      2023-12-05,Product B,130,135,Event Promotion,Upward,No Change
      2023-12-06,Product C,210,205,Market Prediction,Stable,New Competitor Entry
      2023-12-07,Product A,140,145,Seasonal Demand,Upward,Competitor Discount
      2023-12-08,Product B,120,115,Supply Overestimation,Downward,Competitor Launch
      2023-12-09,Product C,200,195,Stock Adjustment,Stable,Competitor Stock Shortage
      2023-12-10,Product A,130,135,Market Analysis,Upward,New Competitor Entry
      2023-12-11,Product B,140,138,Supply Chain Issue,Downward,Competitor Discount
      2023-12-12,Product C,190,185,Inventory Optimization,Stable,Competitor Launch

"""


# プラグインの設定
from semantic_kernel.core_plugins.text_plugin import TextPlugin

plugins_directory = "/root/app/srcs/samples/plugins"
summarize_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory, "SummarizePlugin")
writer_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory, "WriterPlugin")
text_plugin = kernel.import_plugin(TextPlugin(), "TextPlugin")
code_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory, "CodingPlugin")

sk_prompt="""
{{$input}}

Below, please detect any grammatical, logical, or factual errors in the text and provide a detailed explanation for each error.
If no errors are found, please state so
"""
detectFunction = kernel.create_semantic_function(
    prompt_template=sk_prompt,
    function_name="detection",
    plugin_name="DetectPlugin",
    max_tokens=2000,
    temperature=0,
)

# Plannerの設定
from semantic_kernel.planning.basic_planner import BasicPlanner

planner = BasicPlanner()


async def main():
    # あなたの非同期コード
    basic_plan = await planner.create_plan(ask, kernel)
    # 他の非同期コードがあればここに追加
    print(basic_plan.generated_plan)

    results = await planner.execute_plan(basic_plan, kernel)

    print(results)

asyncio.run(main())
