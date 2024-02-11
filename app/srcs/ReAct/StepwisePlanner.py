import os
import asyncio

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    AzureChatCompletion,
)
from semantic_kernel.planning import StepwisePlanner
from semantic_kernel.planning.stepwise_planner.stepwise_planner_config import (
    StepwisePlannerConfig,
)

from semantic_kernel.core_plugins.math_plugin import MathPlugin
from semantic_kernel.core_plugins.time_plugin import TimePlugin


# 初期化
kernel = sk.Kernel()

api_key = os.getenv("OPENAI_API_KEY")

kernel.add_chat_service(
        "openai",
        OpenAIChatCompletion(ai_model_id="gpt-4", api_key=api_key),
    )

kernel.import_plugin(TimePlugin(), "time")
kernel.import_plugin(MathPlugin(), "math")


# class expPlugin:
#     """
#     assistants api

#     ここでいうtmpはクライアントみたいなもの？
#     """
#     from semantic_kernel.orchestration.kernel_context import KernelContext
#     from semantic_kernel.plugin_definition import (
#         kernel_function,
#         kernel_function_context_parameter,
#     )
#     def __init__(self,tmp) -> None:
#         self._tmp = tmp

#     @kernel_function(description="code interpreter", name="codeInterpreter")
#     @kernel_function_context_parameter(
#         name="code",
#         description="code interpreter",
#     )
#     async def codeInterpreter(self, query: str,context: KernelContext) -> str:
#         query = query or context.variables.get("query")[1]
#         result = await self._tmp.search()
#         return str(result)

# 設定
planner = StepwisePlanner(kernel, StepwisePlannerConfig(max_iterations=10, min_iteration_time_ms=1000))

ask = """
The following content consists of pre-modification data of a supplier's supply plan, serving as reference data. Please detect areas that need to be revised based on this reference data. Then, gather information from the reference data and consider calculating averages, among other methods, to propose a robust revision plan.

Processing Flow:
1. Contemplate the necessary actions and their reasons to generate an output for the task.
2. Act based on this contemplation, and then think again about what actions and reasons are needed next, based on the results obtained.
3. Repeat the thinking process from steps 1 to 2, and conclude once a final answer is generated.
   ---
   pre-modification data：
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
   reference data：
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
plan = planner.create_plan(goal=ask)

async def main():
    # 非同期コード
    result = await plan.invoke()

    print(result)

    # 中身何しているか見れるよってやつ
    for index, step in enumerate(plan._steps):
        print("Step:", index)
        print("Description:", step.description)
        print("Function:", step.plugin_name + "." + step._function.name)
        if len(step._outputs) > 0:
            print("  Output:\n", str.replace(result[step._outputs[0]], "\n", "\n  "))

asyncio.run(main())


