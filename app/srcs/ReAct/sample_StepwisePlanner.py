import os
import asyncio
import time
from openai import OpenAI

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

# class WebSearchEnginePlugin:
#     """
#     A search engine plugin.
#     """

#     from semantic_kernel.orchestration.kernel_context import KernelContext
#     from semantic_kernel.plugin_definition import (
#         kernel_function,
#         kernel_function_context_parameter,
#     )

#     def __init__(self, connector) -> None:
#         self._connector = connector

#     @kernel_function(description="Performs a web search for a given query", name="searchAsync")
#     @kernel_function_context_parameter(
#         name="query",
#         description="The search query",
#     )
#     async def search(self, query: str, context: KernelContext) -> str:
#         query = query or context.variables.get("query")
#         result = await self._connector.search(query, num_results=5, offset=0)
#         return str(result)

# from semantic_kernel.connectors.search_engine import BingConnector

# BING_API_KEY = os.getenv("BING_API_KEY")
# connector = BingConnector(BING_API_KEY)


# カーネルの初期化
kernel = sk.Kernel()

api_key = os.getenv("OPENAI_API_KEY")


kernel.add_chat_service(
        "openai",
        OpenAIChatCompletion(ai_model_id="gpt-4", api_key=api_key),
    )

kernel.import_plugin(TimePlugin(), "time")
kernel.import_plugin(MathPlugin(), "math")

kernel.import_plugin(WebSearchEnginePlugin(connector), plugin_name="WebSearch")

# codeinterpreterの初期化



# class codeinterpreter:
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

#     @kernel_function(description="Generates code for the specified query and executes code.", name="codeInterpreter")
#     @kernel_function_context_parameter(
#         name="query",
#         description="Generate and execute Code",
#     )
#     async def codeinterpreter(self, query: str,context: KernelContext) -> str:
#         query = query or context.variables.get("query")[1]
#         result = await self._tmp.search()
#         return str(result)






# 設定
planner = StepwisePlanner(kernel, StepwisePlannerConfig(max_iterations=100, min_iteration_time_ms=1500,max_tokens=4000))

ask = "しつもん"
plan = planner.create_plan(goal=str(ask))

async def main():
    # 非同期コード
    result = await plan.invoke()


    # 中身何しているか見れるよってやつ
    for index, step in enumerate(plan._steps):
        print("Step:", index)
        print("Description:", step.description)
        print("Function:", step.plugin_name + "." + step._function.name)
        if len(step._outputs) > 0:
            print("  Output:\n", str.replace(result[step._outputs[0]], "\n", "\n  "))

    print("##########result##########")
    print(result)

asyncio.run(main())


