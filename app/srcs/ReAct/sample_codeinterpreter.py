import os
import time
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

assistant = client.beta.assistants.create(
  instructions="You are a personal math tutor. When asked a math question, write and run code to answer the question.",
  model="gpt-4-turbo-preview",
  tools=[{"type": "code_interpreter"}]
)

# スレッドの準備
thread = client.beta.threads.create()

task = """
以下の平均をもとめてください
---
7808円
8855円
8169円
7471円
8997円
7007円
8243円
1753円
3238円
9135円
"""
# ユーザーメッセージの追加
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=task
)

# スレッドのメッセージリストの確認
messages = client.beta.threads.messages.list(
    thread_id=thread.id,
    order="asc"
)
for message in messages:
    print(message.role, ":", message.content[0].text.value)

# アシスタントにリクエスト
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

completed = False
while not completed:
    # ステータスの取得
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    print("run.status:", run.status)
    # ステータスが 'completed' かどうかをチェック
    if run.status == 'completed':
        completed = True
    else:
        # ステータスが 'completed' ではない場合、少し待つ
        time.sleep(5)

# スレッドのメッセージリストの確認
messages = client.beta.threads.messages.list(
    thread_id=thread.id,
    order="asc"
)
for message in messages:
    print(message.role, ":", message.content[0].text.value)

# # Run Stepの確認
# run_steps = client.beta.threads.runs.steps.list(
#     thread_id=thread.id,
#     run_id=run.id
# )
# print(run_steps)