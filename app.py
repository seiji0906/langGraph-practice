from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import END, MessageGraph
import streamlit as st
import json
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from dotenv import load_dotenv
import os
from typing import List, Dict

load_dotenv()  # .envファイルから環境変数を読み込む
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ツール関数の定義（multiply関数は単純で、引数二つの数値の掛け算を実施）
@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number

# 言語モデルの設定
model = ChatOpenAI(temperature=0, streaming=True)
model_with_tools = model.bind(tools=[convert_to_openai_tool(multiply)])

# メッセージグラフの作成
graph = MessageGraph()

# モデルを呼び出すノードの定義
def invoke_model(state: List[BaseMessage]):
    return model_with_tools.invoke(state)

graph.add_node("oracle", invoke_model)


# ツール関数を呼び出すノードの定義
# ユーザーのプロンプトから言語モデルがツール関数が必要かどうかを判断し、tool_callsを生成する
# tool_callsの有無により、tool関数を呼び出すか否かを条件分岐している。tool_callsがない場合は、tool関数は呼び出さない。
def invoke_tool(state: List[BaseMessage]):
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    add_call = None
    multiply_call = None

    for tool_call in tool_calls:
        if tool_call.get("function").get("name") == "add":
            add_call = tool_call
        elif tool_call.get("function").get("name") == "multiply":
            multiply_call = tool_call

    if add_call is not None:
        res = add.invoke(
            json.loads(add_call.get("function").get("arguments"))
        )
        return ToolMessage(
            tool_call_id=add_call.get("id"),
            content=res
        )
    elif multiply_call is not None:
        res = multiply.invoke(
            json.loads(multiply_call.get("function").get("arguments"))
        )
        return ToolMessage(
            tool_call_id=multiply_call.get("id"),
            content=res
        )
    else:
        return None  # ツール関数が呼び出されない場合はNoneを返す

graph.add_node("multiply", invoke_tool)

# ノード間のエッジの定義
graph.add_edge("multiply", END)

# エントリーポイントの設定
graph.set_entry_point("oracle")

# ルーター関数の定義
def router(state: List[BaseMessage]):
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if len(tool_calls):
        return "multiply"
    else:
        return "end"

# 条件付きエッジの定義
graph.add_conditional_edges("oracle", router, {
    "multiply": "multiply",
    "end": END,
})

# グラフのコンパイル
runnable = graph.compile()

# Streamlitアプリケーションの作成
st.title("LangGraph Demo")

# ユーザー入力の取得
user_input = st.text_input("質問を入力してください:")
if user_input:
    # ストリーミング出力用のコンテナを作成
    response_container = st.empty()
    agent_response = ""  # agent_responseを初期化する位置を変更

    # エージェントからの返答を取得
    for message in runnable.invoke(HumanMessage(user_input)):
        if isinstance(message, ToolMessage):
            # ツール関数の結果を取得し、エージェントの返答に追加
            tool_result = message.content
            agent_response += f"ツールの結果: {tool_result}\n"
        elif isinstance(message, BaseMessage):
            # エージェントの返答を追加
            agent_response += message.content
        
        # 更新された返答をストリーミング出力
        response_container.write(agent_response)
