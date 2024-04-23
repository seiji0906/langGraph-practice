from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from dotenv import load_dotenv
import os
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
import streamlit as st
import time

load_dotenv()  # .envファイルから環境変数を読み込む
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tools = [TavilySearchResults(max_results=1)]

tool_executor = ToolExecutor(tools)

model = ChatOpenAI(temperature=0, streaming=True)


# format_tool_to_openai_functionは、langchain0.2で削除される（現在0.1.16）
# LangChainDeprecationWarning: The function `format_tool_to_openai_function` was deprecated in LangChain 0.1.16 and will be removed in 0.2.0. Use langchain_core.utils.function_calling.convert_to_openai_function() instead.
functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)
# print(model)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# 継続するか否かを判定する関数を定義
def should_continue(state):
    print(f"should_continue state: {state}")
    messages = state['messages']
    last_message = messages[-1]
    # 関数呼び出しがない場合は終了
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # それ以外の場合は継続
    else:
        return "continue"

# モデルを呼び出す関数を定義
def call_model(state):
    print(f"call_model state: {state}")
    messages = state['messages']
    response = model.invoke(messages)
    # リストを返すのは、既存のリストに追加されるためです
    return {"messages": [response]}

# ツールを実行する関数を定義
def call_tool(state):
    print(f"call_tool state: {state}")
    messages = state['messages']
    # 継続条件に基づいて
    # 最後のメッセージが関数呼び出しを含んでいることがわかっている
    last_message = messages[-1]
    # function_callからToolInvocationを構築
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # tool_executorを呼び出し、応答を取得
    response = tool_executor.invoke(action)
    # 応答を使ってFunctionMessageを作成
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # リストを返すのは、既存のリストに追加されるためです
    return {"messages": [function_message]}



# 新しいグラフを定義
workflow = StateGraph(AgentState)

# 循環する2つのノードを定義
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# エントリーポイントを `agent` に設定
# つまり、このノードが最初に呼び出されるノードになります
workflow.set_entry_point("agent")

# 条件付きエッジを追加します
workflow.add_conditional_edges(
    # 最初に、開始ノードを定義します。`agent` を使用します。
    # つまり、これらはエージェントノードが呼び出された後に取られるエッジです。
    "agent",
    # 次に、次に呼び出されるノードを決定する関数を渡します。
    should_continue,
    # 最後に、マッピングを渡します。
    # キーは文字列で、値は他のノードです。
    # ENDは、グラフが終了することを示す特別なノードです。
    # 実際には、`should_continue` を呼び出し、その出力がこのマッピングのキーと照合されます。
    # どのキーと一致するかに基づいて、そのノードが呼び出されます。
    {
        # `continue` の場合は、ツールノードを呼び出します。
        "continue": "action",
        # それ以外の場合は終了します。
        "end": END
    }
)

# ここで `tools` から `agent` への通常のエッジを追加します。
# つまり、`tools` が呼び出された後、`agent` ノードが次に呼び出されます。
workflow.add_edge('action', 'agent')

# 最後にコンパイルします!
# これにより、LangChain Runnableにコンパイルされます。
# つまり、他の実行可能なものと同じように使用できます。
app = workflow.compile()

#　上記でワークフローが完成

# inputs = {"messages": [HumanMessage(content="東京の天気は？")]}
# output = app.invoke(inputs)

st.title("LangGraph サイクル例")

user_input = st.text_input("質問を入力してください:")

if user_input:
    start_time = time.time()
    input = {
        "messages": [HumanMessage(content=user_input)]
    }
    for output in app.stream(input):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            st.write(f"output from node '{key}")
            print("---")
            st.write("---")
            print(value)
            st.write(value)
        end_time = time.time()
        st.write(f"応答時間: {end_time - start_time}s")
        st.write("\n---\n")
        print("\n---\n")

#TODO
# 応答時間出力


# 「LangGraphとは？」
# という質問に対しては、ツール使って検索したうえで回答した。
