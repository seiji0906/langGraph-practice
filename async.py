from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import asyncio



load_dotenv()  # .envファイルから環境変数を読み込む
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tools = [TavilySearchResults(max_results=1)]

tool_executor = ToolExecutor(tools)

# We will set streaming=True so that we can stream tokens
# See the streaming section for more information on this.
model = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)

functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
async def call_model(state):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
async def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    # We call the tool_executor and get back a response
    response = await tool_executor.ainvoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

# inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
# for output in app.astream(inputs):
#     # stream() yields dictionaries with output keyed by node name
#     for key, value in output.items():
#         print(f"Output from node '{key}':")
#         print("---")
#         print(value)
#     print("\n---\n")

# async def main():
#     inputs = {"messages": [HumanMessage(content="LangGraphについて説明")]}
#     # async for output in app.astream(inputs):
#     #     # stream() yields dictionaries with output keyed by node name
#     #     for key, value in output.items():
#     #         print(f"Output from node '{key}':")
#     #         print("---")
#     #         print(value)
#     #     print("\n---\n")
#     async for output in app.astream_log(inputs, include_types=["llm"]):
#         # astream_log() yields the requested logs (here LLMs) in JSONPatch format
#         for op in output.ops:
#             if op["path"] == "/streamed_output/-":
#                 # this is the output from .stream()
#                 ...
#             elif op["path"].startswith("/logs/") and op["path"].endswith(
#                 "/streamed_output/-"
#             ):
#                 # because we chose to only include LLMs, these are LLM tokens
#                 print(op["value"])

async def main():
    inputs = {"messages": [HumanMessage(content="LangGraphについて説明")]}
    async for output in app.astream_log(inputs, include_types=["llm"]):
        # print(output)
        # astream_log() yields the requested logs (here LLMs) in JSONPatch format
        for op in output.ops:
            if op["path"] == "/streamed_output/-":
                # this is the output from .stream()
                print(op["value"])
            elif op["path"].startswith("/logs/") and op["path"].endswith(
                "/streamed_output/-"
            ):
                # because we chose to only include LLMs, these are LLM tokens
                print(op["value"])

asyncio.run(main())

# 上記でエージェントからの応答がcontentのトークンごとに出力されるはずだが。。
'''
$ python async.py 
/workspaces/langGraph-practice/venv/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:119: LangChainDeprecationWarning: The function `format_tool_to_openai_function` was deprecated in LangChain 0.1.16 and will be removed in 0.2.0. Use langchain_core.utils.function_calling.convert_to_openai_function() instead.
  warn_deprecated(
{'agent': {'messages': [AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"LangGraph"}', 'name': 'tavily_search_results_json'}}, response_metadata={'finish_reason': 'function_call'}, id='run-3b227be9-3bc2-4ce8-b413-32611883d056-0')]}}
{'action': {'messages': [FunctionMessage(content="[{'url': 'https://python.langchain.com/docs/langgraph/', 'content': 'LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain . It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. It is inspired by Pregel and Apache Beam .'}]", name='tavily_search_results_json')]}}
{'agent': {'messages': [AIMessage(content='LangGraphは、LangChainというLLM（Language Learning Model）に基づいて構築された、状態を持つマルチアクターアプリケーション用のライブラリです。LangGraphは、LangChainの式言語を拡張し、複数のステップで複数のチェーン（またはアクター）を循環的に調整する能力を提供します。これは、PregelやApache Beamからインスパイアされたものです。詳細については、[こちらのリンク](https://python.langchain.com/docs/langgraph/)をご参照ください。', response_metadata={'finish_reason': 'stop'}, id='run-19477d69-1b7f-4ee9-a5dd-9a6690f1aeb2-0')]}}
'''

'''
想定の出力
content='.'
content='0'
content=' °'
content='F'
content=' with'
content=' a'
content=' light'
content=' breeze'
content=' of'
content=' '
content='6'
content='.'
content='9'
content=' mph'
content=' coming'
content=' from'
content=' the'
content=' east'
content='.'
content=' The'
content=' sky'
content=' is'
content=' mostly'
'''