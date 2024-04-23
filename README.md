# langGraph-practice

## 目次
* LangGraphについて
* LangGraphとAutoGen
* サイクル
* 環境構築

[LangGraph](https://python.langchain.com/docs/langgraph/)
> （公式）LangGraph は、LLM を使用してステートフルなマルチアクター アプリケーションを構築するためのライブラリであり、 LangChain上に構築されています (LangChain とともに使用されることを目的としています) 。

> 主な用途は、LLM アプリケーションにサイクルを追加することです。サイクルは、ループ内で LLM を呼び出し、次にどのようなアクションを実行するかを尋ねる、エージェントのような動作にとって重要です。

用語説明：

* ステートフルなマルチアクターアプリケーション
> 「ステートフルなマルチアクターアプリケーション」とは、複数の独立したコンポーネント（アクター）が相互に通信しながら動作し、その過程で各アクターが状態（state）を保持するようなアプリケーションを指します。ここでいう「ステートフル」（stateful）とは、アプリケーションが過去のデータやイベントの履歴を記憶し続けることを意味します。これにより、アプリケーションは以前の状態に基づいて行動を決定したり、状態の変化に適応したりすることができます。

* サイクル
> [LangGraphの概要](https://note.com/npaka/n/n01954b4c649e)\
(1) LLMを呼び出して、「どのようなアクションを実行するか」または「ユーザーにどのような応答を返すか」を決定\
(2) 与えられたアクションを実行し、ステップ1に戻る

* DAGワークフロー
> DAG（Directed Acyclic Graph、有向非巡回グラフ）ワークフローとは、タスクやプロセスがノードとして表され、これらの間の依存関係がエッジ（矢印）で示されるグラフ構造の一つです。このグラフは「非巡回」であるため、いかなるノードから始めても同じノードに戻るループが存在しないのが特徴です。プロジェクト管理やデータ処理など、複雑な依存関係を持つタスクを効率的に管理するために使用されます。

* LCEL
> LCEL（LangChain Expression Language）は、LangChainを用いたアプリケーション開発で使用される専用の表現言語です。この言語は特にDAGワークフローの構築に最適化されており、タスク間の依存関係や条件分岐などを簡潔に記述することができます。LCELを使うことで、より複雑なデータ処理やビジネスロジックの実装が可能になります。

## ステートマシン
>[LangGraphの概要](https://note.com/npaka/n/n01954b4c649e)\
エージェントに最初に常に特定のツールを呼び出すように強制したい場合があります。ツールがどのように呼び出されるかをもっと制御したい場合があります。エージェントの状態に応じて異なるプロンプトを使用したい場合があります。\
これらのより制御されたフローのことを「LangChain」では「ステートマシン」と呼んでいます。


**「LangGraph」は、グラフとして指定することで、これらの「ステートマシン」を作成する方法になります。**

## LangGraphの要素
* StateGraph
* Node
* Edge
* Compile


> [LangGraphの概要と使い方](https://zenn.dev/umi_mori/books/prompt-engineer/viewer/langgraph)

> StateGraph エージェントの状態を定義\
LangGraphでは、「StatefulGraph」というグラフが主に使用されます。StatefulGraphでは、各ノードが状態を更新する方法を定義する状態オブジェクトを使用します。この状態は、特定の属性を上書き（SET）するか追加（ADD）するかで操作が分かれます。

> ノードの定義
必要な主要なノードは2つあります：
1. エージェント：どのようなアクションを（もし取るなら）取るかを決定します。
2. ツールを呼び出すための関数：エージェントがアクションを決定した際に、そのアクションを実行します。

> エッジの定義 エッジはノード間の流れを制御します。2種類のエッジがあります：

>条件付きエッジ：エージェントが呼び出された後、次のいずれかを実行する必要があります：\
a. エージェントがアクションを取るよう指示した場合、ツールを呼び出します。\
b. エージェントが処理完了を指示した場合、終了します。
通常のエッジ：ツールの呼び出し後、エージェントに戻り、次のステップを決定します。
この構造を用いて、エージェントの動作や処理の流れを柔軟に管理できます。




コードの説明

ユーザーからのプロンプトを　runnable.invoke(HumanMessage(user_input))を通じて、言語モデルに渡され、言語モデルはプロンプトからtool_callsを必要に応じて生成する。
tool_callsの有無で、multiply関数を用いるかどうかを判断している。



# サイクル

LangChainからクラスを再生成する。エージェント自体は、チャットモデルと関数呼び出しを使用する。
## AgentExecutor
このエージェントは、すべての状態をメッセージのリストとして表す。

例として使用するツール
### [tavily](https://app.tavily.com/)
AIエージェントとLLMに最適化された最初のリサーチエンジン

### ツールのセットアップ
使用するツールの定義

[独自のツールの作成方法](https://python.langchain.com/docs/modules/tools/custom_tools/)


```python
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]
```
> [TavilySearchResults doc](https://api.python.langchain.com/en/latest/tools/langchain_community.tools.tavily_search.tool.TavilySearchResults.html)\
Tavily Search APIをクエリしてjsonを返すツール。
キーワード引数からの入力データを解析して検証することにより、新しいモデルを作成します。
入力データを解析して有効なモデルを形成できない場合、 ValidationError を送出します。

### 使用するモデルの設定
使用するチャットモデルのロード

```python
# openai
from langchain_openai import ChatOpenAI
model = ChatOpenAI(temperature=0, streaming=True)

# 他のLLM(azureやanthropic,huggingfaceなど)
# langchain_communityにあるにはあるっぽい。バージョンとか互換性があるかはわからぬ
# https://api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.llms
```

### 定義したツールをモデルが認識できるか確認
```python
from langchain.tools.render import format_tool_to_openai_function
# format_tool_to_openai_functionは、langchain0.2で削除される（現在0.1.16）
# LangChainDeprecationWarning: The function `format_tool_to_openai_function` was deprecated in LangChain 0.1.16 and will be removed in 0.2.0. Use langchain_core.utils.function_calling.convert_to_openai_function() instead.

functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)
```
上記コードで、LangChainツールをOpenAI関数呼び出しの形式に変換し、モデルクラスにバインドしている

### エージェントの状態の定義（StateGraph）
StateGraphは、各ノードに状態オブジェクトを渡し、状態オブジェクトをアップデートする操作を返すノードを作成する。これらの操作は、状態の特定の属性をSET（既存の値を上書きするなど）するか、既に存在する属性にADDすることができる。
SETするかADDするかは、状態オブジェクトの定義の際に、指定する。
```python
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
```
上記例では、メッセージのリストの状態を追跡するAgentStateクラスを定義している。各ノード（エージェントの状態をSETまたはADDする）は、状態の更新をする際には、メッセージを追加するだけである。故に単一のキーmessagesを持つTypedDictを使用している。第２引数に状態の更新の仕方（SETかADDか）を指定している。


### ノードの定義
ノードは関数または実行可能オブジェクトのいずれかになる。
今回の例では二つのノードを必要とする。

* エージェント：何かしらのアクションを取るかどうかを決定する責任者。
* ツールを起動する関数：エージェントがアクションを取ることを決定したら、このノードがそのアクションを実行します。

> ノード：　グラフ内の個別の点や位置を表す\
エッジ：　ノード間の関係や接続を表す\
数学のグラフ理論の用語

エッジの定義
* 条件付きエッジ：　エージェントが呼び出された後\
a. エージェントがアクションを実行するように指示した場合は、ツールを呼び出す関数を呼び出す必要がある\
b. エージェントが終了したと言った場合は、終了する必要がある\
* 通常のエッジ：　ツールが呼び出された後、常にエージェントに戻り、次に何をすべきかを決定する


```python
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
def call_tool(state):
    messages = state['messages']
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}
```

### 数学グラフ理論



# 環境構築
.envにAPIキー等必要な情報を記述\
python -m venv venv\
source venv/bin/activate\
pip install -r requirements.txt\
streamlit run app.py
