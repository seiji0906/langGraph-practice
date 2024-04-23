from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from dotenv import load_dotenv
import os


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
print(model)





