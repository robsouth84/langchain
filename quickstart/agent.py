## cleaned up OpenAI agent samples from https://python.langchain.com/docs/get_started/quickstart
## When using an agent we allow the LLM to decides what steps to take.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
## from langchain.chains.combine_documents import create_stuff_documents_chain
## from langchain.chains import create_retrieval_chain
## from langchain.chains import create_history_aware_retriever
# from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
## Agent Tools
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
## 
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

print("sample expects:")
print(" - Tavily account (free) and API key")
print(" - langchainhub (via pip)")
print(" - OpenAI account (needs $$) and API key")
print("\n\n")


def buildLangsmithTool():
  #use WebBaseLoader to generate embeddings and vector store from langsmith URL.
  ## https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.openai.OpenAIEmbeddings.html
  try:
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
  
    embeddings = OpenAIEmbeddings()
  
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
  
    # we want the documents to first come from the retriever we just set up. That way, for a given question we can use the retriever to dynamically select the most relevant documents and pass those in.
    retriever = vector.as_retriever()
    
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )
  except Exception as e:
    print(e)
    exit()
  return retriever_tool

## Define Search Agent Tool
search = TavilySearchResults()

##tools = [retriever_tool, search]
tools = [search ,buildLangsmithTool()]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("\n\nTesting Tavily Search Agent Tool....... ")
agent_executor.invoke({"input": "what is the weather in 48455?"})

print("\n\nTesting LangSmith Retrieval Agent Tool.......")
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
print( agent_executor.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
}))

