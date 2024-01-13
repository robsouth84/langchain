## cleaned up local rag chat samples from https://python.langchain.com/docs/get_started/quickstart

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage



llm = Ollama(model="llama2")

#After that, we can import and use WebBaseLoader.
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

# Next, we need to index it into a vectorstore. This requires a few components, namely an embedding model and a vectorstore.
# get embeddings and build vector store
embeddings = OllamaEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# However, we want the documents to first come from the retriever we just set up. That way, for a given question we can use the retriever to dynamically select the most relevant documents and pass those in.

retriever = vector.as_retriever()

# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)


# We can test this out by passing in an instance where the user is asking a follow up question.



## collect documents about testing in LangSmith. This is because the LLM generated a new query, combining the chat history with the follow up question.
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})



## Now that we have this new retriever, we can create a new chain to continue the conversation with these retrieved documents in mind.

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
chat = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
## chat dict_keys(['chat_history', 'input', 'context', 'answer'])
print ("Input:     ", chat['input'])
print ("Answer:\n", chat['answer'])









