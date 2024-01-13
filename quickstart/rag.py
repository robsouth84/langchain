## cleaned up local rag samples from https://python.langchain.com/docs/get_started/quickstart
## for deeper dive https://python.langchain.com/docs/modules/data_connection

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain



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

#First, let's set up the chain that takes a question and the retrieved documents and generates an answer.

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

## If we wanted to, we could run this ourselves by passing in documents directly:
## print(document_chain.invoke({
##     "input": "how can langsmith help with testing?",
##     "context": [Document(page_content="langsmith can let you visualize test results")]
## }))


# However, we want the documents to first come from the retriever we just set up. That way, for a given question we can use the retriever to dynamically select the most relevant documents and pass those in.




retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


#We can now invoke this chain. This returns a dictionary - the response from the LLM is in the answer key


response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])

# LangSmith offers several features that can help with testing:...
