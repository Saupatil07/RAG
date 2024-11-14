from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

loader = UnstructuredFileLoader("ai_adoption_framework_whitepaper.pdf")

docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200)

texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts,embeddings)
llm = Ollama(model='llama3.2:1b')

chain = RetrievalQA.from_chain_type(
    llm,
    retriever = db.as_retriever()
)
question = "Can you please summarize the document"
result = chain.invoke({"query": question})

print(result['result'])