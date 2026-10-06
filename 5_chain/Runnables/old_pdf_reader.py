from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# cmn
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",  # Qwen/Qwen3-4B-Instruct-2507"
    task="text-generation",
    temperature=0.7,          # optional but helps
) 

model = ChatHuggingFace(llm = llm)

# cmn end

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# load doc
loader = TextLoader('docs.txt')
documents = loader.load()

# split the text 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)



# convert text into embeddings & store in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# create a retriever (does semantic search) (i.e fetches relevant docs)
retriever = vectorstore.as_retriever()




# Manually Retrieve Relevant Documents
query = 'query: What are the key takeways from the document?'
retrieved_docs = retriever.get_relevant_documents(query)

# combine Retrieved Text into a Single Prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Manually Pass Retieved text to LLM
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = model.invoke(prompt)
ans = llm.predict(prompt)

print(f"Answer model: {answer}")
print(f"\n\n\n\n\nAnswer llm: {ans}")


"""LLM is basic form of llm, when passed thru ChatHuggingFace it can handle ChatPromptemplate,etc 
similar for OpenAI vs from langchain_openai import ChatOpenAI
"""