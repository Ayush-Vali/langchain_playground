from langchain_core.output_parsers import StrOutputParser

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
    max_new_tokens=1200,
)
model = ChatHuggingFace(llm=llm)

# cmn end


prompt1 = ChatPromptTemplate.from_messages([
    ("system", "Generate a detailed report on {topic}"),
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "Generate a 5 pointer summary from the following text \n {text}"),
])


parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

res = chain.invoke({'topic' : 'Unenmployment in India'})

print(res)

chain.get_graph().print_ascii()