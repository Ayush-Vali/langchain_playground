
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

hist =[]
# cmn end


prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 5 interesting facts about {topic}"),
    MessagesPlaceholder(variable_name="chat_history"),   #
])

parser = StrOutputParser()

chain = prompt | model | parser

res = chain.invoke({'topic':'cricket', 'chat_history' :hist })

print(res)

chain.get_graph().print_ascii()