# from langchain.llms import OpenAI
# llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.7)

# from langchain.chains import LLMChain  # still old
from langchain_core.prompts import PromptTemplate # to check

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


prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# create a llm chain
chain = LLMChain(llm=llm, prompt=prompt)


topic = input('Enter a topic')

formatted_prompt = prompt.format(topic=topic)

# Call the llm directly 
blog_title = llm.predict(formatted_prompt)

print("generated blog title", blog_title)
