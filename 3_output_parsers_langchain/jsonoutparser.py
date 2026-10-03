from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# cmn strt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',  # Or 'Qwen/Qwen3-4B-Instruct-2507' 'google/gemma-2-2b-it' if you prefer (0.3B, good for creative tasks)
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

# cmn end

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)



chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)


## RESULT: IT returns json, but key not as intended like fact1,etc

""" IF used wuthout chain """

# temp.invoke({})
#model.invoke()
# parser.parse(result.content)
