from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema  #- langchain_core doesnt have it so below
# from langchain.output_parsers import StructuredOutputParser , ResponseSchema


# cmn strt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id='meta-llama/Llama-3.2-3B-Instruct',  # Or 'Qwen/Qwen3-4B-Instruct-2507' 'google/gemma-2-2b-it' if you prefer (0.3B, good for creative tasks)
#     task='text-generation'
# )
# model = ChatHuggingFace(llm=llm)

# cmn end
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

print(template.invoke({'topic':'black hole'}))

# chain = template | model | parser

# result = chain.invoke({'topic':'black hole'})

# print(result)
