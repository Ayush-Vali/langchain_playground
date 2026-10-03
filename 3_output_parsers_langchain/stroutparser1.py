from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(max_new_tokens=300)
)
model = ChatHuggingFace(llm=llm)

# 1st prompt -> detaled report
template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template = 'Write a 5 line summary on the following text. /n {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'black hole'})

print(result)

