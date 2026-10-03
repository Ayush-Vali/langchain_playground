from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


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


## Using PromptTemplate
class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)



chain = template | model | parser

final_result = chain.invoke({'place':'chinese'})

print(final_result.name) # or final_result.age,etc



## Using ChatPromptTemplate

class Feedback(BaseModel):

    sentiment: Annotated[
        Literal["positive", "negative"],
        Field(description="Give the sentiment of the feedback")
    ]

parser2 = PydanticOutputParser(pydantic_object=Feedback)


format_instructions = parser2.get_format_instructions()
prompt1 = ChatPromptTemplate.from_messages([
("system", """Classify the sentiment of the following feedback into positive or negative.
Respond ONLY with valid JSON
{feedback}

{format_instructions}"""),
])
prompt1 = prompt1.partial(format_instructions=format_instructions)

classifier_chain = prompt1 | model | parser2
feedback_text = 'This is a wonderful smartphone'

try:
    res = classifier_chain.invoke({"feedback": feedback_text})
    print(res.sentiment)
    print("first try")
except OutputParserException as e:
    print("Parser failed → fallback to raw model")
    # Quick rescue chain without strict parser
 
    rescue_prompt = ChatPromptTemplate.from_messages([
        ("system", "Return exactly one word: positive or negative\nFeedback: {feedback}"),
    ])
    rescue_chain = prompt1 | model | StrOutputParser()
    raw = rescue_chain.invoke({"feedback": feedback_text}).strip().lower()
    
    if "pos" in raw:
        sentiment = "positive"
    elif "neg" in raw:
        sentiment = "negative"
    else:
        sentiment = "unknown"
    
    print(sentiment)