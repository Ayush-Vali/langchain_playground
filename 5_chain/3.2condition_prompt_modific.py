from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate # to check
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from langchain_core.runnables import RunnableBranch


from langchain_core.exceptions import OutputParserException

# cmn
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",  # Qwen/Qwen3-4B-Instruct-2507"
    task="text-generation",
    temperature=0.3,          # optional but helps
) 

model = ChatHuggingFace(llm = llm)

# cmn end

# parser = StrOutputParser()


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
# print(classifier_chain.invoke({'feedback':'This is a wonderful smartphone'}))



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





# ---

# from langchain_core.runnables import RunnableBranch, RunnableLambda

# parser = StrOutputParser()

# prompt3 = PromptTemplate(
#     template='Write an appropriate response to this feedback \n {feedback}',
#     input_variables=['feedback']
# )


# run_chain = RunnableBranch(
#     (lambda x:'pos' in x.sentiment, prompt3 | model | parser),
#     (lambda x:'neg' in x.sentiment, prompt3 | model | parser),
#     RunnableLambda(lambda x: "could not find sentiment")
# )

# chain = classifier_chain | run_chain

# res = chain.invoke({"feedback": feedback_text})
# print(res)
