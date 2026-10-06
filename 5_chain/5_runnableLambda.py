from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

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



prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda tex: len(tex.split()))
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

print(result)
# for better display 
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)  