from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# cmn
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",  # Qwen/Qwen3-4B-Instruct-2507"
    task="text-generation",
    temperature=0.7,          # optional but helps
    max_new_tokens=1200,
) 

llm2 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",  # Qwen/Qwen3-4B-Instruct-2507"
    task="text-generation",
    temperature=0.7,          # optional but helps
    max_new_tokens=1200,
)

model1 = ChatHuggingFace(llm = llm1)
model2 = ChatHuggingFace(llm=llm2)


# cmn end

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "Generate a short and simple notes from the following text \n {text}"),
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "Generate a 5 short question answers from the following text \n {text}"),
])

prompt3 = ChatPromptTemplate.from_messages([
    ("system", "Merge the provided notes and quiz into a single document \n notes ->{notes} and quiz -> {quiz}"),
])

parser = StrOutputParser()


# Parallel chain start uses (RunnableParallel)

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# during merge use any mode
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain


text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

"""


res = chain.invoke({'text' : text})

print(res)

chain.get_graph().print_ascii()