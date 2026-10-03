from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",  # ← typo fix: remove duplicate comment
    task="text-generation",
    temperature=0.7,          # optional but helps
    max_new_tokens=1200,
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# ── First stage: detailed report ──
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable scientific writer."),
    ("human", "Write a detailed, well-structured report on {topic}. Include key facts, history, current understanding, and open questions."),
])

# ── Second stage: summarize previous output ──
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at concise summarization."),
    ("human", "Write a **maximum 5-line summary** of the following text. Be clear and objective.\n\n{text}"),
])

# Chain: report → summary
chain = (
    prompt1
    | model
    | parser
    | {"text": lambda x: x}           # pass output forward as {text}
    | prompt2
    | model
    | parser
)

result = chain.invoke({"topic": "black hole"})
print(result)

