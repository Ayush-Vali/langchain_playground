from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

from langchain.output_parsers import StructuredOutputParser,ResponseSchema

# Task1
- Task is to take first prompt's output and put it into the second prompt

## 1 in stroutparser1.py, test.py  --- uses structed_output_parsers  
4r- replacement of "result.content".... cause with stroutparser.py we can do foll

chain = template1 | model | parser | template2 | model | parser

Correct data flow (with parser)
PromptTemplate
↓
AIMessage
↓ (StrOutputParser)
string
↓
PromptTemplate
↓
AIMessage
↓ (StrOutputParser)
string

- As u can tell StrOutputParser, can remove the metadata + can be used in chain

## 2 in jsonoutparser.py   ---- uses JsonOutputParser

JsonOutParser flaw is that sometimes- 
- IT Cant Enforce Schema i.e it cant specify what is key and what is value
(Like if i want key as 'Fact1:__ , Fact2:__...'  it'll give { 'balckhole :___ , watch :_...'})

- Since its 

## 3 StructuredOutputParser(X)
(NOTE: ITS NOT present in "langchain_core" which houses most reusable components, since not used much) 

-- Help enforce schema
-- But doesnt provide validation

## 4 Pydantic output parser
- does everything, from validation to conversion
- converted to pydantic obj, but can convert back in any format



Extra SYntax query-

1) prompt = template.invoke({})
-- THat additional bracket is cause, it gives error even if no input_variables is there, EG:

template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': "Return a JSON object."}
)

2) What is partial_variables in PromptTemplate(eg. in jsonOutputparser.py)
partial_variables lets you pre-fill some variables in the prompt

These values are fixed and don’t need to be passed every time
(u can see in above eg, template.incoke({}) is empty)

** simple distinc **
🔑 Anything constant → partial_variables
🔑 Anything dynamic → passed during .invoke() or in input_var param

--
3) 