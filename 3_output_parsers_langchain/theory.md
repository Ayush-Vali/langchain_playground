The role of parser.parse is same everywhere right, the only diff betwn stroutputparser,jsonoutputparser & structeredoutputparser is when we create template code and mention partial_variable 



so the main player is get_format_instructions



template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


