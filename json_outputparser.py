import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)

# [
    # {'fact': 'Black holes are regions in space where gravity is so strong that nothing, not even light, can escape.', 'category': 'Fundamental Feature'}, {'fact': 'Black holes form when massive stars collapse at the end of their lives.', 'category': 'Formation'}, {'fact': 'The event horizon is the point of no return around a black 
# hole.', 'category': 'Defining Feature'}, {'fact': 'There are different types of black holes, including stellar-mass black holes, supermassive black holes, and intermediate-mass black holes.', 'category': 'Classification'}, {'fact': 'Black holes are actively researched, with scientists studying their impact on the evolution of the universe and the nature of gravity itself.', 'category': 'Research Importance'}
# ]