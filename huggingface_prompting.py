from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain
import os
import json

#set keys environment variables
with open('keys.json') as keys_file:
    keys = json.load(keys_file)
os.environ['HUGGINGFACEHUB_API_TOKEN'] = keys['HUGGINGFACEHUB_API_TOKEN']

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)



# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=hub_llm
)

qs_str = (
    "What is the capital city of France?\n" +
    "What is the largest mammal on Earth?\n"
)
print(llm_chain.run(qs_str))