from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores.deeplake import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import os
import json

#set keys environment variables
with open('keys.json') as keys_file:
    keys = json.load(keys_file)
os.environ['OPENAI_API_KEY'] = keys['OPENAI_API_KEY']
os.environ['ACTIVELOOP_TOKEN'] = keys['ACTIVELOOP_TOKEN']

#Instantiate LLM and embeddings models
llm=OpenAI(model='text-davinci-003', temperature=0)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

my_activeloop_org_id = 'dylanhashem13'
my_activeloop_dataset_name = 'langchain_course_from_zero_to_hero'
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)


tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

response = agent.run("When was Napoleone born?")
print(response)