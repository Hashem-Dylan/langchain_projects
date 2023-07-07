from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores.deeplake import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
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

#create your own docs
texts = [
    'Napolean Bonaporte was born in 15 August 1769',
    'Louis XIV was born in 5 September 1638'
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

#create Deep Lake Data Set
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = 'dylanhashem13'
my_activeloop_dataset_name = 'langchain_course_from_zero_to_hero'
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

#add docs to Deep Lake Dataset
db.add_documents(docs)