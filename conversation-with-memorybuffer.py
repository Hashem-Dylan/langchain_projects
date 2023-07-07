from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
import json

#set keys environment variables
with open('keys.json') as keys_file:
    keys = json.load(keys_file)
os.environ['OPENAI_API_KEY'] = keys['OPENAI_API_KEY']
os.environ['DEEPLAKE_API_KEY'] = keys['DEEPLAKE_API_KEY']

#declare llm
llm = OpenAI(model='text-davinci-003', temperature=0.9)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
#Start conversation
conversation.predict(input='Tell me about yourself.')

#Continue Conversation
conversation.predict(input='What can you do?')
conversation.predict(input='How can you assist me with data analysis?')



