from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import os
import json

#set keys environment variables
with open('keys.json') as keys_file:
    keys = json.load(keys_file)
os.environ['OPENAI_API_KEY'] = keys['OPENAI_API_KEY']

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
template = "You are an assistant that helps users find information about movies."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find information about the movie {movie_title}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

response = chat(chat_prompt.format_prompt(movie_title="The Witch").to_messages())

print(response.content)