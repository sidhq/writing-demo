import os
import requests
import json
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SID_ACCESS_TOKEN = os.getenv("SID_ACCESS_TOKEN")

query_template = """
Given a writing prompt, return a query that would be useful to find relevant information.

Writing prompt:
{query}
Query:
"""

query_prompt = PromptTemplate(template=query_template, input_variables=['query'])

writing_template = """
Write a text about the following query:
{query}

Use the following context:
{context}
"""
writing_prompt = PromptTemplate(template=writing_template, input_variables=['query', 'context'])

llm = OpenAI(openai_api_key=OPENAI_API_KEY)

query_chain = LLMChain(llm=llm, prompt=query_prompt)
writing_chain = LLMChain(llm=llm, prompt=writing_prompt)

def call_sid(query: str, count: int = 5) -> list[str]:
    url = 'https://api.sid.ai/v1/users/me/query'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {SID_ACCESS_TOKEN}'
    }
    data = {
        'query': query,
        'limit': count
    }
    response_json = requests.post(url, headers=headers, data=json.dumps(data)).json()
    return [result['text'] for result in response_json['results']]

def main():
    while True:
        query = input('What are you writing?\n')
        sid_query = query_chain.run(query)
        results = call_sid(sid_query)

        result_string = '\n'.join([f'{i+1}. {result}' for i, result in enumerate(results)])
        output = writing_chain.run(query=query, context=result_string)
        print(output)

if __name__ == '__main__':
    main()