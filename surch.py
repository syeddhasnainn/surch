import requests
import ollama
from parsel import Selector
from html.parser import HTMLParser
import scraper_helper
from bs4 import BeautifulSoup
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
console = Console()


def llm(prompt):
    stream = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful content summarizer. Summarize the the text that is given to you. Do not hallucinate or invent text. Do not return any other text. Only return the summary. Your response should be in the following format:",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )

    for chunk in stream:
          console.print(chunk['message']['content'], end='', flush=True)

def html_scraper(url):
    req = requests.get(url)
    body = BeautifulSoup(req.text, 'html.parser')
    return body.get_text()

if __name__ == '__main__':
    content = html_scraper("https://cpu.land/the-basics")
    summary = llm(content)
    markdown = Markdown(summary)
    console.print(markdown)