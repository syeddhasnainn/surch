import requests
import ollama
from parsel import Selector
from html.parser import HTMLParser
import scraper_helper
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import asyncio
import aiohttp
import os
from together import Together
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
INITIAL_SYSTEM_PROMPT = """
If you can read this. First line should be "YES I CAN READ THIS"
As an AI assistant, your task is to provide accurate and concise answers based on the given context and your existing knowledge. Carefully read the provided information, then directly address the user's question, prioritizing the context as your primary reference. If the answer isn't clear from the context, draw on your built-in knowledge. Should you be unsure or lack sufficient information, state I don't have enough information to answer this question accurately. Focus on delivering relevant, concise responses without fabricating information or extrapolating beyond whats provided or known with high confidence. Your ultimate goal is to offer helpful, accurate answers while avoiding speculation or hallucination. Also add the source number where you are referencing from after the paragraph or sentence.
"""

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(texts):
    return model.encode(texts)

def cosine_sim(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def find_most_relevant_chunks(query, chunks, top_k=3):
    query_embedding = create_embeddings([query])[0]
    chunk_embeddings = create_embeddings(chunks)
    
    similarities = [cosine_sim(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
    sorted_indices = np.argsort(similarities)[::-1]
    
    return [chunks[i] for i in sorted_indices[:top_k]]

def together_ai(prompt):
    client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

    stream = client.chat.completions.create(model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                                            messages=[
                                                {
                                                    "role": "system",
                                                    "content": INITIAL_SYSTEM_PROMPT,
                                                },
                                                {
                                                    "role": "user",
                                                    "content": prompt,
                                                }
                                            ],
                                            stream=True,
                                            )

    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

def get_headers():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    return headers

def ollama_chat(prompt):
    stream = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def google_search(query):
    req = requests.get(
        f"https://www.google.com/search?q={query}", headers=get_headers())
    print(req.status_code)
    resp = Selector(text=req.text)
    links = resp.xpath('//span[@jscontroller]/a[@jsname]/@href').getall()
    return links

async def fetch(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f'Error fetching {url}: {e}')
        pass

def clean_html(html_content, remove_tags = [
    'script', 
    'style', 
    'nav', 
    'noscript', 
    'header', 
    'footer', 
    'iframe', 
    'frameset', 
    'frame', 
    'noframes', 
    'applet', 
    'embed', 
    'object', 
    'param', 
    'base', 
    'bgsound', 
    'link', 
    'meta', 
    'xml'
]):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for tag in remove_tags:
        for element in soup.find_all(tag):
            element.decompose()
    
    body = soup.body
    
    if body:
        lines = (line.strip() for line in body.get_text().splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    else:
        return "No body tag found in the HTML."
    
async def main(urls):
    all_responses = []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    source_number = 1
    for response in results:
        try:
            parsed_response = scraper_helper.cleanup(response)
            body_content = clean_html(parsed_response)
            all_responses.append(f'Source: {source_number}\n\n{body_content}')
        except Exception as e:
            print(f'Error parsing response: {e}')
            pass
        source_number += 1

    # Split the content into smaller chunks
    chunks = []
    for response in all_responses:
        chunks.extend(response.split('\n\n'))
    
    return chunks

if __name__ == '__main__':
   
    search_query = "what are promises in js?"
    links = google_search(search_query)
    chunks = asyncio.run(main(links))
    
    # Perform vector search
    relevant_chunks = find_most_relevant_chunks(search_query, chunks)
    
    # Combine the relevant chunks into a single context
    context = '\n\n'.join(relevant_chunks)
    
    llm_query = f"{search_query}\n\n{context}"
    together_ai(llm_query)