import ollama

def mw(prompt):
    stream = ollama.chat(
        model="phi3.5",
        messages=[
            {"role": "system", "content": "you are a helpful assistant. always say hola amigo before the answer."},
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

mw("btc price in usd")