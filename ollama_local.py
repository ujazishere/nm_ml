import ollama


def lprompt(prompt):
    stream = ollama.chat(
        model='llama2',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

prompt = "give me python code for pandas library. Include most widely used and most valuable"

lprompt(prompt)


