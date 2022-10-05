import openai
from typing import List
# from utils.constants import OPENAI_API_KEY
from tqdm import tqdm
import time

OPENAI_API_KEYS = [
    # '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'
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
]
api_key_ptr = 0
openai.api_key = OPENAI_API_KEYS[api_key_ptr]

def request(
        prompt: str,
        engine='text-curie-001',
        max_tokens=64,
        temperature=1.0,
        top_p=1.0,
        n=2,
        stop=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
):
    # retry request (handles connection errors, timeouts, and overloaded API)
    global api_key_ptr
    while True:
        try:
            # print(prompt)
            # print(max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty)
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            # print(response)
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            # time.sleep(1)
            api_key_ptr = (api_key_ptr+1) % len(OPENAI_API_KEYS)
            openai.api_key = OPENAI_API_KEYS[api_key_ptr]

    # print(response)
    generations = [gen['text'].strip() for gen in response['choices']]
    generations = [_ for _ in generations if _ != '']
    # print(generations)
    return generations