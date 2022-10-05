from .gpt3_generation import request
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import click
import re
from typing import List


def prompt_format(prompt: str, keywords: List[str]):
    context_string = prompt
    if keywords is not None:
        n = np.random.choice(range(1, len(keywords)+1))      # number of keywords
        keywords = random.sample(keywords, n)                # subset of keywords
        context_string = context_string.replace('{keywords}', ', '.join(keywords))

    context_string = re.sub(r'\s+([?,.!"])', r'\1', context_string)
    context_string = re.sub(r' (\n)', r'\1', context_string)
    return context_string


def get_input_from_DD_dialogue(df):
    emotion_map = {'no emotion': 'no emotion', 'surprise': 'surprised', 'fear': 'feared',
                   'happiness': 'happy', 'sadness': 'sad', 'anger': 'angry', 'disgust': 'disgusted'}
    prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative and clever.\n\n"
    length = len(df)
    if length <= 1:
        print("Skipped dialogue with 1 utterance")
        return None

    for i, row in df.iterrows():

        if (length - i) % 2 == 0:
            speaker = 'Human: '
        else:
            speaker = 'AI: '
        prompt = prompt + speaker + row['Utterance'] + '\n'
        if i == length - 2:
            break
    return prompt


@click.command()
@click.option('--task', type=str, default=None)
@click.option('--input_path', type=str, default=None)
@click.option('--output_path', type=str, default=None)
@click.option('--num_knowledge', type=int, default=1)
@click.option('--top_p', default=0.5, type=float)
@click.option('--temperature', default=1.0, type=float)
@click.option('--max_tokens', default=64, type=int)
@click.option('--n', default=None, type=int)
def main(
    task: str,
    input_path: str,
    output_path: str,
    num_knowledge: int,
    top_p: float,
    temperature: float,
    max_tokens: int,
    n: int,
):
    # read examples for inference
    eval_df = pd.read_csv(input_path)
    eval_df = eval_df[:n]
    eval_df['Knowledge'] = ['' for i in range(len(eval_df))]
    generated_df = pd.DataFrame()
    for item in tqdm(list(eval_df.groupby("Dialogue_ID"))):
        dialog_id = item[0]
        item[1].sort_values(by=["Utterance_ID"], inplace=True)
        item[1].reset_index(drop=True, inplace=True)
        prompt = get_input_from_DD_dialogue(item[1])
        if prompt is None:
            knowledges = ['' for i in range(num_knowledge)]
        else:
            context_string = prompt_format(
                prompt,
                keywords=None
            )
            print(context_string)
            knowledges = request(
                context_string,
                n=num_knowledge,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens)
        knowledges = [re.sub(r'^(AI:)', '', knowledge) for knowledge in knowledges]
        knowledges = [knowledge.lstrip() for knowledge in knowledges]
        item[1].loc[len(item[1])-1, "Knowledge"] = str(knowledges)
        generated_df = pd.concat([generated_df, item[1]])

    generated_df.to_csv(output_path)

if __name__ == '__main__':
    main()
