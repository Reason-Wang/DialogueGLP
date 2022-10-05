from gpt3_generation import request
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import click
import re
from pathlib import Path
from typing import List


def prompt_format(prompt: str, keywords: List[str]):
    context_string = prompt
    if keywords is not None:
        n = np.random.choice(range(1, len(keywords)+1))      # number of keywords
        keywords = random.sample(keywords, n)                # subset of keywords
        context_string = context_string.replace('{keywords}', ', '.join(keywords))

    context_string = re.sub(r'\s+([?,.!"])', r'\1', context_string)
    context_string = re.sub(r' (\n)', r'\1', context_string)
    context_string = re.sub(r' +', r' ', context_string)
    # context_string = re.sub(r' _ ', r"'", context_string)
    return context_string


def get_input_from_DD_dialogue(df):
    prompt = "The following is a conversation.\n\n"
    length = len(df)
    if length <= 1:
        print("Skipped dialogue with 1 utterance")
        return None

    for i, row in df.iterrows():

        if (length - i) % 2 == 0:
            speaker = 'Other: '
        else:
            speaker = 'I: '
        prompt = prompt + speaker + row['Utterance'] + '\n'
        if i == length - 2:
            break
    prompt = prompt + '\nWhat do I feel now and why?\n'
    return prompt


def get_input_from_meld_dialogue(df):
    prompt = "The following is a conversation between me and other people.\n\n"
    length = len(df)
    if length <= 1:
        print("Skipped dialogue with 1 utterance")
        return None

    addressee = df.loc[length-1, 'Speaker']
    for i, row in df.iterrows():
        if row['Speaker'] != addressee:
            speaker = row['Speaker'] + ": "
        else:
            speaker = 'I: '
        prompt = prompt + speaker + row['Utterance'] + '\n'
        if i == length - 2:
            break
    prompt = prompt + '\nWhat do I feel now and why?\n'
    return prompt


def get_input_from_emory_dialogue(df):
    prompt = "The following is a conversation between me and other people.\n\n"
    length = len(df)
    if length <= 1:
        print("Skipped dialogue with 1 utterance")
        return None

    addressee = df.loc[length - 1, 'Speaker']
    for i, row in df.iterrows():
        if row['Speaker'] != addressee:
            speaker = row['Speaker'] + ": "
        else:
            speaker = 'I: '
        prompt = prompt + speaker + row['Utterance'] + '\n'
        if i == length - 2:
            break
    prompt = prompt + '\nWhat do I feel now and why?\n'
    return prompt


@click.command()
@click.option('--task', type=str, default=None)
@click.option('--input_path', type=str, default=None)
@click.option('--output_path', type=str, default=None)
@click.option('--num_knowledge', type=int, default=1)
@click.option('--top_p', default=0.5, type=float)
@click.option('--temperature', default=1.0, type=float)
@click.option('--max_tokens', default=64, type=int)
@click.option('--engine', default='text-curie-001', type=str)
@click.option('--n', default=None, type=int)
def main(
    task: str,
    input_path: str,
    output_path: str,
    num_knowledge: int,
    top_p: float,
    temperature: float,
    max_tokens: int,
    engine: str,
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
        if task == 'daily_dialogue':
            prompt = get_input_from_DD_dialogue(item[1])
        elif task == 'meld':
            prompt = get_input_from_meld_dialogue(item[1])
        elif task == 'emorynlp':
            prompt = get_input_from_emory_dialogue(item[1])
        else:
            raise RuntimeError("Task must be either daily_dialogue, meld or emorynlp.")
        if prompt is None:
            knowledges = ['' for i in range(num_knowledge)]
        else:
            context_string = prompt_format(
                prompt,
                keywords=None
            )
            # print(context_string)
            knowledges = request(
                context_string,
                n=num_knowledge,
                engine=engine,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens)
        knowledges = list(set(knowledges))
        # knowledges = [re.sub(r'^(AI:)', '', knowledge) for knowledge in knowledges]
        # knowledges = [knowledge.lstrip() for knowledge in knowledges]
        item[1].loc[len(item[1])-1, "Knowledge"] = str(knowledges)
        generated_df = pd.concat([generated_df, item[1]])

    generated_df.to_csv(output_path, index=False, columns=['Dialogue_ID', 'Utterance_ID', 'Knowledge'])

if __name__ == '__main__':
    main()