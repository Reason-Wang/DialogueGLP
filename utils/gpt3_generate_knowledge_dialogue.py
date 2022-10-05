from gpt3_generation import request
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import click
import re
from pathlib import Path
from typing import List


def prompt_format(prompt_path: str, keywords: List[str], input: dict):
    with open(prompt_path) as f:
        context_string = f.read().strip('\n')
    if keywords is not None:
        n = np.random.choice(range(1, len(keywords)+1))      # number of keywords
        keywords = random.sample(keywords, n)                # subset of keywords
        context_string = context_string.replace('{keywords}', ', '.join(keywords))
    # if input is not None:
    #     context_string = context_string.replace('{utterance}', input)
    if input is not None:                                    # daily dialogue
        context_string = context_string.replace('{emotion_tag}',
                                '' if input['emotion'] == 'no emotion' else ' and '+input['emotion'])
        context_string = context_string.replace('{emotion}',
                                'peaceful' if input['emotion'] == 'no emotion' else input['emotion'])
        context_string = context_string.replace('{utterance1}', input['utterance1'])
        if input['utterance2'] is not None:
            context_string = context_string.replace('{utterance2}', input['utterance2'])

    context_string = re.sub(r'\s+([?,.!"])', r'\1', context_string)
    context_string = re.sub(r' (\n$)', r'\1', context_string)
    return context_string


def get_input_from_DD_dialogue(df):
    emotion_map = {'no emotion': 'no emotion', 'surprise': 'surprised', 'fear': 'feared',
                   'happiness': 'happy', 'sadness': 'sad', 'anger': 'angry', 'disgust': 'disgusted'}
    length = len(df)
    if length <= 1:
        print("Skipped dialogue with 1 utterances")
        return None
    if length == 2:
        emotion = emotion_map[df.loc[0, "Emotion"]]
        u1 = df.loc[0, "Utterance"]
        return {"emotion": emotion, "utterance1": u1, "utterance2": None}

    emotion = emotion_map[df.loc[length-3, "Emotion"]]
    u1 = df.loc[length-3, "Utterance"]
    u2 = df.loc[length-2, "Utterance"]
    input = {"emotion": emotion, "utterance1": u1, "utterance2": u2}
    return input


@click.command()
@click.option('--task', type=str, default=None)
@click.option('--input_path', type=str, default=None)
@click.option('--output_path', type=str, default=None)
@click.option('--prompt_path_one', type=str, default=None)
@click.option('--prompt_path_two', type=str, default=None)
@click.option('--num_knowledge', type=int, default=1)
@click.option('--top_p', default=0.5, type=float)
@click.option('--temperature', default=1.0, type=float)
@click.option('--max_tokens', default=64, type=int)
@click.option('--n', default=None, type=int)
def main(
    task: str,
    input_path: str,
    output_path: str,
    prompt_path_one: str,
    prompt_path_two: str,
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
    # generate knowledge!
    # generated_examples = []
    generated_df = pd.DataFrame()
    for item in tqdm(list(eval_df.groupby("Dialogue_ID"))):
        dialog_id = item[0]
        item[1].sort_values(by=["Utterance_ID"], inplace=True)
        item[1].reset_index(drop=True, inplace=True)
        input = get_input_from_DD_dialogue(item[1])
        if input is None:
            knowledges = ['' for i in range(num_knowledge)]

        else:
            prompt_path = prompt_path_two
            if input['utterance2'] is None:
                prompt_path = prompt_path_one

            context_string = prompt_format(
                prompt_path,
                keywords=None,
                input=input)
            # print(context_string)
            knowledges = request(
                context_string,
                n=num_knowledge,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens)
            # print(list(set(knowledges)))
            # print(knowledges)
        knowledges = [re.sub(r'^(AI:)', '', knowledge) for knowledge in knowledges]
        knowledges = [knowledge.lstrip() for knowledge in knowledges]
        item[1].loc[len(item[1])-1, "Knowledge"] = str(knowledges)
        # generated_examples.append(row.to_dict())
        generated_df = pd.concat([generated_df, item[1]])
    # with open(output_path, 'w') as fo:
        # json.dump(generated_examples, fo, indent=4)

    generated_df.to_csv(output_path)

if __name__ == '__main__':
    main()
