import pandas as pd
import ast


#============================================================================================
# Data Format:
# {
#   'dialogue': [[utterance 0_0', ...], [..., ..., ...], [..., ..., ...],
#   'label': [{'utterance': ..., 'emotion': ..., 'emb': ..., 'speaker': ...}, { ... }, ...],
#   'speaker': [['speaker1, ..., ...], [..., ..., ...], [..., ..., ...],
#   'embedding': [[emb1, ..., ..., ], [..., ..., ...], [..., ..., ...]
# }
#============================================================================================

def get_knowledge(df, knowledge):
    # print(knowledge)
    knowledges = []
    if knowledge == 'comet':
        for item in list(df.groupby("Dialogue_ID")):
            k = []
            for i, row in item[1].reset_index(drop=True).iterrows():
                if i != len(item[1]) -1:
                    k.append([[k0.replace("personx", "other").replace("persony", "I") for k0 in k1] for k1 in row['Knowledge']])
            knowledges.append(k)

    if knowledge == 'feeling' or knowledge == 'utterance':
        for item in list(df.groupby("Dialogue_ID")):
            k = []
            for i, row in item[1].reset_index(drop=True).iterrows():
                if isinstance(row['Knowledge'], list):
                    k.append([klg.strip() for klg in row["Knowledge"]])
                else:
                    k.append([])
            knowledges.append(k)

    if knowledge == 'none':
        knowledges = [None for i in range(len(df))]
    return knowledges

def process_daily_dialog(split, emb_dict, knowledge):
    emotion_label_map = {'no emotion': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5,
                         'surprise': 6}
    input_dir = 'data/'
    df = pd.read_csv(input_dir + f'daily_dialogue/{split}.csv')
    if knowledge != 'none':
        if knowledge == 'comet':
            k_df = pd.read_csv(input_dir + f'daily_dialogue/knowledge/{split}_comet_knowledge.csv')
            k_df['Knowledge'] = k_df['Knowledge'].apply(ast.literal_eval)
        elif knowledge == 'feeling':
            k_df = pd.read_csv(input_dir + f'daily_dialogue/knowledge/{split}_knowledge_full_dialogue_feeling.csv')
            k_df['Knowledge'] = k_df['Knowledge'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        elif knowledge == 'utterance':
            k_df = pd.read_csv(input_dir + f'daily_dialogue/knowledge/{split}_knowledge_full_dialogue_chatbot.csv')
            k_df['Knowledge'] = k_df['Knowledge'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        else:
            raise "knowledge must be either none, comet, feeling or utterance"

        df = pd.merge(df, k_df, how='left', on=['Dialogue_ID', 'Utterance_ID'])


    knowledges = get_knowledge(df, knowledge)


    dialogues = []
    labels = []
    speakers = []
    emotions = []
    embeddings = []
    for item in list(df.groupby("Dialogue_ID")):
        dialog_id = item[0]
        dialog = []
        label = None
        emotion = []
        speaker = []
        emb = []
        item[1].sort_values(by=['Utterance_ID'], inplace=True)
        dialog_df = item[1].reset_index(drop=True)
        for i, row in dialog_df.iterrows():
            if i != len(dialog_df) - 1:
                # print(row['Knowledge'])
                dialog.append(row['Utterance'])
                speaker.append(i % 2)  # because DD is dataset with 2 speakers
                emb.append(emb_dict[str(dialog_id) + '_' + str(row['Utterance_ID'])])
                emotion.append(emotion_label_map[row['Emotion']])
            else:
                label = {
                    'utterance': row['Utterance'],
                     'emotion': emotion_label_map[row['Emotion']],
                     'emb': emb_dict[str(dialog_id) + '_' + str(row['Utterance_ID'])],
                     'speaker': i % 2,
                }
        if len(dialog) < 1:
            print("Skip dialog with only 1 utterance.")
            continue
        dialogues.append(dialog)
        # if label is None:
        #     print("Warning: label is None")
        labels.append(label)
        emotions.append(emotion)
        embeddings.append(emb)
        speakers.append(speaker)

    return {'dialogue': dialogues,
            'label': labels,
            'emotion': emotions,
            'speaker': speakers,
            'embedding': embeddings,
            'knowledge': knowledges}


# def preprocess_df(dialog_df):
#     while True:
#         length = len(dialog_df)
#         if length < 2 or dialog_df.loc[length - 1, 'Speaker'] != dialog_df.loc[length - 2, 'Speaker']:
#             return dialog_df
#         else:
#             dialog_df.drop(length - 1, axis=0, inplace=True)


def process_meld(split, emb_dict, knowledge, cls_3=False):
    emotion_label_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
    sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    input_dir = 'data/'
    df = pd.read_csv(input_dir + f'meld/processed/{split}_full.csv')
    if knowledge != 'none':
        if knowledge == 'comet':
            k_df = pd.read_csv(input_dir + f'meld/knowledge/{split}_comet_knowledge.csv')
            k_df['Knowledge'] = k_df['Knowledge'].apply(ast.literal_eval)
        elif knowledge == 'feeling':
            k_df = pd.read_csv(input_dir + f'meld/knowledge/{split}_knowledge_full_dialogue_feeling.csv')
            k_df['Knowledge'] = k_df['Knowledge'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        elif knowledge == 'utterance':
            k_df = pd.read_csv(input_dir + f'meld/knowledge/{split}_knowledge_full_dialogue_chatbot.csv')
            k_df['Knowledge'] = k_df['Knowledge'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        else:
            raise "knowledge must be either none, comet, feeling or utterance"

        df = pd.merge(df, k_df, how='left', on=['Dialogue_ID', 'Utterance_ID'])

    knowledges = get_knowledge(df, knowledge)

    skipped_count = 0
    dialogues = []
    labels = []
    emotions = []
    speakers = []
    embeddings = []
    for item in list(df.groupby("Dialogue_ID")):
        dialog_id = item[0]
        dialog = []
        label = None
        emotion = []
        speaker = []
        emb = []
        item[1].sort_values(by=['Utterance_ID'], inplace=True)
        dialog_df = item[1].reset_index(drop=True)
        # dialog_df = preprocess_df(dialog_df)
        # display(dialog_df)
        for i, row in dialog_df.iterrows():
            if i != len(dialog_df) - 1:
                dialog.append(row['Utterance'])
                speaker.append(row['Speaker'])
                emb.append(emb_dict[str(dialog_id) + '_' + str(row['Utterance_ID'])])
                emotion.append(sentiment_map[row['Sentiment']] if cls_3 else emotion_label_map[row['Emotion']])
            else:
                label = {'utterance': row['Utterance'],
                         'emotion': sentiment_map[row['Sentiment']] if cls_3 else emotion_label_map[row['Emotion']],
                         'emb': emb_dict[str(dialog_id) + '_' + str(row['Utterance_ID'])],
                         'speaker': row['Speaker']
                         }
        # if len(dialog) < 1:
        #     print("Warning: no dialogue")
        if len(dialog) < 1:
            skipped_count += 1
            continue

        dialogues.append(dialog)
        labels.append(label)
        emotions.append(emotion)
        embeddings.append(emb)
        speakers.append(speaker)

    print("Skipped %d dialogues with too few dialogues." % skipped_count)
    return {'dialogue': dialogues,
            'label': labels,
            'emotion': emotions,
            'speaker': speakers,
            'embedding': embeddings,
            'knowledge': knowledges}

def process_emorynlp(split, emb_dict, knowledge, cls_3=False):
    emotion_label_map = {'Neutral': 0, 'Joyful': 1, 'Peaceful': 2, 'Powerful': 3,
                         'Scared': 4, 'Mad': 5, 'Sad': 6}
    sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    input_dir = 'data/'
    df = pd.read_csv(input_dir + f'emorynlp/processed/{split}_full.csv')
    if knowledge != 'none':
        if knowledge == 'comet':
            k_df = pd.read_csv(input_dir + f'emorynlp/knowledge/{split}_comet_knowledge.csv')
            k_df['Knowledge'] = k_df['Knowledge'].apply(ast.literal_eval)
        elif knowledge == 'feeling':
            k_df = pd.read_csv(input_dir + f'emorynlp/knowledge/{split}_knowledge_full_dialogue_feeling.csv')
            k_df['Knowledge'] = k_df['Knowledge'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        elif knowledge == 'utterance':
            k_df = pd.read_csv(input_dir + f'emorynlp/knowledge/{split}_knowledge_full_dialogue_chatbot.csv')
            k_df['Knowledge'] = k_df['Knowledge'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else None)
        else:
            raise "knowledge must be either none, comet, feeling or utterance"

        df = pd.merge(df, k_df, how='left', on=['Dialogue_ID', 'Utterance_ID'])

    knowledges = get_knowledge(df, knowledge)

    skipped_count = 0
    dialogues = []
    labels = []
    emotions = []
    speakers = []
    embeddings = []
    for item in list(df.groupby("Dialogue_ID")):
        dialog_id = item[0]
        dialog = []
        label = None
        emotion = []
        speaker = []
        emb = []
        item[1].sort_values(by=['Utterance_ID'], inplace=True)
        dialog_df = item[1].reset_index(drop=True)
        # dialog_df = preprocess_df(dialog_df)
        # display(dialog_df)
        for i, row in dialog_df.iterrows():
            if i != len(dialog_df) - 1:
                dialog.append(row['Utterance'])
                speaker.append(row['Speaker'])
                emb.append(emb_dict[str(dialog_id) + '_' + str(row['Utterance_ID'])])
                emotion.append(sentiment_map[row['Sentiment']] if cls_3 else emotion_label_map[row['Emotion']])
            else:
                label = {'utterance': row['Utterance'],
                         'emotion': sentiment_map[row['Sentiment']] if cls_3 else emotion_label_map[row['Emotion']],
                         'emb': emb_dict[str(dialog_id) + '_' + str(row['Utterance_ID'])],
                         'speaker': row['Speaker']
                         }
        # if len(dialog) < 1:
        #     print("Warning: no dialogue")
        if len(dialog) < 1:
            skipped_count += 1
            continue

        dialogues.append(dialog)
        labels.append(label)
        emotions.append(emotion)
        embeddings.append(emb)
        speakers.append(speaker)

    print("Skipped %d dialogues with too few dialogues." % skipped_count)
    return {'dialogue': dialogues,
            'label': labels,
            'emotion': emotions,
            'speaker': speakers,
            'embedding': embeddings,
            'knowledge': knowledges}

