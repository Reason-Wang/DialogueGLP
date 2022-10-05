import pickle
import pandas as pd
from data.process_data import process_daily_dialog, process_meld, process_emorynlp

def load_data(dataset, split, metric, knowledge, cls_3=False):
    input_dir = 'data/'
    data_path = ''

    if split == 'train':
        data_path = input_dir + dataset + f"/processed/train_{'cls3_' if cls_3 else ''}{metric}.pkl"
        with open(data_path, 'rb') as f:
            emb_dict = pickle.load(f)
    elif split == 'val':
        data_path = input_dir + dataset + f"/processed/dev_{'cls3_' if cls_3 else ''}{metric}.pkl"
        with open(data_path, 'rb') as f:
            emb_dict = pickle.load(f)
    elif split == 'test':
        data_path = input_dir + dataset + f"/processed/test_{'cls3_' if cls_3 else ''}{metric}.pkl"
        with open(data_path, 'rb') as f:
            emb_dict = pickle.load(f)

    print("Loaded embedding from %s" % data_path)
    data = None
    if dataset == 'daily_dialogue':
        data = process_daily_dialog(split, emb_dict, knowledge)
    elif dataset == 'meld':
        data = process_meld(split, emb_dict, knowledge, cls_3)
    elif dataset == 'emorynlp':
        data = process_emorynlp(split, emb_dict, knowledge, cls_3)

    return data

def load_feature_data(dataset, split, cls_3=False):
    input_dir = 'data/'
    if dataset == 'daily_dialogue':
        emotion_map = {'no emotion': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
        utterances = []
        labels = []
        df = pd.read_csv(input_dir + f"daily_dialogue/{split}.csv")

        for i, row in df.iterrows():
            utterances.append(row['Utterance'])
            labels.append(emotion_map[row['Emotion']])

    elif dataset == 'meld':
        if split == 'train':
            df = pd.read_csv(input_dir + "meld/train_sent_emo.csv")
        elif split == 'val':
            df = pd.read_csv(input_dir + "meld/dev_sent_emo.csv")
        elif split == 'test':
            df = pd.read_csv(input_dir + "meld/test_sent_emo.csv")

        emotion_label_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
        sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        emotions = df['Sentiment'].values if cls_3 else df['Emotion'].values
        labels = []
        for emotion in emotions:
            labels.append(sentiment_map[emotion] if cls_3 else emotion_label_map[emotion])

        utterances = df['Utterance'].values

    elif dataset == 'emorynlp':
        if split == 'train':
            df = pd.read_csv(input_dir + "emorynlp/train.csv")
        elif split == 'val':
            df = pd.read_csv(input_dir + "emorynlp/dev.csv")
        elif split == 'test':
            df = pd.read_csv(input_dir + "emorynlp//test.csv")
        emotion_label_map = {'Neutral': 0, 'Joyful': 1, 'Peaceful': 2, 'Powerful': 3,
                             'Scared': 4, 'Mad': 5, 'Sad': 6}
        sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        emotions = df['Sentiment'].values if cls_3 else df['Emotion'].values
        labels = []
        for emotion in emotions:
            labels.append(sentiment_map[emotion] if cls_3 else emotion_label_map[emotion])

        utterances = df['Utterance'].values

    return utterances, labels