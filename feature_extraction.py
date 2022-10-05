from model.base_models import model_class_map
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from utils.options import Options



def get_meld_data(split, cls_3=False):
    input_dir = 'data/'
    if split == 'train':
        df = pd.read_csv(input_dir + "meld/processed/train_full.csv")
    elif split == 'val':
        df = pd.read_csv(input_dir + "meld/processed/val_full.csv")
    elif split == 'test':
        df = pd.read_csv(input_dir + "meld/processed/test_full.csv")

    emotion_label_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
    sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    labels = []
    for i, row in df.iterrows():
        labels.append(sentiment_map[row['Sentiment']] if cls_3 else emotion_label_map[row['Emotion']])
    return df['Utterance'].values, labels, df['Dialogue_ID'].values, df['Utterance_ID'].values


def get_daily_dialogue_dialog(split):
    input_dir = 'data/'
    emotion_map = {'no emotion': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5,
                   'surprise': 6}
    utterances = []
    labels = []
    dialogue_ids = []
    utterance_ids = []
    df = pd.read_csv(input_dir + f"daily_dialogue/{split}.csv")
    for i, row in df.iterrows():
        utterances.append(row['Utterance'])
        labels.append(emotion_map[row['Emotion']])
        dialogue_ids.append(row['Dialogue_ID'])
        utterance_ids.append(row['Utterance_ID'])

    return utterances, labels, dialogue_ids, utterance_ids

def get_emorynlp_data(split, cls_3=False):
    input_dir = 'data/'
    if split == 'train':
        df = pd.read_csv(input_dir + "emorynlp/processed/train_full.csv")
    elif split == 'val':
        df = pd.read_csv(input_dir + "emorynlp/processed/val_full.csv")
    elif split == 'test':
        df = pd.read_csv(input_dir + "emorynlp/processed/test_full.csv")
    emotion_label_map = {'Neutral': 0, 'Joyful': 1, 'Peaceful': 2, 'Powerful': 3,
                         'Scared': 4, 'Mad': 5, 'Sad': 6}
    sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    labels = []
    for i, row in df.iterrows():
        labels.append(sentiment_map[row['Sentiment']] if cls_3 else emotion_label_map[row['Emotion']])
    return df['Utterance'].values, labels, df['Dialogue_ID'].values, df['Utterance_ID'].values

class FeatureInferDataset(Dataset):
    def __init__(self, utterances, emotions, dialog_ids, utterance_ids):
        super().__init__()
        self.utterances = utterances
        self.emotions = emotions
        self.dialog_id = dialog_ids
        self.utterance_id = utterance_ids

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, item):
        return {"text": self.utterances[item], "dialog": self.dialog_id[item],
                "utterance": self.utterance_id[item]}


class Collator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    def __call__(self, batch):
        inputs = self.tokenizer(
            [ex['text'] for ex in batch],
            max_length=self.cfg.max_len,
            return_tensors='pt',
            padding=True,
            truncation=True,
            return_offsets_mapping=False)
        dialogs = torch.tensor([ex['dialog'] for ex in batch], dtype=torch.long)
        utterances = torch.tensor([ex['utterance'] for ex in batch], dtype=torch.long)

        return inputs, dialogs, utterances




def inference_fn(model, opt, ds, device):
    collator = Collator(opt)
    dataloader =  DataLoader(ds,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=opt.num_workers,
                    pin_memory=False,
                    drop_last=False)
    model.eval()
    model.to(device)
    features = []
    dialogues = []
    utterances = []
    with torch.no_grad():
        for step, (inputs, dialog, utterance) in enumerate(dataloader):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            feature = model.feature(inputs)
            features.extend(list(feature.detach().cpu().numpy()))
            dialogues.extend(dialog.detach().cpu().numpy().tolist())
            utterances.extend(utterance.detach().cpu().numpy().tolist())
    return features, dialogues, utterances

def get_emb_dict(features, dialogues, utterances):
    emb_dict = {}
    for feature, dialogue, utterance in zip(features, dialogues, utterances):
        emb_dict[str(dialogue)+'_'+str(utterance)] = feature
    return emb_dict


def main():
    options = Options()
    options.add_extractor_options()
    opt = options.parse()[0]
    if opt.cls_3:
        opt.target_size = 3

    INPUT_DIR = 'data/ckpts/' + opt.name + '/'
    model_state = torch.load(INPUT_DIR + f"{opt.model}.pth")
    model = model_class_map[opt.model](opt)
    model.load_state_dict(model_state)
    print("Loaded model state from "+INPUT_DIR)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if opt.dataset == 'meld':
        utterances, labels, dialogue_ids, utterance_ids = get_meld_data('train', opt.cls_3)
        train_ds = FeatureInferDataset(utterances, labels, dialogue_ids, utterance_ids)
        utterances, labels, dialogue_ids, utterance_ids = get_meld_data('val', opt.cls_3)
        val_ds = FeatureInferDataset(utterances, labels, dialogue_ids, utterance_ids)
        utterances, labels, dialogue_ids, utterance_ids = get_meld_data('test', opt.cls_3)
        test_ds = FeatureInferDataset(utterances, labels, dialogue_ids, utterance_ids)
        OUTPUT_DIR = 'data/meld/processed/'
    elif opt.dataset == 'daily_dialogue':
        utterances, labels, dialogue_ids, utterance_ids = get_daily_dialogue_dialog('train')
        train_ds = FeatureInferDataset(utterances, labels, dialogue_ids, utterance_ids)
        utterances, labels, dialogue_ids, utterance_ids = get_daily_dialogue_dialog('val')
        val_ds = FeatureInferDataset(utterances, labels, dialogue_ids, utterance_ids)
        utterances, labels, dialogue_ids, utterance_ids = get_daily_dialogue_dialog('test')
        test_ds = FeatureInferDataset(utterances, labels, dialogue_ids, utterance_ids)
        OUTPUT_DIR = 'data/daily_dialogue/processed/'
    elif opt.dataset == 'emorynlp':
        utterances, labels, dialogue_ids, utterance_ids = get_emorynlp_data('train')
        train_ds = FeatureInferDataset(utterances, labels, dialogue_ids, utterance_ids)
        utterances, labels, dialogue_ids, utterance_ids = get_emorynlp_data('val')
        val_ds = FeatureInferDataset(utterances, labels, dialogue_ids, utterance_ids)
        utterances, labels, dialogue_ids, utterance_ids = get_emorynlp_data('test')
        test_ds = FeatureInferDataset(utterances, labels, dialogue_ids, utterance_ids)
        OUTPUT_DIR = 'data/emorynlp/processed/'

    if opt.dataset == 'meld':
        features, dialogues, utterances = inference_fn(model, opt, val_ds, device)
        dev_emb_dict = get_emb_dict(features, dialogues, utterances)
        with open(f'data/meld/processed/dev_cls3_{opt.metric}.pkl' if opt.cls_3 else f'data/meld/processed/dev_{opt.metric}.pkl', 'wb') as handle:
            pickle.dump(dev_emb_dict, handle)

        features, dialogues, utterances = inference_fn(model, opt, train_ds, device)
        train_emb_dict = get_emb_dict(features, dialogues, utterances)
        with open(f'data/meld/processed/train_cls3_{opt.metric}.pkl' if opt.cls_3 else f'data/meld/processed/train_{opt.metric}.pkl',
                  'wb') as handle:
            pickle.dump(train_emb_dict, handle)

        features, dialogues, utterances = inference_fn(model, opt, test_ds, device)
        test_emb_dict = get_emb_dict(features, dialogues, utterances)
        with open(f'data/meld/processed/test_cls3_{opt.metric}.pkl' if opt.cls_3 else f'data/meld/processed/test_{opt.metric}.pkl', 'wb') as handle:
            pickle.dump(test_emb_dict, handle)

    elif opt.dataset == 'daily_dialogue':
        features, dialogues, utterances = inference_fn(model, opt, val_ds, device)
        dev_emb_dict = get_emb_dict(features, dialogues, utterances)
        with open(OUTPUT_DIR + f'dev_{opt.metric}.pkl', 'wb') as handle:
            pickle.dump(dev_emb_dict, handle)

        features, dialogues, utterances = inference_fn(model, opt, train_ds, device)
        train_emb_dict = get_emb_dict(features, dialogues, utterances)
        with open(OUTPUT_DIR + f'train_{opt.metric}.pkl', 'wb') as handle:
            pickle.dump(train_emb_dict, handle)

        features, dialogues, utterances = inference_fn(model, opt, test_ds, device)
        test_emb_dict = get_emb_dict(features, dialogues, utterances)
        with open(OUTPUT_DIR + f'test_{opt.metric}.pkl', 'wb') as handle:
            pickle.dump(test_emb_dict, handle)
    elif opt.dataset == 'emorynlp':
        features, dialogues, utterances = inference_fn(model, opt, val_ds, device)
        dev_emb_dict = get_emb_dict(features, dialogues, utterances)
        with open(f'data/emorynlp/processed/dev_cls3_{opt.metric}.pkl' if opt.cls_3 else f'data/emorynlp/processed/dev_{opt.metric}.pkl', 'wb') as handle:
            pickle.dump(dev_emb_dict, handle)

        features, dialogues, utterances = inference_fn(model, opt, train_ds, device)
        train_emb_dict = get_emb_dict(features, dialogues, utterances)
        with open(f'data/emorynlp/processed/train_cls3_{opt.metric}.pkl' if opt.cls_3 else f'data/emorynlp/processed/train_{opt.metric}.pkl', 'wb') as handle:
            pickle.dump(train_emb_dict, handle)

        features, dialogues, utterances = inference_fn(model, opt, test_ds, device)
        test_emb_dict = get_emb_dict(features, dialogues, utterances)
        with open(f'data/emorynlp/processed/test_cls3_{opt.metric}.pkl' if opt.cls_3 else f'data/emorynlp/processed/test_{opt.metric}.pkl', 'wb') as handle:
            pickle.dump(test_emb_dict, handle)

if __name__ == '__main__':
    main()