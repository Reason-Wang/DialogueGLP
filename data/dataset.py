from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class FeatureTuningDataset(Dataset):
    def __init__(self, utterances, emotions):
        super().__init__()
        self.utterances = utterances
        self.emotions = emotions

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, item):
        return {"text": self.utterances[item], "label": self.emotions[item]}

class BaseModelEIDataset(Dataset):
    def __init__(self, data, opt):
        super().__init__()
        self.dialogues = data['dialogue']
        self.labels = data['label']
        self.emotions = data['emotion']
        self.speakers = data['speaker']
        self.embeddings = data['embedding']
        self.knowledges = data['knowledge']
        self.cfg = opt

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, item):
        emotions = self.emotions[item]
        label = self.labels[item]['emotion']
        speakers = [1 if speaker == self.labels[item]['speaker'] else 0 for speaker in self.speakers[item]]
        local_utterances = self.dialogues[item][-self.cfg.n_sentences:]
        local_speakers = speakers[-self.cfg.n_sentences:]
        if self.cfg.knowledge == 'comet':
            local_knowledges = self.knowledges[item][-self.cfg.n_knowledges:]
        elif self.cfg.knowledge == 'feeling':
            local_knowledges = self.knowledges[item][-self.cfg.n_knowledges:]
        elif self.cfg.knowledge == 'utterance':
            local_knowledges = self.knowledges[item][-self.cfg.n_knowledges:]
        else:
            local_knowledges = None

        return {"text": self.dialogues[item], "knowledges": self.knowledges[item],
                "label": label, "speakers": speakers,
                "emb": self.embeddings[item], "emotions": emotions,
                "local_utterances": local_utterances, 'local_speakers': local_speakers,
                "local_knowledges": local_knowledges}


class DialogueInferEIDataset(Dataset):
    def __init__(self, data, opt=None):
        super().__init__()
        self.dialogues = data['dialogue']
        self.labels = data['label']
        self.emotions = data['emotion']
        self.speakers = data['speaker']
        self.embeddings = data['embedding']

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, item):
        emotions = self.emotions[item]
        label = self.labels[item]['emotion']
        speakers = [1 if speaker == self.labels[item]['speaker'] else 0 for speaker in self.speakers[item]]
        return {"text": self.dialogues[item], "label": label,
                "speakers": speakers, "emb": self.embeddings[item],
                "emotions": emotions}


class DialogueRNNEIDataset(Dataset):
    def __init__(self, data, opt=None):
        super().__init__()
        self.dialogues = data['dialogue']
        self.labels = data['label']
        self.emotions = data['emotion']
        self.speakers = data['speaker']
        self.embeddings = data['embedding']

        self.speaker_idx = []
        for dialog_speakers in self.speakers:
            dialog_speakers_idx = []
            speaker_map = {}
            idx = -1
            for speaker in dialog_speakers:
                if speaker not in speaker_map:
                    idx += 1
                    speaker_map[speaker] = idx
                one_hot_idx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # print(speaker, speaker_map[speaker])
                one_hot_idx[speaker_map[speaker]] = 1
                dialog_speakers_idx.append(one_hot_idx)
            self.speaker_idx.append(dialog_speakers_idx)

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, item):
        emotions = self.emotions[item]
        label = self.labels[item]['emotion']
        speakers = [1 if speaker == self.labels[item]['speaker'] else 0 for speaker in self.speakers[item]]
        return {"text": self.dialogues[item], "label": label,
                "speakers": speakers, "emb": self.embeddings[item],
                "emotions": emotions, "speaker_idx": self.speaker_idx[item]}

class DialogueCRNEIDataset(Dataset):
    def __init__(self, data, opt=None):
        super().__init__()
        self.dialogues = data['dialogue']
        self.labels = data['label']
        self.emotions = data['emotion']
        self.speakers = data['speaker']
        self.embeddings = data['embedding']

        self.speaker_idx = []
        for dialog_speakers in self.speakers:
            dialog_speakers_idx = []
            speaker_map = {}
            idx = -1
            for speaker in dialog_speakers:
                if speaker not in speaker_map:
                    idx += 1
                    speaker_map[speaker] = idx
                one_hot_idx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # print(speaker, speaker_map[speaker])
                one_hot_idx[speaker_map[speaker]] = 1
                dialog_speakers_idx.append(one_hot_idx)
            self.speaker_idx.append(dialog_speakers_idx)

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, item):
        emotions = self.emotions[item]
        label = self.labels[item]['emotion']
        speakers = [1 if speaker == self.labels[item]['speaker'] else 0 for speaker in self.speakers[item]]
        return {"text": self.dialogues[item], "label": label,
                "speakers": speakers, "emb": self.embeddings[item],
                "emotions": emotions, "speaker_idx": self.speaker_idx[item]}


class DialogueGCNEIDataset(Dataset):
    def __init__(self, data, opt=None):
        super().__init__()
        self.dialogues = data['dialogue']
        self.labels = data['label']
        self.emotions = data['emotion']
        self.speakers = data['speaker']
        self.embeddings = data['embedding']

        self.speaker_idx = []
        for dialog_speakers in self.speakers:
            speaker_map = {}
            speaker_idxs = []
            idx = -1
            for speaker in dialog_speakers:
                if speaker not in speaker_map:
                    idx += 1
                    speaker_map[speaker] = idx
                speaker_idxs.append(speaker_map[speaker])
            self.speaker_idx.append(speaker_idxs)

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, item):
        emotions = self.emotions[item]
        label = self.labels[item]['emotion']
        speakers = [1 if speaker == self.labels[item]['speaker'] else 0 for speaker in self.speakers[item]]
        return {"text": self.dialogues[item], "label": label,
                "speakers": speakers, "emb": self.embeddings[item],
                "emotions": emotions, "speaker_idx": self.speaker_idx[item]}

dataset_map = {
    'BaseModel': BaseModelEIDataset,
    'DialogueInfer': DialogueInferEIDataset,
    'DialogueRNN': DialogueRNNEIDataset,
    'DialogueGCN': DialogueGCNEIDataset,
    'DialogueCRN': DialogueCRNEIDataset,
    'Extractor': FeatureTuningDataset,
}


class BaseModelCollator():
    def __init__(self, opt, device):
        self.cfg = opt
        self.padding_emb = torch.zeros(opt.input_size, dtype=torch.float)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.sep = self.tokenizer.sep_token
        self.addressee_prefix = {1: "I: ", 0: "Other: "}
        self.printed = False
        self.k_knowledge = 3 if (opt.dataset == "emorynlp" and opt.knowledge == "feeling") else 10
        self.comet_k_num = 2 if self.cfg.dataset == "meld" else 5

    def __call__(self, batch):
        # embd: [tensor size BxD]
        # speaker_infos: [tensor size B] with 0 or 1 as values
        # labels: tensor size B
        max_len = max([len(ex['emb']) for ex in batch])
        speaker_infos = [torch.tensor([ex['speakers'][i] if i < len(ex['speakers']) else 0 for ex in batch]).unsqueeze(1).to(self.device) for i in range(max_len)]

        embs = [torch.stack(
            [torch.tensor(ex['emb'][i], dtype=torch.float) if i < len(ex['emb']) else self.padding_emb for ex in
             batch]).to(self.device) for i in range(max_len)]
        # print(embs)
        labels = torch.tensor([ex['label'] for ex in batch]).to(self.device)

        inputs1 = {'embs': embs, 'speakers': speaker_infos}

        texts = []

        for ex in batch:
            if self.cfg.knowledge == 'comet':
                text = self.addressee_prefix[ex['local_speakers'][0]] + ex['local_utterances'][0]

                text = text + ' ' + " I am " \
                       + ', '.join(ex['local_knowledges'][0][0][:self.comet_k_num]) + '. I am ' \
                       + ex['local_knowledges'][0][1][0] + '. ' \
                       + ex['local_knowledges'][0][2][0]
                for a, u, k in zip(ex['local_speakers'][1:], ex['local_utterances'][1:], ex['local_knowledges'][1:]):
                    text = text + ' ' + self.sep + ' ' + self.addressee_prefix[a] + u + ' ' + " I am " \
                           + ', '.join(k[0][:self.comet_k_num]) + '. | I am ' \
                           + k[1][0] + '. | ' \
                           + k[2][0]
            elif self.cfg.knowledge == 'utterance':
                text = self.addressee_prefix[ex['local_speakers'][0]] + ex['local_utterances'][0]
                for a, u in zip(ex['local_speakers'][1:], ex['local_utterances'][1:]):
                    text = text + ' ' + self.sep + ' ' + self.addressee_prefix[a] + u
                for klgs in ex['local_knowledges']:
                    for k in klgs[:self.k_knowledge]:
                        text = text + ' ' + self.sep + ' ' + self.addressee_prefix[1] + k
            elif self.cfg.knowledge == 'feeling':
                text = self.addressee_prefix[ex['local_speakers'][0]] + ex['local_utterances'][0]
                for a, u in zip(ex['local_speakers'][1:], ex['local_utterances'][1:]):
                    text = text + ' ' + self.sep + ' ' + self.addressee_prefix[a] + u
                for klgs in ex['local_knowledges']:
                    for k in klgs[:self.k_knowledge]:
                        text = text + ' ' + self.sep + ' ' + k
            else:
                text = self.addressee_prefix[ex['local_speakers'][0]] + ex['local_utterances'][0]
                for a, u in zip(ex['local_speakers'][1:], ex['local_utterances'][1:]):
                    text = text + ' ' + self.sep + ' ' + self.addressee_prefix[a] + u

            texts.append(text)
        if not self.printed:
            print(texts)
            self.printed = True

        inputs2 = self.tokenizer(
            texts,
            max_length=self.cfg.max_len,
            return_tensors='pt',
            padding=True,
            truncation=True,
            return_offsets_mapping=False)
        for k, v in inputs2.items():
            inputs2[k] = v.to(self.device)

        return {'inputs1': inputs1, 'inputs2': inputs2}, labels


class DialogueInferCollator(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.padding_emb = torch.zeros(cfg.input_size, dtype=torch.float)
        self.device = device

    def __call__(self, batch):
        # embd: [tensor size BxD]
        # speaker_infos: [tensor size B] with 0 or 1 as values
        # labels: tensor size B
        max_len = max([len(ex['emb']) for ex in batch])
        speaker_infos = [
            torch.tensor([ex['speakers'][i] if i < len(ex['speakers']) else 0 for ex in batch]).unsqueeze(1).to(self.device)
            for i in range(max_len)]
        # print(speaker_infos)

        embs = [torch.stack(
            [torch.tensor(ex['emb'][i], dtype=torch.float) if i < len(ex['emb']) else self.padding_emb for ex in
             batch]).to(self.device) for i in range(max_len)]
        labels = torch.tensor([ex['label'] for ex in batch]).to(self.device)

        return {'embs': embs, 'speakers': speaker_infos}, labels


class DialogueRNNCollator(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.padding_emb = torch.zeros(cfg.input_size)
        self.device = device

    def __call__(self, batch):

        embs = pad_sequence([torch.tensor([ex['emb'][i] for i in range(len(ex['emb']))]) for ex in batch]).to(self.device)
        qmask = pad_sequence([torch.tensor([ex['speaker_idx'][i] for i in range(len(ex['emb']))]) for ex in batch]).to(
            self.device)
        labels = torch.tensor([ex['label'] for ex in batch]).to(self.device)

        return {'U': embs, 'qmask': qmask}, labels

class DialogueCRNCollator(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.padding_emb = torch.zeros(cfg.input_size)
        self.device = device

    def __call__(self, batch):

        lens = [len(ex['emb']) for ex in batch]
        embs = pad_sequence([torch.tensor([ex['emb'][i] for i in range(len(ex['emb']))]) for ex in batch]).to(self.device)
        qmask = pad_sequence([torch.tensor([ex['speaker_idx'][i] for i in range(len(ex['emb']))]) for ex in batch]).to(
            self.device)
        labels = torch.tensor([ex['label'] for ex in batch]).to(self.device)

        # print("embs: " + str(embs.shape))
        # print("qmask: " + str(qmask.shape))
        return {'U': embs, 'qmask': qmask, 'seq_lengths': lens}, labels


class DialogueGCNCollator(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.padding_emb = torch.zeros(cfg.input_size, dtype=torch.float)

    def __call__(self, batch):
        # embd: [tensor size BxD]
        # speaker_infos: [tensor size B] with 0 or 1 as values
        # labels: tensor size B
        max_len = max([len(ex['emb']) for ex in batch])
        speaker_infos = [
            torch.tensor([ex['speakers'][i] if i < len(ex['speakers']) else 0 for ex in batch]).unsqueeze(1).to(self.device)
            for i in range(max_len)]

        lens = torch.tensor([len(ex['emb']) for ex in batch])
        embs = pad_sequence([torch.tensor(np.array([ex['emb'][i] for i in range(len(ex['emb']))])) for ex in batch],
                            batch_first=True).to(self.device)
        speakers = pad_sequence(
            [torch.tensor(np.array([ex['speaker_idx'][i] for i in range(len(ex['speaker_idx']))])) for ex in batch],
            batch_first=True).to(self.device)
        labels = torch.tensor([ex['label'] for ex in batch]).to(self.device)
        # print(speakers)
        return {'text_len_tensor': lens, 'text_tensor': embs, 'speaker_tensor': speakers}, labels


class FeatureTuningCollator(object):
    def __init__(self, opt, tokenizer):
        self.cfg = opt
        self.tokenizer = tokenizer
        self.sep = self.tokenizer.sep_token

    def __call__(self, batch):
        input_texts = []
        for ex in batch:
            input_text = ex['text']
            input_texts.append(input_text)
        inputs = self.tokenizer(
            input_texts,
            max_length=self.cfg.max_len,
            return_tensors='pt',
            padding=True,
            truncation=True,
            return_offsets_mapping=False)
        labels = torch.tensor([ex['label'] for ex in batch], dtype=torch.long)

        return inputs, labels


collator_map = {
    'BaseModel': BaseModelCollator,
    'DialogueInfer': DialogueInferCollator,
    'DialogueRNN': DialogueRNNCollator,
    'DialogueGCN': DialogueGCNCollator,
    'DialogueCRN': DialogueCRNCollator,
    'Extractor': FeatureTuningCollator
}