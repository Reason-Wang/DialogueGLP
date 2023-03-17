from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from model.com_pm.utils import make_batch_roberta_bert
import json


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
        elif self.cfg.knowledge == 'U_and_F':
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


class CogBartEIDataset(Dataset):
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
        label = self.labels[item]['emotion']
        speaker_names = self.speakers[item]
        return {"text": self.dialogues[item], "label": label,
                'speaker_names': speaker_names
                }


class CoMPMEIDataset(Dataset):
    def __init__(self, data, opt=None):
        super().__init__()
        self.dialogues = data['dialogue']
        self.labels = data['label']
        self.emotions = data['emotion']
        self.speakers = data['speaker']
        self.embeddings = data['embedding']
        self.context_speakers = []
        for label, ss in zip(self.labels, self.speakers):
            tmp_speakers = []
            speaker2idx_map = {}
            idx = 0
            for s in ss:
                if s not in speaker2idx_map:
                    speaker2idx_map[s] = idx
                    idx += 1
                tmp_speakers.append(speaker2idx_map[s])
            if label['speaker'] not in speaker2idx_map:
                speaker2idx_map[label['speaker']] = idx
            tmp_speakers.append(speaker2idx_map[label['speaker']])
            self.context_speakers.append(tmp_speakers)
    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, item):

        return {'context_speakers': self.context_speakers[item], 'context': self.dialogues[item], 'label': self.labels[item]['emotion']}


class DagERCEIDataset(Dataset):
    def __init__(self, data, opt=None):
        with open("model/dag_erc/speakers.json", 'r') as f:
            speaker_vocabs = json.load(f)
        self.speaker_vocab = speaker_vocabs[opt.dataset]
        # print(self.speaker_vocab)
        # self.label_vocab = label_vocab
        self.args = opt
        self.data = self.read(data)
        self.labels = data['label']
        print(len(self.data))

        self.len = len(self.data)

    def read(self, data):
        # process dialogue
        dialogs = []
        dd_name_map = {0: 'A', 1: 'B'}
        for utterances, speakers_names, features in zip(data['dialogue'], data['speaker'], data['embedding']):
            speakers = []
            for s in speakers_names:
                if s == 0 or s == 1:
                    s = dd_name_map[s]
                if 'Richard' in s and 'Date' in s:
                    s = "Richard's Date"
                speakers.append(self.speaker_vocab['stoi'][s])
            dialogs.append({
                'utterances': utterances,
                'speakers': speakers,
                'features': features
            })
        # random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            speaker
            length
            text
            label
        '''
        return {'feature': torch.FloatTensor(self.data[index]['features']),
                'speaker': self.data[index]['speakers'],
                'length': len(self.data[index]['features']),
                'utterance': self.data[index]['utterances'],
                'label': self.labels[index]['emotion']}

    def __len__(self):
        return self.len


dataset_map = {
    'BaseModel': BaseModelEIDataset,
    'DialogueInfer': DialogueInferEIDataset,
    'DialogueRNN': DialogueRNNEIDataset,
    'DialogueGCN': DialogueGCNEIDataset,
    'DialogueCRN': DialogueCRNEIDataset,
    'CogBart': CogBartEIDataset,
    'CoMPM': CoMPMEIDataset,
    'DAG': DagERCEIDataset,
    'Extractor': FeatureTuningDataset,
}


class BaseModelCollator:
    def __init__(self, opt, device):
        self.cfg = opt
        self.padding_emb = torch.zeros(opt.input_size, dtype=torch.float)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.sep = self.tokenizer.sep_token
        self.addressee_prefix = {1: "I: ", 0: "Other: "}
        self.printed = False
        self.k_knowledge = 3 if (opt.dataset == "emorynlp" and opt.knowledge == "feeling") else 10
        self.comet_k_num = 2 if self.cfg.dataset == "meld" else 3

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

                text = text + ' ' \
                       + ', '.join(ex['local_knowledges'][0][0][:self.comet_k_num]) + '. ' \
                       + ex['local_knowledges'][0][1][0] + '. '
                       # + ex['local_knowledges'][0][2][0]
                for a, u, k in zip(ex['local_speakers'][1:], ex['local_utterances'][1:], ex['local_knowledges'][1:]):
                    text = text + ' ' + self.sep + ' ' + self.addressee_prefix[a] + u + ' ' \
                           + ', '.join(k[0][:self.comet_k_num]) + '. ' \
                           + k[1][0] + '. '
                           # + k[2][0]
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
            elif self.cfg.knowledge == 'U_and_F':
                text = self.addressee_prefix[ex['local_speakers'][0]] + ex['local_utterances'][0]
                for a, u in zip(ex['local_speakers'][1:], ex['local_utterances'][1:]):
                    text = text + ' ' + self.sep + ' ' + self.addressee_prefix[a] + u
                klgs = ex['local_knowledges'][-1]
                for k in klgs[0][:2]:
                    text = text + ' ' + self.sep + ' ' + self.addressee_prefix[1] + k
                for k in klgs[1][:2]:
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


class CogBartCollator(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.max_seq_length = 128
        self.print = True
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base",
                                                       cache_dir=None,
                                                       use_fast=True)

    def get_bart_feature(self, sentence, tokenizer):
        # print(sentence)
        # stop
        inputs = tokenizer(sentence, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def __call__(self, batch):
        # embd: [tensor size BxD]
        # speaker_infos: [tensor size B] with 0 or 1 as values
        # labels: tensor size B
        texts = []
        for ex in batch:
            text = ''
            for s, t in zip(ex['speaker_names'][-5:], ex['text'][-5:]):
                text = text + str(s) + " : " + t + ' '
            texts.append(text)

        if self.print:
            print(texts)
            self.print = False
        # inputs = {
        #     'input_ids': pad_sequence([self.get_bart_feature(data[0], self.tokenizer)['input_ids'] for data in texts],
        #                               batch_first=True,
        #                               padding_value=1),
        #     'attention_mask': pad_sequence(
        #         [self.get_bart_feature(data[0], self.tokenizer)['attention_mask'] for data in texts],
        #         batch_first=True, padding_value=0)}
        inputs = self.tokenizer(texts, max_length=self.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        labels = torch.tensor([ex['label'] for ex in batch]).to(self.device)

        return inputs, labels


class CoMPMCollator(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.print = True
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base",
                                                       cache_dir=None,
                                                       use_fast=True)

    def __call__(self, batch):
        batch_input_tokens, batch_speaker_tokens = make_batch_roberta_bert(batch)
        inputs = {'batch_input_tokens': batch_input_tokens.to(self.device), 'batch_speaker_tokens': batch_speaker_tokens}
        labels = torch.tensor([ex['label'] for ex in batch]).to(self.device)

        return inputs, labels


class DagERCCollator(object):
    def __init__(self, cfg, device):
        self.args = cfg
        self.device = device

    def __call__(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d['length'] for d in data])
        features = pad_sequence([d['feature'] for d in data], batch_first = True) # (B, N, D)
        # labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value = -1) # (B, N )
        adj = self.get_adj_v1([d['speaker'] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d['speaker'] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d['length'] for d in data])
        speakers = pad_sequence([torch.LongTensor(d['speaker']) for d in data], batch_first = True, padding_value = -1)
        utterances = [d['utterance'] for d in data]

        inputs = {"features": features.to(self.device), "adj": adj.to(self.device), "s_mask": s_mask.to(self.device),
                  "s_mask_onehot": s_mask_onehot.to(self.device), "lengths": lengths.to(self.device),
                }
        labels = torch.tensor([d['label'] for d in data]).to(self.device)
        return inputs, labels

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i,j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i,j] = 1
                    if get_global_pred and get_local_pred:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):
                    a[i,j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt==self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
                        s_onehot[i,j,1] = 1
                    else:
                        s_onehot[i,j,0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)


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
    'CogBart': CogBartCollator,
    'CoMPM': CoMPMCollator,
    'DAG': DagERCCollator,
    'Extractor': FeatureTuningCollator
}