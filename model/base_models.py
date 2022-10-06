import torch
import torch.nn as nn
from model.ei_roberta import EIRoberta
from model.dialogue_infer import DialogueInfer
from model.dialogue_gcn.DialogueGCN import DialogueGCN
from model.dialogue_rnn import DialogueRNN
from model.dialogue_crn import DialogueCRN
from transformers import AutoConfig, AutoModel

class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ei_roberta = EIRoberta()
        self.ei_roberta.over_write_embedding_forward()
        self.dialogue_infer = DialogueInfer(cfg.input_size, cfg.lstm_hidden_size)
        self.project = nn.Linear(self.cfg.lstm_hidden_size, self.cfg.hidden_size)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.cfg.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.cfg.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        inputs1, inputs2 = inputs['inputs1'], inputs['inputs2']
        outputs = self.dialogue_infer(**inputs1)
        last_hidden_states = outputs[0]
        # print(last_hidden_states.shape)
        stream_embedding = last_hidden_states
        # print(stream_embedding.shape)
        stream_embedding = self.project(stream_embedding)

        inputs2['additional_embedding'] = stream_embedding
        outputs = self.ei_roberta(**inputs2)
        last_hidden_states = outputs[0]

        # feature = torch.mean(last_hidden_states, 1)
        # weights = self.attention(last_hidden_states)
        # feature = torch.sum(weights * last_hidden_states, dim=1)
        feature = last_hidden_states[:, 0, :]

        # feature = last_hidden_states
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output

    def freeze_plm(self):
        for p in self.ei_roberta.parameters():
            p.requires_grad = False

    def unfreeze_plm(self):
        for p in self.ei_roberta.parameters():
            p.requires_grad = True


paras = {'wp': 5, 'wf': 5, 'device': 'cuda', 'n_speakers': 10, 'rnn': 'lstm', 'drop_rate': 0.0, 'class_weight': False}


class DialogueGCNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = DialogueGCN(paras)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(300, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.cfg.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        # print(inputs['speaker_tensor'])
        outputs = self.model(inputs)[torch.cumsum(inputs['text_len_tensor'], dim=0) - 1]
        # outputs = self.model(inputs)
        # print(outputs.shape)
        last_hidden_states = outputs
        feature = last_hidden_states
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


class DialogueInferModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = DialogueInfer(cfg.input_size, cfg.hidden_size)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.cfg.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.cfg.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        # weights = self.attention(last_hidden_states)
        # feature = torch.sum(weights * last_hidden_states, dim=1)
        # feature = last_hidden_states[:, 0, :]
        feature = last_hidden_states
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


class DialogueRNNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        # self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        self.model = DialogueRNN(1024, 100, 100, 100, listener_state=True, dropout=0.1)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.cfg.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.cfg.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states, emotions = outputs[0], outputs[1]
        seq_length, batch_size = last_hidden_states.shape[0], last_hidden_states.shape[1]
        last_hidden_states = last_hidden_states.permute(1, 0, 2)
        emotions = emotions.permute(1, 0, 2)
        # weights = self.attention(emotions)
        # feature = torch.sum(weights * emotions, dim=1) + last_hidden_states[:, -1, :]
        feature = last_hidden_states[:, -1, :] + emotions[:, -1, :]
        return feature

    def forward(self, inputs):
        # feature = self.batch_norm(self.feature(inputs))
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


class DialogueCRNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = DialogueCRN(base_layer=cfg.base_layer,
                                 input_size=cfg.input_size,
                                 hidden_size=cfg.hidden_size,
                                 n_speakers=cfg.n_speakers,
                                 dropout=0.2,
                                 cuda_flag=True,
                                 reason_steps=[2, 2]
                                 )
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(800, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.cfg.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        hidden_states = self.model(**inputs) # max_seq_length x batch_size x dim
        # print("hidden: " + str(hidden_states.shape))
        feature = torch.mean(hidden_states.permute(1, 0, 2), dim=1).squeeze(dim=1)
        # print("feature: " + str(feature.shape))
        return feature

    def forward(self, inputs):
        # feature = self.batch_norm(self.feature(inputs))
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


class RobertaFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained('roberta-large', output_hidden_states=True)
        self.model = AutoModel.from_pretrained('roberta-large', config=self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        # feature = last_hidden_states[:, 0, :]
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output

model_class_map = {
    'BaseModel': BaseModel,
    'DialogueRNN': DialogueRNNModel,
    'DialogueGCN': DialogueGCNModel,
    'DialogueInfer': DialogueInferModel,
    'DialogueCRN': DialogueCRNModel,
    'Extractor': RobertaFeatureExtractor
}