import torch
import torch.nn as nn

class DialogueInfer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_store = nn.LSTMCell(input_size, hidden_size)
        self.lstm_affect = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, embs, speakers):
        batch_size, input_size = embs[0].shape[0], embs[0].shape[1]
        h_o = torch.zeros(batch_size, self.hidden_size, dtype=torch.float).to(self.device)
        c_o = torch.zeros(batch_size, self.hidden_size, dtype=torch.float).to(self.device)
        h_n = h_o
        c_n = c_o
        for emb, lambd in zip(embs, speakers):
            # print(emb)
            # print(emb.shape)
            # stop
            # print(emb.shape, lambd.shape)
            h_o = h_n
            c_o = c_n
            h_store, c_store = self.lstm_store(emb, (h_o, c_o))
            h_affect, c_affect = self.lstm_affect(emb, (h_o, c_o))
            h_n = lambd * h_store + (1 - lambd) * h_affect
            c_n = lambd * c_store + (1 - lambd) * c_affect
        # return F.relu(self.fc(h_n)), c_n
        return self.fc(h_n), c_n