import torch
import torch.nn as nn


class KQVSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.kqv = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

    def forward(self, x):
        k, q, v = [l(x) for l in self.kqv]
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        x = torch.matmul(attention_weights, v)
        return x
    
class TeamsComparissonModel(nn.Module):
    def __init__(self, embedding_layer, d_model):
        super().__init__()
        self.embedding = embedding_layer
        self.kqv_self_attention = KQVSelfAttention(d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.kqv_self_attention(x)
        x = torch.sum(x, dim=-2)/2
        return x
    
class PastAnalyser(nn.Module):
    def __init__(self, embedding_layer, d_model) -> None:
        super().__init__()
        self.teams_comparisson_model = TeamsComparissonModel(embedding_layer, d_model)
        self.lstm1 = nn.LSTM(18, d_model, 3, batch_first=True)
        self.lstm2 = nn.LSTM(2*d_model, d_model, 3, batch_first=True)
        self.cnn = nn.Conv1d(5, 5, 3, padding=1)

        self.fc_block1 = nn.Sequential(
            nn.Linear(2*d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, 2*d_model),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.teams_comparisson_model(x[:, :, :2].type(torch.int))

        x2 = self.cnn(x[:, :, 2:].type(torch.float32))
        x2, _ = self.lstm1(x2)
        x = torch.concat([x1, x2], -1)
        x = self.fc_block1(x)
        x, _ = self.lstm2(x)
        return x[:, -1, :]
    
    
class Model(nn.Module):
    def __init__(self, emb_size, d_model) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(emb_size, d_model)
        self.past_analyser = PastAnalyser(self.embedding, d_model)
        self.teams_comparisson_model = TeamsComparissonModel(self.embedding, d_model)
        self.stats_comparisson = KQVSelfAttention(7)

        self.fc_block1 = nn.Sequential(
            nn.Linear(7*2, 2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
        )

        self.fc_block2 = nn.Sequential(
            nn.Linear(2*d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.ReLU(),
        )

        self.fc_block3 = nn.Sequential(
            nn.Linear(4*d_model, 8*d_model),
            nn.ReLU(),
            nn.Linear(8*d_model, 3),
            nn.Softmax(-1),
        )

    def forward(self, match, matches_ab, matches_a, matches_b):
        x1 = self.teams_comparisson_model(torch.squeeze(match[:, :, :1], -1).type(torch.int))

        x2 = self.stats_comparisson(match[:, :, 1:].type(torch.float32))
        x2 = self.fc_block1(torch.flatten(x2, -2))

        x = torch.concat([x1, x2], -1)
        x = self.fc_block2(x)

        matches_ab_vector = self.past_analyser(matches_ab)
        matches_a_vector = self.past_analyser(matches_a)
        matches_b_vector = self.past_analyser(matches_b)

        x = torch.concat([x, matches_ab_vector, matches_a_vector, matches_b_vector], dim=-1)
        x = self.fc_block3(x)
        return x
