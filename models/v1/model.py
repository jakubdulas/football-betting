import torch
import torch.nn as nn

class PastAnalyser(nn.Module):
    def __init__(self, embedding_layer, d_model) -> None:
        super().__init__()
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(18, d_model, 3, batch_first=True)
        self.cnn = nn.Conv1d(5, 5, 3, padding=1)

        self.fc_block1 = nn.Sequential(
            nn.Linear(2*d_model, 3*d_model),
            nn.GELU(),
            nn.Linear(3*d_model, d_model),
            nn.GELU()
        )

        self.fc_block2 = nn.Sequential(
            nn.Linear(5*d_model, 5*d_model),
            nn.GELU(),
            nn.Linear(5*d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x):
        embedded_teams = self.embedding(x[:, :, :2].type(torch.int))
        flattened_embedded_teams = embedded_teams.flatten(-2)
        strength_comparisson = self.fc_block1(flattened_embedded_teams)
        x = self.cnn(x[:, :, 2:].type(torch.float32))
        x, _ = self.lstm(x)
        x = torch.add(strength_comparisson, x)
        x = torch.flatten(x, -2, -1)
        x = self.fc_block2(x)
        return x
    


class Model(nn.Module):
    def __init__(self, emb_size, d_model) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(emb_size+1, d_model)
        self.past_analyser = PastAnalyser(self.embedding, d_model)
        self.fc_block1 = nn.Sequential(
            nn.Linear(7, 2*d_model),
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
        embedded_teams = self.embedding(match[:, :, :1].type(torch.int))
        embedded_teams = torch.flatten(embedded_teams, -2)
        x = self.fc_block1(match[:, :, 1:].type(torch.float32))
        x = x + embedded_teams
        x = torch.flatten(x, -2)
        x = self.fc_block2(x)

        matches_ab_vector = self.past_analyser(matches_ab)
        matches_a_vector = self.past_analyser(matches_a)
        matches_b_vector = self.past_analyser(matches_b)

        x = torch.concat([x, matches_ab_vector, matches_a_vector, matches_b_vector], dim=-1)

        x = self.fc_block3(x)

        return x
