import torch.nn as nn
import torch

class GRUNetMultiLabel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocabSize, n_layers= 1, drop_prob=0):
        super(GRUNetMultiLabel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.emb = nn.Embedding(vocabSize, input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # Passage à travers le GRU
        out, _ = self.gru(self.emb(x))

        # On prend uniquement la dernière sortie temporelle
        out = self.fc(out)

        # Application de la sigmoïde pour obtenir des probabilités
        proba = torch.sigmoid(out)

        # Appliquer un seuil pour convertir les probabilités en prédictions de classe
        predicted_labels = (proba > 0.5).float()

        return proba, predicted_labels
