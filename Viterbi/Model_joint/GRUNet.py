import torch
import torch.nn as nn
import torch.nn.functional as F
from bigrams import Bigrams
from viterbi import Viterbi

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocabSize, bigramFile, n_layers= 1, drop_prob=0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.nb_classe = output_dim

        # Récupération des valeurs des probas des Bigrammes
        big = Bigrams()
        big.load(bigramFile)
        self.big = big

        self.emb = nn.Embedding(vocabSize, input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward_train(self, x):
        # Application du GRU pour les proba d'emitions
        out, h = self.gru(self.emb(x))
        out = self.fc(self.relu(out))
        # print("out GRU",len(out), len(out[0]), len(out[1]))
        proba_emit = F.softmax(out, dim=1)

        # Viterbi
        # Initialisation du treillis
        treillis = Viterbi(self.nb_classe, self.big.initialProb, proba_emit.detach().numpy(), self.big.bigramMatrix, x)
        # Calcul du treillis complet
        treillis.calcul_delta()
        # Calcul de la suite d'étiquette la plus probable
        best_label = treillis.proba_label()
        # out = torch.from_numpy(best_label)

        return out, h

    def forward_test(self, x):
        out, h = self.gru(self.emb(x))
        out = self.fc(self.relu(out))
        return out, h
