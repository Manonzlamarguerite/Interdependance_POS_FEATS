import torch.nn as nn
import torch

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocabSize, n_layers= 1, drop_prob=0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dim_emb = intput_dim

        self.emb = nn.Embedding(vocabSize, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(2*hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, h = self.lstm(self.emb(self.get_char_wise_word_embedding(x)))
        out = self.fc(self.relu(out))
        return out, h

    def get_char_wise_word_embedding(word, vocab_character, kernel_size=3):
        char_to_idx_map = {char: idx for idx, char in enumerate(vocab_character)}

        ohe_characters = torch.eye(n=len(vocab_character))

        words = word.split()  # split the sentence into words

        max_length = max([len(word) for word in words]) or 1

        ohe_words = torch.empty(size=(0, len(vocab_character), max_length))

        for word in words:
            idx_representation = [char_to_idx_map[char] for char in word]
            ohe_representation = ohe_characters[idx_representation].T
            padded_ohe_representation = F.pad(ohe_representation, (0, max_length-len(word)))
            ohe_words = torch.cat((ohe_words, padded_ohe_representation.unsqueeze(dim=0)))

        convolution_layer = nn.Conv1d(in_channels=len(vocab_character), out_channels=self.dim_emb, kernel_size=kernel_size, bias=False)
        activation_layer = nn.Tanh()
        max_pooling_layer = nn.MaxPool1d(kernel_size=max_length-kernel_size+1)

        conv_out = convolution_layer(ohe_words)
        activation_out = activation_layer(conv_out)
        max_pool_out = max_pooling_layer(activation_out)

        return max_pool_out.squeeze()
