import torch.nn as nn
import torch
import torch.nn.functional as F


class BiLSTMConvolutionTagger(nn.Module):

    def __init__(self, input_dim,CharEmbdDim, hidden_dim, output_dim, vocabSize,vocabCharSize ,n_layers= 1 ):
        super(BiLSTMConvolutionTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.char_embedding_dim=CharEmbdDim



        # Word Embeddings
        self.word_embeddings = nn.Embedding(vocabSize, input_dim)

        # Character Embeddings
        self.char_embeddings = nn.Embedding(vocabCharSize, self.char_embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=self.char_embedding_dim, out_channels=self.char_embedding_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Combine word and character embeddings
        self.embedding_dim = input_dim + self.char_embedding_dim

        # LSTM Layer
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, bidirectional=True, num_layers=n_layers)

        # Dense Layer
        self.fc = nn.Linear( hidden_dim*2, output_dim)

        self.relu = nn.ReLU()
    def forward(self, x,c):
        # Word Embeddings
        word_embeds = self.word_embeddings(x)
        #print("Word Embeds size:", word_embeds.size())
        # Character Embeddings
        #print (c)
        char_embeds = F.embedding(c, self.char_embeddings.weight, padding_idx=0)

        char_conv = self.conv1d(char_embeds.permute(0, 2, 1))
        char_conv = self.relu(char_conv)
        char_conv = char_conv.permute(0, 2, 1)
        char_pooled, _ = torch.max(char_conv, dim=1)
        #print("char Embeds size:", word_embeds.size())


        # Concatenate word and character embeddings
        combined_embeds = word_embeds+ char_pooled
        #print("combined_embeds size:", combined_embeds.size())

        # LSTM Layer
        lstm_output,  (hidden_states, cell_states)= self.lstm(combined_embeds)

        logits = self.fc(self.relu(lstm_output))
        return logits , (hidden_states, cell_states)



