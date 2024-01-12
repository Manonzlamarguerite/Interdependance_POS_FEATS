import torch.nn as nn

class BiLSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, vocabSize, n_layers= 1):
        super(BiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embeddings = nn.Embedding(vocabSize, input_dim)


        # 2. LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, num_layers=1)


        # 4. Dense Layer
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        embeddings = self.embeddings(x)
        lstm_output, (hidden_states, cell_states) = self.lstm(embeddings)  # Fix the unpacking here

        logits = self.fc(self.relu(lstm_output))
        return logits, (hidden_states, cell_states)




