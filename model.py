import torch
import torch.nn as nn
import torch.nn.functional as F

# Input size: (2,4)
# Embedding(10,3)
# Output size (2,4,3)

# Inputs GRU
# Input (seq_len, batch, input_size) -> (4, 2, 3)
# Initial hidden state (num_layers * num_directions, batch, hidden_size) -> (1, 2, 3)
# GRU (3, 3)
# Output GRU:
# Output (seq_len, batch, num_directions*hidden_size) -> (4, 2, 3)
# h_n (num_layers * num_directions, batch, hidden_size) -> (1, 2, 3)
# Output: Contains output features from last layer of the GRU for each t
# h_n: hidden state for t=seq_len


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, self.hidden_size)  # num_embedding, embedding_dim
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)  # input_size, hidden_size

    def forward(self, input, hidden):
        embedded = self.embedding(input).view()
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax()
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1,1, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding
        embedded = self.dropout(embedded)

        attn_weights =