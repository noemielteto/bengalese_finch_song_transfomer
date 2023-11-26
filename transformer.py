import torch
import argparse
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Songformer(nn.Module):
    def __init__(self, ntokens, latent_dims = 512, n_heads = 8, num_layers=3, context_length=100):
        super(Songformer, self).__init__()
        self.ntokens=ntokens
        self.encoder_layer = TransformerEncoderLayer(d_model=latent_dims, nhead=n_heads)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.embedding = nn.Embedding(ntokens, latent_dims)

        self.positional_encoding = PositionalEncoding(d_model=latent_dims, dropout=0.1, max_len=context_length)

        self.out = nn.Sequential(nn.Linear(latent_dims, ntokens))
        

    def predict(self, z):
        h = self.out(z)
        logits = h#torch.softmax(h, dim=-1)
        return logits


    def forward(self, x):

        x = self.embedding(x)
        x = self.positional_encoding(x)

        src_mask = nn.Transformer.generate_square_subsequent_mask(len(x))#.to(device)

        
        z = self.transformer_encoder(x, src_mask)
        return z
    
    def train(self, x, y):
        z = self(x)
        y_hat_logits = self.predict(z[-1, :, :])
        loss_func = nn.CrossEntropyLoss()
        return loss_func(y_hat_logits, y)



class RhythmicSongformer(Songformer):
    def __init__(self, ntokens, latent_dims = 512, n_heads = 8, num_layers=3, context_length=100):
        super(RhythmicSongformer, self).__init__(ntokens, latent_dims, n_heads, num_layers, context_length)

        
        self.embedding = nn.Linear(ntokens+1, latent_dims)
        self.out_duration = nn.Sequential(nn.Linear(latent_dims, 1))

    def predict(self, z):
        logits_token = self.out(z)
        pred_duration = self.out_duration(z)
        return logits_token, pred_duration

    
    def train(self, x, y):
        z = self(x)
        y_tokens = y[:, :, :-1]
        y_duration = y[:, :, -1]

        logits_token, pred_duration = self.predict(z)
        loss_func = nn.CrossEntropyLoss()
        loss_mse = nn.MSELoss()
        loss_token = loss_func(logits_token, y_tokens)
        loss_duration = loss_mse(pred_duration, y_duration)
        return loss_token + loss_duration
    

# model = Birdformer()


# bs = 10
# l = 50

# x = torch.randint(model.ntokens, (l, bs))

# model(x).shape



    #def train_model(self, memory, tgt):



        # memory = torch.rand(10, 32, 512)
        # tgt = torch.rand(20, 32, 512)
        # out = transformer_decoder(tgt, memory)