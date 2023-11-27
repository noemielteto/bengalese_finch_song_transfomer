from importlib import reload
import torch
import transformer
reload(transformer)
from transformer import RhythmicSongformer, Songformer

from torch import nn, optim


bird = 'wh09pk88'

data = torch.load(f'data/{bird}_torch.pt')

N = len(data.keys())

train_idx = torch.randperm(N)[:int(N*0.6)]
test_idx = torch.randperm(N)[int(N*0.6):]

all_songs_tokens_flattened = torch.cat([data[i][0] for i in range(len(data))])
n_tokens = all_songs_tokens_flattened.argmax() + 1  # +1 since indexing is from 0
max_length = 25

model = Songformer(ntokens=n_tokens, n_heads=8, num_layers=2, context_length=max_length)

optimizer = optim.Adam(model.parameters(), lr=0.0003)

n_iters = 250
batch_size = 50
losses = []
for i in range(n_iters):
    idx = torch.randint(len(train_idx), (batch_size, ))
    songs = [data[song_idx.item()] for song_idx in idx]

    # song[0] is the token tensor
    # TODO 1: clip max length when not possible
    # TODO2: sample from different parts of the song

    # segment_start = torch.randint(low=0, high=len(), size=(1,)).item()

    tokens = torch.stack([song[0][:max_length] for song in songs]).permute(1, 0)  # in this case, permute transposes the axes
    # make this consistent with above
    next_tokens = torch.stack([song[0][max_length] for song in songs])#.unsqueeze(-1)
    durations = torch.stack([song[1][:max_length] for song in songs]).permute(1, 0)

    loss = model.train(tokens, next_tokens) # cross-entropy loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
    losses.append(loss.item())

    # todo look at pred probs
    def predict(self, z):
        h = self.out(z)
        #logits = h#torch.softmax(h, dim=-1)
        return h

len(data.keys())
