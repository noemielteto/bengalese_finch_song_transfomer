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

n_tokens = data[0][0].argmax()
max_length = 5


model = Songformer(ntokens=n_tokens, n_heads=8, num_layers=2, context_length=max_length)

optimizer = optim.Adam(model.parameters(), lr=0.0003)



n_iters = 250
batch_size = 50
losses = []
for i in range(n_iters):
    idx = torch.randint(len(train_idx), (batch_size, ))
    songs = [data[song_idx.item()] for song_idx in idx]

    tokens = torch.stack([song[0][:max_length] for song in songs]).permute(1, 0)
    next_tokens = torch.stack([song[0][max_length] for song in songs])#.unsqueeze(-1)
    durations = torch.stack([song[1][:max_length] for song in songs]).permute(1, 0)

    loss = model.train(tokens, next_tokens)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
    losses.append(loss.item())






len(data.keys())