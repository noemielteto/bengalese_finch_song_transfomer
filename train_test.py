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
#all_songs_tokens_flattened.argmax()
n_tokens = all_songs_tokens_flattened.max() + 1  # +1 since indexing is from 0
max_length = 50

model = Songformer(ntokens=n_tokens, n_heads=50, num_layers=3, context_length=max_length, latent_dims=100)


optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_iters = 1000
batch_size = 50
losses = []
for i in range(n_iters):
    idx_ = torch.randint(len(train_idx), (batch_size, ))
    idx = train_idx[idx_]
    songs = [data[song_idx.item()] for song_idx in idx]

    # song[0] is the token tensor; song[1] is the duration tensor

    min_song_length = min([len(song[0]) for song in songs])
    max_length_in_iter = min(max_length, min_song_length) - 1  # minus one because the sequence has to contain one more element to be predicted
    max_length_in_iter = torch.randint(1, max_length_in_iter, (1, )).item()
    tokens      = []
    next_tokens = []
    durations   = []
    for song in songs:
        start_idx = torch.randint(low=0, high=len(song[0])-max_length_in_iter, size=(1,)).item()
        next_idx   = start_idx + max_length_in_iter
        tokens.append(song[0][start_idx:next_idx])
        next_tokens.append(song[0][next_idx])
        durations.append(song[1][start_idx:next_idx])

    tokens = torch.stack(tokens).permute(1, 0)  # in this case, permute transposes the axes
    next_tokens = torch.stack(next_tokens)
    durations = torch.stack(durations).permute(1, 0)


    loss = model.train(tokens, next_tokens) # cross-entropy loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
    losses.append(loss.item())


context_window = 3
idx = torch.randint(len(train_idx), (10, ))
eval_songs = [data[song_idx.item()][0][:context_window] for song_idx in idx]
context = torch.stack(eval_songs).permute(1, 0)


synthetic_songs = model.generate_autoregressive(context, n_steps=15, deterministic=False)
mappings = torch.load(f'data/{bird}_syllable_mapping.pt')
synthetic_songs[:, 0]
def tensor2song(x):
    songs = []
    for i in range(x.size(1)):
        arr = x[:, i]
        idx_list = arr.tolist()
        syll_list = [mappings['idx2syl'][idx] for idx in idx_list]
        songs.append(syll_list)
    return songs

song_list = tensor2song(synthetic_songs)

for song in song_list:
    print(song)



eval_songs = [data[song_idx.item()][0][:context_window+15] for song_idx in idx]
context = torch.stack(eval_songs).permute(1, 0)
context
song_list = tensor2song(context)

for song in song_list:
    print(song)





def evaluate(context_window=25, batch_size=250):
    
    idx_ = torch.randint(len(test_idx), (batch_size, ))
    idx = test_idx[idx_]
    eval_contexts = [data[song_idx.item()][0][:context_window] for song_idx in idx]
    eval_targets = [data[song_idx.item()][0][context_window] for song_idx in idx]
    context = torch.stack(eval_contexts).permute(1, 0)
    tgt = torch.stack(eval_targets)#.permute(1, 0)
    embedding = model(context)
    preds = model.predict_next(embedding[-1, :])
    diff = preds - tgt
    idx = diff == 0
    acc = idx.sum()/len(idx)
    return acc.item()

for i in range(1, 18):

    accuracy = evaluate(context_window=i)
    print('context size: ', i, ' Accuracy: ', accuracy*100, '%')


# Evaluation
