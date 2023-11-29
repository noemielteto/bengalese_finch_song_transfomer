import pandas as pd
import torch
import os
import sys
import numpy as np

def preprocess(arr):
    tensor = torch.zeros(len(arr))
    for k, d in enumerate(arr):

        val = float(d)

        tensor[k] = val

    #nan_idx = torch.isnan(tensor)
    #tensor[nan_idx] = 0.0
    return tensor



def get_duration_and_latency(start, end):
    start, end = preprocess(start), preprocess(end)
    nan_idx_start = torch.isnan(start)
    nan_idx_end = torch.isnan(end)
    duration = torch.zeros(len(start))
    latency = torch.zeros(len(start))
    for i in range(len(start)):
        if nan_idx_end[i] or nan_idx_start[i]:
            continue
        d = end[i] - start[i]
        duration[i] = d
        if i > 0:
            if not nan_idx_end[i-1]:
                l = start[i] - end[i-1]
                latency[i] = l

    return duration, latency





birds = os.listdir('data/timestamped_data')

birds


for i, bird in enumerate(birds):
    
    songs = os.listdir(f'data/timestamped_data/{bird}/baseline')

    data_dict = {}

    # with open(f'data/baseline_data/{bird}.txt', 'r') as file:
    #     all_songs = file.read()

    #all_syls = set(all_songs)
   # syl2idx = {syl:i for i, syl in enumerate(all_syls)}
    all_syls = []
    song_list = []
    duration_list = []
    latency_list = []

    for j, song in enumerate(songs):
        data = pd.read_csv(f'data/timestamped_data/{bird}/baseline/{song}', header=None).values
        syllables = data[0]
        start = data[1]
        end =  data[2]
        all_syls += list(syllables)
        song_list.append(syllables)

        duration, latency = get_duration_and_latency(start, end)
        
        # duration = preprocess(duration)
        # latency = preprocess(latency)

        duration_list.append(duration)
        latency_list.append(latency)

    all_syls = set(all_syls)
    syl2idx = {syl:i for i, syl in enumerate(all_syls)}
    idx2syl = {i:syl for i, syl in enumerate(all_syls)} # inverse mapping

    torch.save({'syl2idx': syl2idx, 'idx2syl':idx2syl}, f'data/{bird}_syllable_mapping.pt')


    for j, song in enumerate(song_list):

        tokens =  torch.tensor([syl2idx[syl] for syl in song])
        data_dict[j] = (tokens, duration_list[j], latency_list[j])

    torch.save(data_dict, f'data/{bird}_torch.pt')




syl_list2 = []
for j, song in enumerate(song_list):
    syl_list2 += song.tolist()

syl_list2

set(syl_list2)
all_syls
len(data_dict)

all_songs_tokens_flattened = torch.cat([data_dict[k][0] for k in data_dict.keys()])
[data_dict[k][0] for k in data_dict.keys()]


all_songs_tokens_flattened.argmax()


song_list
set(np.array(song_list).ravel().tolist())

song_list