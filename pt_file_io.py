import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def build_raw_events(df, tqdm_comment):
    """Create a list of dictionnaries for each event"""
    events = []
    grouped = df.groupby('event_id')

    for evt_id, group in tqdm(grouped, total=len(grouped), desc=tqdm_comment):
        
        # Hit level features
        position = group[['I', 'J', 'K']].values.astype(np.float32)
        time = group[['time']].values.astype(np.float32)
        thr_values = group['thr'].values.astype(int) - 1  # 0,1,2
        thr_onehot = np.eye(3)[thr_values].astype(np.float32)
        hit_level_features = np.concatenate([position, time, thr_onehot], axis=1)

        # Global features
        nb_hits_event = len(position)
        nb_hits_in_last_layer = np.sum(position[:,2] == 47 )
        if len(thr_values) > 0:
            ratio_thr3 = (thr_values == 2).sum() / len(thr_values)
        else:
            print(f"Event {evt_id} has 0 hit")
            ratio_thr3 = 0.0

        z_position = position[:, 2]
        layers, counts = np.unique(z_position, return_counts=True)
        hit_count = dict(zip(layers, counts))
        layers_sorted = np.sort(layers)
        first_layer_bool = None
        for i, layer in enumerate(layers_sorted):
            next_layers = layers_sorted[i+1:i+4]
            if len(next_layers) < 3:
                continue
            if hit_count[layer] >= 4 and all(hit_count[l] >= 4 for l in next_layers):
                first_interaction_layer_event = layer
                first_layer_bool = True
                break
        if first_layer_bool is None:
#            print(f"No first interaction layer found for event : {evt_id}")
            first_interaction_layer_event = 0  # Default value if not found

        # Global informations
        label = group['label'].iloc[0]
        mc_energy = group['mc_energy'].iloc[0]

        events.append({
                    'hit_level_features': torch.tensor(hit_level_features, dtype=torch.float32),
                    'nb_hits': nb_hits_event,
                    'ratio_thr3': ratio_thr3,
                    'nb_hits_in_last_layer': nb_hits_in_last_layer,
                    'first_interaction_layer': first_interaction_layer_event,
                    'PID_label': label,
                    'mc_energy': mc_energy,
                    'event_id': evt_id
                   })

    return events