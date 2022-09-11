import os
from collections import defaultdict

data_dir = '../data/annotated_data'
species = ['cat','dog']
action_dict = {
    'dog': ['sit', 'bodylower', 'taillow', 'turn', 'lying', 'heading', 'walkrun', 'tailing', 'feetup', 'footup',
            'bodyscratch', 'bodyshake', 'mounting'],
    'cat': ['tailing', 'laydown', 'grooming', 'footpush', 'walkrun', 'sitdown', 'armstretch', 'roll', 'getdown',
            'lying', 'heading', 'arch']
}

data_list = os.listdir(data_dir)
for sp in species:
    label_count = defaultdict()
    label_prob = {}
    sp_data = [x for x in data_list if x.startswith(sp)]
    total_count = len(sp_data)
    for vid in sp_data:
        pose = vid.split('-')[1]
        if pose in label_count:
            label_count[pose] += 1
        else:
            label_count[pose] = 1

    for idx,action in enumerate(action_dict[sp]):
        label_prob[idx] = (round(label_count[action] / total_count,5))


    print(label_prob)

