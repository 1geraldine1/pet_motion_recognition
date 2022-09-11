import os.path as osp
import os
import pickle
from tqdm import tqdm

result = []
path = 'sample_data/vid_anno/'
species = ['cat','dog']

for sp in species:
    sp_list = [x for x in os.listdir(path) if x.startswith(sp)]
    num_sp = len(sp_list)
    train_num = int(num_sp * 0.8)
    test_num = num_sp - train_num

    for d in tqdm(sp_list[:train_num]):
        with open(osp.join(path, d), 'rb') as f:
            content = pickle.load(f)
        result.append(content)
    with open(sp+'_train.pkl', 'wb') as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)

    print(sp+'_train.pkl created')

    result = []
    for d in tqdm(sp_list[train_num:]):
        with open(osp.join(path, d), 'rb') as f:
            content = pickle.load(f)
        result.append(content)
    with open(sp+'_val.pkl', 'wb') as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)