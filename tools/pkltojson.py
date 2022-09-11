import pandas as pd
import os
from glob import glob
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

np.set_printoptions(precision=6, suppress=True)

## [x,y] * total_frame
columns = ['frame_dir', 'label', 'img_shape', 'original_shape', 'total_frames', 'keypoint', 'keypoint_score']
cat_list = []
dog_list = []

action_dict = {
    'dog': ['sit', 'bodylower', 'taillow', 'turn', 'lying', 'heading', 'walkrun', 'tailing', 'feetup', 'footup',
            'bodyscratch', 'bodyshake', 'mounting'],
    'cat': ['tailing', 'laydown', 'grooming', 'footpush', 'walkrun', 'sitdown', 'armstretch', 'roll', 'getdown',
            'lying', 'heading', 'arch']
}

cat_data_limit_num = {x: y for x, y in zip(action_dict['cat'], [5] * len(action_dict['cat']))}
dog_data_limit_num = {x: y for x, y in zip(action_dict['dog'], [5] * len(action_dict['dog']))}

cat_img_shape = []
dog_img_shape = []

for species in ['CAT', 'DOG']:
    with open(os.path.join('../model', 'annotation', species, 'config.yaml')) as f:
        shape_annotation = yaml.load(f, Loader=yaml.FullLoader)

    for vid_dirs in shape_annotation['video_sets']:
        shape = list(map(int, shape_annotation['video_sets'][vid_dirs]['crop'].split(',')))
        if species == 'CAT':
            cat_img_shape.append((shape[1], shape[3]))
        else:
            dog_img_shape.append((shape[1], shape[3]))

cat_img_shape = iter(cat_img_shape)
dog_img_shape = iter(dog_img_shape)

data_folder_dir = '../data/annotated_data'
data_folder_list = os.listdir(data_folder_dir)

for idx, folder_name in tqdm(enumerate(data_folder_list)):
    species, pose, vid_name = folder_name.split('-')
    if (species == 'cat' and cat_data_limit_num[pose] > 0) or (species == 'dog' and dog_data_limit_num[pose] > 0):
        anno_data = pd.read_hdf(glob(os.path.join(data_folder_dir, folder_name) + '/*.h5')[0])
        keypoint = []
        keypoint_score = []
        for data in anno_data.iloc:
            kpt = []
            score = []
            for i in range(len(data) // 3):
                kpt.append([round(data[3 * i], 1), round(data[3 * i + 1], 1)])
                score.append(round(float(data[3 * i + 2]), 4))

            keypoint.append(kpt)
            keypoint_score.append(score)

        if species == 'cat':
            shape_data = next(cat_img_shape)
            cat_data_limit_num[pose] -= 1
        else:
            shape_data = next(dog_img_shape)
            dog_data_limit_num[pose] -= 1

        frame_data_json = {
            'frame_dir': folder_name,
            'label': action_dict[species].index(pose),
            'img_shape': shape_data,
            'original_shape': shape_data,
            'total_frames': len(anno_data),
            'keypoint': np.array([keypoint]),
            'keypoint_score': np.array([keypoint_score])
        }

        # print(frame_data_json['keypoint'])
        # print(frame_data_json['keypoint_score'])

        if species == 'cat':
            cat_list.append(frame_data_json)
        else:
            dog_list.append(frame_data_json)


print(len(cat_list))
print(len(dog_list))
cat_train, cat_val = train_test_split(cat_list, test_size=0.2)
dog_train, dog_val = train_test_split(dog_list, test_size=0.2)

for _name, _data in zip(['cat_train_balance', 'cat_val_balance', 'dog_train_balance', 'dog_val_balance'],
                        [cat_train, cat_val, dog_train, dog_val]):
    with open('sample_data/' + _name + '.pkl', 'wb') as f:
        pickle.dump(_data, f)

# for _name, _data in zip(['cat_train', 'cat_val', 'dog_train', 'dog_val'], [cat_train, cat_val, dog_train, dog_val]):
#     with open('sample_data/' + _name + '.pkl', 'wb') as f:
#         pickle.dump(_data, f)
