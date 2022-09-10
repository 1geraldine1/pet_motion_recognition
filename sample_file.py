import pandas as pd

pkl_cat = pd.read_pickle('./cat_train.pkl')
pkl_sample = pd.read_pickle('data/aihub_animalpose/det_pose_annotation/aihub_animalpose_cat_Training.pkl')

pkl1 = pd.DataFrame(pkl_cat)
pkl2 = pd.DataFrame(pkl_sample)

print(pkl_sample)

# print(pkl1['keypoint_score'][0])
# print(pkl2['keypoint_score'][0])
# #
# for i,j in zip(pkl1.iloc[:2],pkl2.iloc[:2]):
#     for column_name in pkl1.columns:
#         print(type(i[column_name]))
#         print(type(j[column_name]))
#         print('='*10)
