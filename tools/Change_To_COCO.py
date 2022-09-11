import os
from tqdm import tqdm
import json
import shutil
import pickle

# 원본 데이터셋 디렉터리 및 변환 결과물 저장 디렉터리
dataset_path = 'D:\\AnimalPoseData\\AnimalPoseData'
output_path = '../data/aihub_animalpose/'

image_id = 0



def find_path(path):
    path_list = [os.path.join(path, x) for x in os.listdir(path)]
    return path_list


def label_data_process(label_data_path):
    global image_id
    for label_data_path_2 in tqdm(find_path(label_data_path + '\\' + pose)):
        # images
        json_output = {}
        images = []
        annotations = []
        categories = []

        with open(label_data_path_2, 'r', encoding='utf8') as f:
            json_object = json.load(f)

            # image width, height
            mp4_name = '_'.join(json_object['file_video'].split('/'))
            mp4_code = mp4_name.split('-')[-1].split('.')[0]
            height = json_object['metadata']['height']
            width = json_object['metadata']['width']

            for data in json_object['annotations']:
                img_file_path = label_data_path_2.replace('.json', '').replace('[라벨]', '[원천]')
                img_file_name = 'frame_' + str(data['frame_number']) + '_timestamp_' + str(
                    data['timestamp']) + '.jpg'
                img_real_path = os.path.join(img_file_path, img_file_name)

                if not (os.path.isfile(img_real_path) or os.path.isfile(img_real_path.replace('.mp4', ''))):
                    # print('file not exist : ',os.path.join(img_file_path,img_file_name))
                    continue

                # image 데이터 변환
                image_data = {}
                file_name = pose + '_' + mp4_code + '_' + str(data['frame_number']) + '_' + str(
                    data['timestamp']) + '.jpg'

                image_data['file_name'] = file_name
                image_data['height'] = height
                image_data['width'] = width
                image_data['id'] = int(image_id)

                images.append(image_data)

                # annotation 변환
                image_annotations = {}
                segmentation = []
                keypoints = []
                # keypoint의 경우, COCO는 x,y,v를 이어서 작성하는 식으로 제작된다.
                # v는 label되지 않을경우 0 (이때, x와 y 모두 0으로 표기), label되었으나 보이지 않을시 1, label되고 보이면 2로 작성한다.
                # 데이터셋에서는 안보이는 관절은 레이블링 하지 않았으므로, v값은 None일때 0, 값이 존재할때 2로 표기한다.
                num_keypoints = 0
                for i in range(1, len(data['keypoints']) + 1):
                    if data['keypoints'][str(i)]:
                        x = data['keypoints'][str(i)]['x']
                        y = data['keypoints'][str(i)]['y']
                        v = 2
                        num_keypoints += 1
                    else:
                        x = 0
                        y = 0
                        v = 0

                    keypoints += [x, y, v]

                area = data['bounding_box']['width'] * data['bounding_box']['height']

                # 여러 개체 인식 여부에 대한 옵션. 아닐땐 0, 맞을땐 1
                iscrowd = 0

                # 바운딩 박스. x,y,width,
                bbox = [data['bounding_box']['x'], data['bounding_box']['y'], data['bounding_box']['width'],
                        data['bounding_box']['height']]

                image_annotations['segmentation'] = segmentation
                image_annotations['keypoints'] = keypoints
                image_annotations['num_keypoints'] = num_keypoints
                image_annotations['area'] = area
                image_annotations['iscrowd'] = iscrowd
                image_annotations['image_id'] = image_id
                image_annotations['bbox'] = bbox
                image_annotations['category_id'] = category_id
                image_annotations['id'] = int(image_id)

                annotations.append(image_annotations)

                image_id += 1

    category = {'id': category_id, 'name': category_name}
    categories.append(category)

    # json_output['images'] = images
    # json_output['det_pose_annotation'] = det_pose_annotation
    # json_output['categories'] = categories

    # if not os.path.isdir(output_path + '\\' + tv + '\\' + category_name + '\\' + pose):
    #     os.makedirs(output_path + '\\' + tv + '\\' + category_name + '\\' + pose)
    # with open(output_path + '\\' + tv + '\\' + category_name + '\\' + pose + '\\' + file_name + '.json', 'w') as f:
    #     json.dump(json_output,f,indent=2)

    return images, annotations, categories


def img_data_process(label_data_path):
    for label_data_path_2 in tqdm(find_path(label_data_path + '\\' + pose)):
        mp4_code = label_data_path_2.split('\\')[-1].split('-')[-1].split('.')[0]
        for file_path in find_path(label_data_path_2):
            img_name = file_path.split('\\')[-1].split('_')
            frame_num = img_name[1]
            timestamp_num = img_name[3]
            file_name = pose + '_' + mp4_code + '_' + frame_num + '_' + timestamp_num
            new_file_path = os.path.join(output_path, tv, category_name, file_name)
            if not os.path.isdir(os.path.join(output_path, tv, category_name)):
                os.makedirs(os.path.join(output_path, tv, category_name), exist_ok=True)
            if not os.path.isfile(new_file_path):
                os.link(file_path, new_file_path)


# Customize datasets by reorganizing data to COCO format

train_path = find_path(dataset_path)[0]
validation_path = find_path(dataset_path)[1]

for train_valid_path in find_path(dataset_path):
    # images
    cat_json_output = {}
    cat_images_output = []
    cat_annotations_output = []
    cat_categories_output = []

    dog_json_output = {}
    dog_images_output = []
    dog_annotations_output = []
    dog_categories_output = []

    tv = train_valid_path.split('\\')[-1]
    for cat_dog_path in find_path(train_valid_path):
        if cat_dog_path.endswith('CAT'):
            category_id = 0
            category_name = 'cat'
        else:
            category_id = 1
            category_name = 'dog'
        for label_data_path in tqdm(find_path(cat_dog_path)):
            pose = label_data_path.split('\\')[-1].split(']')[-1]
            if '[라벨]' in str(label_data_path):
                images, annotations, categories = label_data_process(label_data_path)
                if cat_dog_path.endswith('CAT'):
                    cat_images_output += images
                    cat_annotations_output += annotations
                    cat_categories_output = categories
                else:
                    dog_images_output += images
                    dog_annotations_output += annotations
                    dog_categories_output = categories

            if '[원천]' in str(label_data_path):
                img_data_process(label_data_path)

    cat_json_output['images'] = cat_images_output
    cat_json_output['det_pose_annotation'] = cat_annotations_output
    cat_json_output['categories'] = cat_categories_output

    dog_json_output['images'] = dog_images_output
    dog_json_output['det_pose_annotation'] = dog_annotations_output
    dog_json_output['categories'] = dog_categories_output

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if not os.path.isdir(output_path + '\\' + 'det_pose_annotation'):
        os.makedirs(output_path + '\\' + 'det_pose_annotation')
    with open(output_path + '\\' + 'det_pose_annotation' + '\\' + 'aihub_animalpose_cat_' + tv + '.json', 'w') as f:
        json.dump(cat_json_output, f, indent=2)
    with open(output_path + '\\' + 'det_pose_annotation' + '\\' + 'aihub_animalpose_cat_' + tv + '.pkl', 'wb') as file:
        pickle.dump(cat_json_output, file)
    with open(output_path + '\\' + 'det_pose_annotation' + '\\' + 'aihub_animalpose_dog_' + tv + '.json', 'w') as f:
        json.dump(dog_json_output, f, indent=2)
    with open(output_path + '\\' + 'det_pose_annotation' + '\\' + 'aihub_animalpose_dog_' + tv + '.pkl', 'wb') as file:
        pickle.dump(dog_json_output, file)

    with open(output_path + '\\' + 'det_pose_annotation' + '\\' + 'aihub_animalpose_animals_' + tv + '.json', 'w') as f:
        json.dump(dict(zip(cat_json_output,dog_json_output)), f, indent=2)

    print('=' * 40)
    print(tv + ' json created.')
    print('=' * 40)

# Create a custom dataset_info config file for the dataset
