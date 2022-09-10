dataset_info = dict(
    dataset_name='aihub_animalpose',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),

    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='front_head_bone',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='angle_of_mouth',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='mandible_symphysis_body',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        4:
        dict(
            name='thoracic_vertebrae',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        5:
        dict(
            name='right_point_of_elbow',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='left_point_of_elbow'),
        6:
        dict(
            name='left_point_of_elbow',
            id=6,
            color=[255, 128, 0],
            type='lower',
            swap='right_point_of_elbow'),
        7:
        dict(
            name='right_carpus',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='left_carpus'),
        8:
        dict(
            name='left_carpus',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='right_carpus'),
        9:
        dict(
            name='right_patella',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='left_patella'),
        10:
        dict(
            name='left_patella',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='right_patella'),
        11:
        dict(
            name='right_calcaneus',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='left_calcaneus'),
        12:
        dict(
            name='left_calcaneus',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='right_calcaneus'),
        13:
        dict(
            name='lumbosacral_junction',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        14:
        dict(
            name='tail_end',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap=''),
        # 15:
        # dict(
        #     name='left_ankle',
        #     id=15,
        #     color=[0, 255, 0],
        #     type='lower',
        #     swap='right_ankle'),
        # 16:
        # dict(
        #     name='right_ankle',
        #     id=16,
        #     color=[255, 128, 0],
        #     type='lower',
        #     swap='left_ankle')
    },
    skeleton_info={
        0:
        dict(link=('nose', 'front_head_bone'), id=0, color=[51, 153, 255]),
        1:
        dict(link=('nose', 'angle_of_mouth'), id=1, color=[51, 153, 255]),
        2:
        dict(link=('nose', 'mandible_symphysis_body'), id=2, color=[51, 153, 255]),
        3:
        dict(link=('front_head_bone', 'angle_of_mouth'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('front_head_bone', 'mandible_symphysis_body'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('angle_of_mouth', 'mandible_symphysis_body'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('thoracic_vertebrae', 'front_head_bone'), id=6, color=[0, 255, 0]),
        7:
        dict(
            link=('thoracic_vertebrae', 'lumbosacral_junction'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('thoracic_vertebrae', 'right_point_of_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('thoracic_vertebrae', 'left_point_of_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('right_point_of_elbow', 'left_point_of_elbow'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_point_of_elbow', 'right_carpus'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_point_of_elbow', 'left_carpus'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('lumbosacral_junction', 'right_patella'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('lumbosacral_junction', 'left_patella'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('right_patella', 'left_patella'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_patella', 'right_calcaneus'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_patella', 'left_calcaneus'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('lumbosacral_junction', 'tail_end'), id=18, color=[51, 153, 255])
    },
    joint_weights=[1.]*15,
    # animalpose dataset을 기반으로 편집함.
    sigmas=[0.035,0.025,0.025,0.025,0.10,
            0.107,0.107,0.089,0.089,0.107,
            0.107,0.089,0.089, 0.1, 0.1])


#