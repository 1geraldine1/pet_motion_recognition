# Description

동물의 영상을 입력받아 해당 동물이 어떤 동작을 수행하는지 출력하는 프로젝트입니다.

자세한 소개는 [이 PowerPoint 문서](https://docs.google.com/presentation/d/1Fb6dci7JGdTulolco9j_GbpclSZF51zDmwvvkpEzXoU/edit?usp=sharing)를 통해 확인하실수 있습니다.

# Prepare environment

1. 가상환경 생성
```
conda create -n {가상환경 } -y
```

2. Pytorch 설치(본인 CUDA Toolkit 버전 맞춰서. 예시는 11.3 기준)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

<details>
<summary>deprecated</summary>

<!-- summary 아래 한칸 공백 두어야함 -->
## 이전 버전에서의 mm라이브러리 설치 과정

3. mmcv-full 설치
```
pip install -U openmim
mim install mmcv-full
```

4. mmdetection 설치
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..
```


5. mmpose 설치
```
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
cd ..
```


6. mmaction2 설치
```
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .
cd ..
```
</details>

3. mmaction 관련 라이브러리 설치 (mmengine, mmcv, mmdetection, mmpose)

```
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose
```

4. mmaction2 설치

```
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```
  

5. moviepy 설치 
```
pip install moviepy
```

6. sklearn 설치 (훈련 데이터 생성용)
```
pip install sklearn
```

7. pytables 설치 (훈련 데이터 생성용)
```
conda install pytables
```

# Demo

```
python tools/mmaction_terminal.py ${비디오 파일 경로} ${출력 비디오 이름}
\ [--config ${행동 인식 모델 Config 경로}] [--checkpoint ${행동 인식 모델 체크포인트 경로}] 
\ [--det-config ${detection 모델 경로}] [--det-checkpoint ${detection 모델 체크포인트 경로}] 
\ [--pose-config ${포즈 추정 모델 경로}] [--pose-checkpoint ${포즈 추정 모델 체크포인트 경로}] 
\ [--label-map ${행동 label 파일 경로}]
```
## Optional arguments:

* config : mmaction2 라이브러리를 활용해 훈련시킨 골격 기반 행동 인식 모델의 config 경로를 넣습니다.
* checkpoint : 훈련된 골격 기반 행동 인식 모델의 checkpoint 경로를 넣습니다.
* det-config : mmdetection 라이브러리를 활용해 훈련시킨 Detection 모델의 config 경로를 넣습니다.
* det-checkpoint : 훈련된 Detection 모델의 checkpoint 경로를 넣습니다.
* pose-config : mmpose 라이브러리를 활용해 훈련시킨 포즈 추정 모델의 config 경로를 넣습니다.
* pose-checkpoint : 훈련된 포즈 추정 모델의 checkpoint 경로를 넣습니다.
* label-map : 골격 기반 행동 인식 모델의 label을 기록한 txt파일의 경로를 넣습니다.

## Examples

시연 영상은 다음과 같은 명령어를 통해 생성할수 있습니다.

```
python tools/mmaction_terminal.py data/sample_vid/Sample.mp4 vis_result.mp4
```


# Used-Data

이번 프로젝트에서 사용한 공공데이터는 다음과 같습니다.

* [AI-Hub 반려동물 구분을 위한 동물 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=59)

# Development Environment

이번 프로젝트는 다음과 같은 환경에서 개발되었습니다.

## Hardware 

* OS : Windows 10
* CPU : AMD Ryzen 7 2700X
* RAM : 16GB
* GPU : NVidia GeForce RTX 2060 (6GB)

## Software

* Used-Framework : Pytorch 
* CUDA Toolkit Version : 11.3
* Used-Language : Python 3.9
* DevTool : Pycharm, Visual Studio Code

# Demonstration Video

이미지 클릭시 시연 영상이 실행됩니다.
<br><br>
[![시연 영상](http://img.youtube.com/vi/FJJz9_eXonQ/0.jpg)](https://youtu.be/FJJz9_eXonQ)
