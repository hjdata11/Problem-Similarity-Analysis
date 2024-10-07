# 문제 유사도 분석을 위한 Problem Similarity Analysis 시스템

## 개요

문제 유사도 분석 모델: BERT(Bidirectional Encoder Representations from Transformers)와 VIT(Vision Transformer)의 초기 모델들을 활용하여, 문제의 텍스트와 이미지 간의 유사도 분석하고, 이를 바탕으로 유사한 문제를 추출해주는 모델
BERT모델에 관한 자세한 내용은 [BERT 논문](https://arxiv.org/pdf/1810.04805)을 참조하세요.
VIT모델에 관한 자세한 내용은 [VIT 논문](https://arxiv.org/pdf/2010.11929)을 참조하세요.

### 주요 기능
- 문제 유사도 분석을 위한 텍스트 및 이미지 Self-supervised learning 학습
- 다중 GPU 지원을 통한 고속 학습
- 3D Embedding Space 시각화 및 유사도 분석 결과 분석 지원

## 사용 방법

### 실행 방법
raw 파일 메타데이터 형태 정보를 dataframe 형식으로 trainset과 testset으로 저장:
```shell
python preprocessing.py
```
Image와 Text를 통합 임베딩하여 문제의 커리큘럼 및 유형(UK)를 분류하는 모델 학습:
(Hugginface의 transformers 모델을 다운받아 modeling bert.py. configuration_bert.py 파일을 overriding하여 VIT 모델 및 BERT 통합 모델 사용)
```shell
python ITBERT_classifier.py
```
Image 및 Text 통합 입베딩 분류 모델 확인 및 문제 Class별 모델 분류 정확도 확인:
```shell
python testset_result.py
```
Image 및 Text 통합 입베딩 벡터 추출 및 메타데이터 추출
(https://projector.tensorflow.org/ 사이트에 3Doutput.tsv, 3Dmetadata.tsv 파일을 넣어 문제 임베딩 상태 확인)
```shell
python similarity_map_3D.py
```
문제의 쿼리 인덱스를 통한 유사 문제의 Image 및 Text 확인
(Euclidean Distance 방식을 통한 유사 문제 추출)
```shell
python similarity_find.py --query_index 10
```

## 모델 설명

### Problem Similarity Analysis 설명
- 와플 수학 문제 스튜디오 9000개의 문제 이미지 유사도 분석
- 총 문제 중 4000개의 문제가 도형 이미지를 포함하고 있으며, 도형 이미지가 있는 경우에만 Image와 Text 통합 임베딩
- 문제는 56개 유형(Class)으로 분류되며, 각각의 유형은 임베딩 스페이스에서 표현
- 도형 이미지의 196개 Token Embedding과 문제 텍스트의 256개 Token Embedding을 겹합하여 CLS Token으로 문제 간 유사도 분석

## 데이터셋

### 데이터셋 구조
데이터셋의 구조는 다음과 같습니다:
```shell
data/
├── problems15k.pkl : 문제 Text, 문제 유형, 문제 커리큘럼, 이미지 경로 등 문제의 raw 메타 정보가 들어있는 파일
├── 256_images/ : 256 * 256 크기 이미지로 resized 된 이미지 jpg 디렉토리
├── PT15k/ : 문제를 trainset 및 testset으로 나누고 문제 text, 유형, 커리큘럼, 이미지 경로 등을 dataframe으로 변환한 메타데이터 디렉토리
│   └── PTTrainTest.pkl
├── PT15ktensor : 문제의 trainset, testset, total, weights 등 문제의 Tensor 정보로 저장된 디렉토리
│   ├──PTtensorTrain.pt
│   ├──PTtensorTest.pt
│   ├──PTtensorTotal.pt
│   └───Possible_Labels.pt
├── checkpoint15k : 체크 포인트 저장 디렉토리
└── imgs_retrieval : 해당 유형안에서 문제 Query와 비슷한 도형의 이미지를 찾아 이미지를 jpg 형태로 저장되는 디렉토리
```

### 학습 데이터셋
|데이터 이름|개수|설명|
|:---:|:---:|:---:|
|WAPL 수학 데이터셋|8000장|수학 텍스트와 도형 이미지가 포함된 Dataset|

### 검증 데이터셋
|데이터 이름|개수|설명|
|:---:|:---:|:---:|
|WAPL 수학 데이터셋|1000장|수학 텍스트와 도형 이미지가 포함된 Dataset 평가|

## 성능

검증 데이터셋에서의 문제 56개 유형(Class) 분류 성능 결과:
 - Accuracy = 91%

## 문제 임베딩 스페이스 결과

![문제_임베딩_스페이스](/uploads/642fc863dda942901645acc4734380db/문제_임베딩_스페이스.jpg)


## Problem Similarity Analysis 결과

- 이미지 유사도 결과 (256 * 256)

![Image_유사도_결과](/uploads/eb86ce276e050ee8f95d6e2224b6d74a/Image_유사도_결과.jpg)


- 텍스트 유사도 결과

![Text_유사도_결과](/uploads/5f98616b786252510b1bc4066e7ebf8b/Text_유사도_결과.jpg)
