[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zHsKfIy0)

# Title (Please modify the title)
## Team

| ![기현우](https://github.com/user-attachments/assets/446f86a6-a08a-4d60-a846-4e470b031ad1)| ![조수한](https://github.com/user-attachments/assets/e1160f18-4441-4156-bd58-f5be8e076782)| ![김민준](https://github.com/user-attachments/assets/3a6a96ca-3d4e-4669-b5c9-19a3cca18a7a)| ![김홍석](https://github.com/user-attachments/assets/b0a17b67-255d-4eec-9303-b39673ef2352)| 
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [기현우](https://github.com/UpstageAILab)             |            [조수한](https://github.com/UpstageAILab)             |            [김민준](https://github.com/UpstageAILab)             |            [김홍석](https://github.com/UpstageAILab)             |            
|                         EDA, Modeling                            |                            EDA, Modelin, BackTranslation                            |                            EDA, Modeling, Data Generate, BackTranslation                          |                            EDA, Modeling

## 0.Overview
### Environment
- Upstage에서 제공해주신 서버.

    - OS : Ubuntu
    - CPU : AMD Ryzen Threadripper 3970X 32-Core Processor
    - Memory : 252GB
    - GPU : RTX 3090


### Requirements
pip install -r requirements.txt

## 1.Competiton Info

### Overview

본 대회는 여러 명의 화자들이 나눈 대화 내용을 입력으로 받아 요약문을 생성하는 Dialogue Summarization.

### Timeline

- Start Date : 2024.08.29 10:00
- Final submission deadline : 2024.09.10 19:00


## 2.Data descrption

### Dataset overview

![img](./doc_imgs/0001.png)

- train, dev(valid), test 세 가지 csv 파일이 주어진다.
- train, dev set에서는 위 이미지에 나와있듯 fname, dialogue, summary, topic 4가지 컬럼으로 구성되며 test는 fname, dialogue로만 구성된다.
- 목표는 train, dev에 있는 dialogue를 학습시키고, test dialogue를 입력 받아 정답에 가까운 summary를 구해야한다.

### EDA

#### 1.Dialogue, Summary 길이 파악

![img](./doc_imgs/0002.png)

요약 모델을 학습하기 전에 대화문과 요약문의 길이에 대해 파악합니다. 이는 모델 학습시 학습에 반영될 문장의 최대 길이, 생성할 문장의 최대 길이를 결정하는데 중요한 역할을 한다.

![img](./doc_imgs/0003.png)

또한, 대화문과 요약문 길이간 적절한 비율을 파악하기 위해 선형회귀를 수행했을 때 기울기(slope)가 약 0.14임을 알 수 있습니다. 이를 근거로 dialogue 길이의 0.14 정도로 generate max length를 설정할 수도 있을 것 같다.

#### 2.Topic

![img](./doc_imgs/0004.png)

그래프를 그렸을 때 너무 지저분해져 따로 첨부하지는 않았으나, train과 dev를 취합했을 때 topic은 6337개.

많은 수의 topic은 1개의 row로 구성되어 있기 때문에 특별한 영향력을 갖는다고 해석하기 어렵습니다. 따라서 상위 5개 또는 10개의 topic을 선정해서 샘플 데이터를 선별하는데 사용.

선별된 샘플 데이터는 이후에 SOLA API를 이용해 데이터를 생성하는데 사용.


### Data Processing

#### 1.Special Tokens

dialogue에 등장하는 #문자열# 형태의 특수토큰은 다음과 같다.

![img](./doc_imgs/0005.png)

처음엔 이러한 토큰이 필요하지 않은듯 했으나, 데이터 분석 과정에서 summary에 특수토큰이 등장하는 것을 확인하고 요약문 생성시 주어, 목적어로 활용된다는 점을 감안해 제거하지 않았다.

#### 2.반복되는 글자 제거

![img](./doc_imgs/0006.png)

구어체임에도 불구하고 dialogue 컬럼에는 "ㅋㅋ"와 같은 값들이 존재합니다. 극소량만 존재하지만 이왕 발견한거 제거했다.

![img](./doc_imgs/0007.png)

또한 위와 같이 동일한 글자가 연속적으로 등장하는 경우에도 문자열의 길이만 증가시킬 뿐 특별한 의미를 갖지 않는다고 판단하여 최초 한글자만 남겨두고 뒷부분은 제거하는 방식으로 처리했다.

#### 3.한글 이외의 글자

![img](./doc_imgs/0008.png)

한글이 아닌 값이 어떤 것들이 있는지 파악해봤다. 

분석 결과

- 대화 중 화자 A가 말하는 동안 화자 B의 대답이나 제스처를 취하는 경우 ```[문자열]``` 또는 ```(문자열)```로 나타나는 경우가 있었습니다. 이는 별도의 턴(turn)으로 분할하기에 애매하고 요약문에 별다른 영향을 주지못한다고 판단해 #Reaction# 토큰으로 대체.
- 숫자와 알파벳은 의미를 갖는 단어(주문번호, 날짜, 병명 등)를 나타냈기 때문에 별도로 처리하지 않았다.
- 이밖에도 ㅏ, ㅍ, ㄷ와 같은 오타도 존재했는데 문맥에 맞는 글자로 변환하거나 제거하였다.


## 4. Modeling

### 1.Tokenizer, Pretraining

- 대회 데이터에 토크나이저와 pretrained model간 부적합한 부분이 있어 점수에 한계가 있는 것은 아닐까라는 생각에 토크나이저를 새로 구성하고 추가적인 pretraining을 해봤다.
- BART : SentencePiece, T5 : BPE
- Pretrain Task : Span Corruption
- 토크나이저와 pretrained weight가 서로 상응하지 않는 경우 pretrained weight를 사용하기 어렵다는 점을 알게되어 더 이상 시도하지 않았다.

### 2.문제점

- Bart, T5를 기반으로 몇가지 실험을 진행하였지만 리더보드 점수는 최대 40점.
- 한국어로 학습된 pretrained weight는 선택 폭이 너무 좁았다.
- 대회 데이터는 원래 영어로 구성된 데이터였으나 한국어로 번역된 것이기 때문에 형태소 분석시 적합한 결과라고 받아들이기 힘든 경우가 있었다. ex) '스미스' -> ['스', '미스']

### 3.QLoRA

BART, T5 모델들은 성능 향상이 이루어지지 않아서 LLM 모델을 시도. ```LLama3-8B-int4```를 모델을 선택하고, 컴퓨팅 자원이 부족하기 때문에 효율성을 위해 QLoRA를 적용.

학습 결과 43.25로 Bart, T5보다 높은 성능을 보였다.

### 4.Data Augmentation

- SamSum 데이터셋을 한국어로 번역해 추가하고 대회 데이터를 Back translation해서 동의어 데이터 생성하고 학습시 44.6으로 1%개선.
- train(구어체)과 test(문어체)간 어체차이를 통일 시키기 위해 dialogue-to-summary, summary-to-dialogue 데이터 생성하고 학습시 45.72로 2%개선.