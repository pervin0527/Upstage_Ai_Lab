[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FVjNDCrt)
# Document Type ImageClassification 11조

## Team

| ![기현우](https://github.com/user-attachments/assets/446f86a6-a08a-4d60-a846-4e470b031ad1)| ![조수한](https://github.com/user-attachments/assets/e1160f18-4441-4156-bd58-f5be8e076782)| ![김민준](https://github.com/user-attachments/assets/3a6a96ca-3d4e-4669-b5c9-19a3cca18a7a)| ![김홍석](https://github.com/user-attachments/assets/b0a17b67-255d-4eec-9303-b39673ef2352)| 
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [기현우](https://github.com/UpstageAILab)             |            [조수한](https://github.com/UpstageAILab)             |            [김민준](https://github.com/UpstageAILab)             |            [김홍석](https://github.com/UpstageAILab)             |            
|                            EDA, Modeling                            |                            EDA, Modeling                            |                            EDA, Modeling, Data Generate, Calibration                          |                            EDA, Modeling, Calibration

## 0. Overview
### Environment
- Upstage에서 제공해주신 서버를 사용했습니다.

    - OS : Ubuntu
    - CPU : AMD Ryzen Threadripper 3970X 32-Core Processor
    - Memory : 252GB
    - GPU : RTX 3090

### Requirements

    pip install -r requirements.txt

## 1. Competiton Info

### Overview

- 본 대회는 Upstage AI Lab 3기에서 수행한 문서 이미지 분류 대회입니다.
- 총 17개의 클래스가 존재.
- train set은 1570개의 이미지로 구성.
- test set은 3140개의 이미지로 구성.

### Timeline

- Start Date : 2024.07.30
- End Date : 2024.08.11

## 2. Components

### Directory

```
.
├── README.md
├── best_config.yaml
├── calib_test.py
├── config.yaml
├── data
│   ├── Augmentation
│   │   ├── augraphy
│   │   │   └── augraphy.ipynb
│   │   ├── augraphy2
│   │   │   └── augraphy.ipynb
│   │   ├── mask_deleted_augraphy
│   │   │   └── augraphy.ipynb
│   │   ├── test.ipynb
│   │   └── title_augraphy
│   │       └── title_augraphy.ipynb
│   ├── augmentation.py
│   ├── augmentation_best_score.py
│   ├── dataset.py
│   ├── del_black_mask.ipynb
│   ├── generate_data.py
│   ├── reformat.py
│   └── save_per_class.py
├── loss.py
├── model maker
│   ├── baseline_code.ipynb
│   ├── v1.ipynb
│   ├── v10.ipynb
│   ├── v11.ipynb
│   ├── v12.ipynb
│   ├── v13.ipynb
│   ├── v14.ipynb
│   ├── v2.ipynb
│   ├── v3.ipynb
│   ├── v4.ipynb
│   ├── v5.ipynb
│   ├── v6.ipynb
│   ├── v7.ipynb
│   ├── v8.ipynb
│   └── v9.ipynb
├── model.py
├── notebook
│   ├── Augmentation.ipynb
│   ├── Augraphy.ipynb
│   ├── CRAFT.ipynb
│   ├── Custom_augmentation.ipynb
│   ├── Custom_generate.ipynb
│   ├── Grad-CAM.ipynb
│   ├── Ocr-test.ipynb
│   ├── Skew_correction.ipynb
│   ├── SwinIR-Test-Image-Denoising.ipynb
│   ├── Test-EDA.ipynb
│   ├── Train-EDA.ipynb
│   ├── compare.ipynb
│   └── lr_scheduler.ipynb
├── requirements.txt
├── test.py
├── test_ocr.py
├── train.py
├── transformer_config.yaml
├── transformer_test.py
├── transformer_train.py
├── tta.py
└── utils
    ├── config_util.py
    ├── data_util.py
    ├── plot_util.py
    ├── test_util.py
    └── train_util.py
```

### 3. Data descrption

### EDA

![image](./imgs/docs/train_class_dist.png)

train data의 클래스별로 이미지 수를 파악해보면 이력서, 소견서, 임신 의료비 지급 신청서를 제외한 나머지 클래스는 100개의 데이터로 균등한 양상을 보입니다.


### Data Processing

![image](./imgs/docs/data_sample.png)

train 데이터는 노이즈나 변형이 없는 클린 데이터고, test 데이터는 노이즈, 변형이 적용된 데이터들입니다.


따라서 test 데이터에 적용된 augmentation이 어떻게 구성되어 있는지 확인하고, train data에 적용하여 학습하는 방식을 채택했고 이후에는 test 데이터의 노이즈를 제거하는 방식을 추가로 도입했습니다.

![image](./imgs/docs/data_generate.png)

또한 OCR 기술을 활용하던 중 추출된 글자를 활용하여 데이터를 생성할 수 있다고 판단해 위와 같은 커스텀 이미지를 생성해 학습에 적용했습니다.

## 4. Modeling

### Model descrition

CNN 계열로 ResNet, EfficientNet 그리고 transformer 계열로 ViT, Swin-transformer를 베이스라인 모델로 실험해봤습니다.

특이한 점은 transformer 계열 모델들이 Attention 매커니즘으로 자주 틀리는 유형들에 더 강건할 것이라 생각했지만 오히려 전반적인 성능이 2~3% 더 낮은 결과를 보였습니다.

또한, 베이스라인으로 실험했던 CNN 계열 모델들은 정확도가 빠르게 포화되는 문제가 발생하여 EfficientNetV2로 Scale Up했고 그 결과 성능이 더 좋아졌습니다.

### Modeling Process

자주 틀리는 케이스들은 [입퇴원 확인서, 진단서, 외래 진료 증명서, 소견서]와 같이 서류 형식의 데이터틀이었으며 입퇴원 확인서를 외래 진료 증명서로 혹은 외래 진료 증명서를 입퇴원 확인서로 예측하는 일관적인 경향이 있습니다.

모델의 성능이 높아져도 틀리는 개수가 줄어들 뿐 근본적으로 이러한 양상은 제거되지 않았습니다.

따라서 위 클래스들만 학습하는 Calibration 모델을 추가했었으나 이 모델 역시 문제를 해결하지 못하여 기각했습니다.

## 5. Result

### Leader Board

![image](./imgs/docs/LB.png)

마지막 날에는 TTA에 올인하여 성능을 조금이라도 높히려고 노력했고 1~2%정도 점수를 향상시킬 수 있었습니다.

### Presentation

- [발표 자료](./[패스트캠퍼스]%20Upstage%20AI%20Lab%203기_CV%20경진대회_발표자료_7조.pptx)

## etc

### Reference

CRAFT - [https://github.com/clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)

TrOCR - [https://huggingface.co/docs/transformers/model_doc/trocr](https://huggingface.co/docs/transformers/model_doc/trocr)

SwinIR - [https://github.com/JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
