## 1.tensorflow에서 이미지 모델을 로드.
## 2.이미지 데이터를 입력하고 예측 결과값을 출력.

import keras
import tensorflow as tf

def load_model():
    model = keras.applications.MobileNetV2(weights="imagenet")
    print("Model Loaded.")

    return model

model = load_model()