import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st
import numpy as np

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data
y = iris.target

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 테스트 데이터로 예측 및 정확도 계산
y_pred = model.predict(X_test)
print(y_pred[:10])
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 모델 저장
with open("iris_model.pkl", "wb") as f: 
    pickle.dump(model, f)


# 모델 로드
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit 앱 제목
st.title("Iris Flower Classification")

# 사용자 입력
st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 0.2)
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# 사용자 입력 값 표시
st.subheader("User Input parameters")
st.write(df)

# 예측
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
st.balloons()

# 클래스 이름
class_names = iris.target_names 

# 예측 결과 표시
st.subheader("Class labels and their corresponding index number")
st.write(class_names)

st.subheader("Prediction")
st.write(class_names[prediction[0]])

st.subheader("Prediction Probability")
st.write(prediction_proba)
