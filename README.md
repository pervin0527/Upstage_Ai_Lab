# Upstage_Ai_Lab

## 1.아나콘다 가상환경 만들기.

    conda create --name AiLab python=3.8
    conda activate AiLab

    ## in env
    pip install pexpect
    pip install psutil

## 2.Crawling

    ## 셀레니움, 웹드라이버 매니저 설치.
    pip install selenium
    pip install webdriver-manager

    ## 브라우저 버전이 호환되지 않는 경우, 라이브러리를 업데이트 설치
    pip install selenium --upgrade
    pip install webdriver-manager --upgrade

    pip install pandas

## 3.FastAPI

    ## 가상환경 만들기. 아나콘다 설치했으면 필요 없음.
    python -m venv .venv

    ## FastAPI, uvicorn 설치
    pip install fastapi
    pip install "uvicorn[standard]"

    ## tensorflow 설치
    pip install tensorflow

    pip install python-multipart

## 4.EDA

- [https://www.kaggle.com/competitions/instacart-market-basket-analysis/data](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)
- [https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2)
- [https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b](https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b)

## 5.Coding Test

## 6.MachineLearning Basic

## 7.DeepLearning

## 8.ML Advanced