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