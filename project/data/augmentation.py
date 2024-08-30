import re
import time
import random
import pandas as pd

from tqdm import tqdm
from koeda import EasyDataAugmentation, AEasierDataAugmentation
from googletrans import Translator


def timeit(func):
    """이 데코레이터는 함수의 실행 시간을 측정합니다."""
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 시작 시간 측정
        result = func(*args, **kwargs)
        end_time = time.time()    # 종료 시간 측정
        elapsed_time = end_time - start_time
        print(f"'{func.__name__}' 함수 실행 시간: {elapsed_time:.4f}초")
        return result
    return wrapper


# @timeit
def augment_text_data_with_EDA(text: str, repetition=1):
    eda = EasyDataAugmentation(morpheme_analyzer="Mecab")
    
    pattern = r'#.*?#:?'
    
    lines = text.split('\n')
    augmented_texts = []
    
    for line in lines:
        matches = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, line)]
        cleaned_line = line
        for start, end, match in matches:
            cleaned_line = cleaned_line.replace(match, "")
        
        result = eda(cleaned_line, p=(0.1, 0.1, 0.1, 0.1), repetition=repetition)
        
        for start, end, match in matches:
            result = result[:start] + match + result[start:]
        
        augmented_texts.append(result.strip())

    return augmented_texts


# @timeit
def augment_text_data_with_AEDA(text, repetition):
    """입력된 문장에 대해서 AEDA를 통해 데이터 증강"""
    aeda = AEasierDataAugmentation(morpheme_analyzer="Okt", punctuations=[".", ",", "!", "?", ";", ":"])

    pattern = r'#.*?#:?'
    lines = text.split('\n')
    augmented_texts = []
    
    for line in lines:
        matches = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, line)]
        
        cleaned_line = line
        for start, end, match in matches:
            cleaned_line = cleaned_line.replace(match, "")
    
        result = aeda(text, p=0.1, repetition=repetition)
    
        for start, end, match in matches:
            result = result[:start] + match + result[start:]
        
        augmented_texts.append(result.strip())

    return augmented_texts


# @timeit
def augment_text_data_with_BT(text, repetition):
    """입력된 문장에 대해서 BT를 통해 데이터 증강"""
    translator = Translator()
    result = []

    # 패턴 정의: #문자열#: 또는 #문자열# 추출
    pattern = r'#.*?#:?'
    matches = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, text)]
    
    cleaned_text = text
    for start, end, match in matches:
        cleaned_text = cleaned_text.replace(match, "")

    # 번역 실행 (한국어 > 영어 > 한국어)
    for i in range(repetition):
        translated = translator.translate(cleaned_text, src='ko', dest='en')
        re_translated = translator.translate(translated.text, src='en', dest='ko')
        augmented_text = re_translated.text
        
        for start, end, match in matches:
            augmented_text = augmented_text[:start] + match + augmented_text[start:]
        
        result.append(augmented_text)
    
    return result


def augment_dataframe(df, repetition=1):
    augmented_rows = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Augmenting Data"):
        original_dialogue = row['dialogue']
        summary = row['summary']
        topic = row['topic']

        eda_augmented_texts = augment_text_data_with_EDA(original_dialogue, repetition)
        aeda_augmented_texts = augment_text_data_with_AEDA(original_dialogue, repetition)
        bt_augmented_texts = augment_text_data_with_BT(original_dialogue, repetition)

        for text in eda_augmented_texts + bt_augmented_texts + aeda_augmented_texts:
        # for text in eda_augmented_texts + bt_augmented_texts:
            augmented_rows.append({
                'dialogue': text,
                'summary': summary,
                'topic': topic
            })


    augmented_df = pd.DataFrame(augmented_rows)
    result_df = pd.concat([df, augmented_df], ignore_index=True)

    return result_df


if __name__ == "__main__":
    df = pd.read_csv('../dataset/cleaned_train.csv')
    augment_dataframe(df)