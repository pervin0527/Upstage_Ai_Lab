import re
import time
import random
import signal
import pandas as pd

from koeda import EDA, AEDA
from tqdm import tqdm
from contextlib import contextmanager
from koeda import EasyDataAugmentation, AEasierDataAugmentation
from googletrans import Translator


def augment_text_data_with_EDA(text: str, repetition=1):
    eda = EDA(morpheme_analyzer="Okt", alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, prob_rd=0.1)
    
    pattern = r'#.*?#:?'
    
    lines = text.split('\n')
    augmented_texts = []
    
    for line in lines:
        matches = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, line)]
        cleaned_line = line
        for start, end, match in matches:
            cleaned_line = cleaned_line.replace(match, "")
        
        result = eda(cleaned_line)
        
        for start, end, match in matches:
            result = result[:start] + match + result[start:]
        
        augmented_texts.append(result.strip())

    return augmented_texts


def augment_text_data_with_AEDA(text, repetition):
    """입력된 문장에 대해서 AEDA를 통해 데이터 증강"""
    aeda = AEDA(
        morpheme_analyzer="Okt", punc_ratio=0.1, punctuations=[".", ",", "!", "?", ";", ":"]
    )


    pattern = r'#.*?#:?'
    lines = text.split('\n')
    augmented_texts = []
    
    for line in lines:
        matches = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, line)]
        
        cleaned_line = line
        for start, end, match in matches:
            cleaned_line = cleaned_line.replace(match, "")
    
        result = aeda(text)
    
        for start, end, match in matches:
            result = result[:start] + match + result[start:]
        
        augmented_texts.append(result.strip())

    return augmented_texts


def augment_text_data_with_BT(text, repetition):
    """입력된 문장에 대해서 BT를 통해 데이터 증강"""
    translator = Translator()
    result = []

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


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timeout!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def augment_dataframe(df, repetition=1):
    augmented_rows = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Augmenting Data"):
        original_dialogue = row['dialogue']
        summary = row['summary']
        topic = row['topic']

        eda_augmented_texts = augment_text_data_with_EDA(original_dialogue, repetition)
        aeda_augmented_texts = augment_text_data_with_AEDA(original_dialogue, repetition)
        bt_augmented_texts = augment_text_data_with_BT(original_dialogue, repetition)

        print("-" * 10)
        print(f"원본 : {original_dialogue}")
        print(f"EDA : {eda_augmented_texts}")
        print(f"AEDA : {aeda_augmented_texts}")
        print(f"BT : {bt_augmented_texts}")
        print("-" * 10)
        print("\n\n")

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


# def augment_dataframe(df, repetition=1):
#     augmented_rows = []

#     for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Augmenting Data"):
#         original_dialogue = row['dialogue']
#         summary = row['summary']
#         topic = row['topic']

#         # 증강 함수 호출과 타임아웃 설정
#         try:
#             with time_limit(10):
#                 eda_augmented_texts = augment_text_data_with_EDA(original_dialogue, repetition)
#         except TimeoutException:
#             eda_augmented_texts = []

#         try:
#             with time_limit(10):
#                 aeda_augmented_texts = augment_text_data_with_AEDA(original_dialogue, repetition)
#         except TimeoutException:
#             aeda_augmented_texts = []

#         try:
#             with time_limit(10):
#                 bt_augmented_texts = augment_text_data_with_BT(original_dialogue, repetition)
#         except TimeoutException:
#             bt_augmented_texts = []

#         # 증강된 텍스트를 결과에 추가
#         for text in eda_augmented_texts + bt_augmented_texts + aeda_augmented_texts:
#             augmented_rows.append({
#                 'dialogue': text,
#                 'summary': summary,
#                 'topic': topic
#             })

#     augmented_df = pd.DataFrame(augmented_rows)
#     result_df = pd.concat([df, augmented_df], ignore_index=True)

#     return result_df


if __name__ == "__main__":
    df = pd.read_csv('../dataset/cleaned_train.csv')
    small_df = df.sample(n=10)
    # small_df.to_csv("../dataset/sample.csv")
    augment_dataframe(df)