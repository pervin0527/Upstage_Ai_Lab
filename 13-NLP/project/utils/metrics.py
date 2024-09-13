import torch
import numpy as np
from rouge import Rouge
from konlpy.tag import Mecab

mecab = Mecab()

def tokenize_text(text):
    # 빈 문자열 처리
    if not text.strip():
        return 'empty'
    return ' '.join(mecab.morphs(text))

def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    # -100 값들을 padding token으로 교체
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # 모델의 토크나이저로 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    # 불필요한 생성토큰들을 제거
    replaced_predictions = decoded_preds.copy()
    replaced_labels = decoded_labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token, " ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token, " ") for sentence in replaced_labels]

    # 빈 문자열 확인 및 토크나이즈
    tokenized_preds = [tokenize_text(sentence) for sentence in replaced_predictions]
    tokenized_labels = [tokenize_text(sentence) for sentence in replaced_labels]

    # 빈 예측이 있는지 확인 및 처리
    if any(pred == 'empty' for pred in tokenized_preds):
        print("Empty prediction detected, skipping this evaluation.")
        return {}

    print('-' * 150)
    print(f"PRED: {tokenized_preds[0]}")
    print(f"GOLD: {tokenized_labels[0]}")
    print('-' * 150)
    print(f"PRED: {tokenized_preds[1]}")
    print(f"GOLD: {tokenized_labels[1]}")
    print('-' * 150)
    print(f"PRED: {tokenized_preds[2]}")
    print(f"GOLD: {tokenized_labels[2]}")

    # 최종적인 ROUGE 점수를 계산합니다.
    results = rouge.get_scores(tokenized_preds, tokenized_labels, avg=True)

    # ROUGE 점수 중 F1 score를 사용하여 평가합니다.
    rouge_1_f1 = results['rouge-1']['f']
    rouge_2_f1 = results['rouge-2']['f']
    rouge_l_f1 = results['rouge-l']['f']

    # Final Score 계산
    final_score = (rouge_1_f1 + rouge_2_f1 + rouge_l_f1) / 3

    # 결과 출력
    result = {
        "ROUGE-1 F1": rouge_1_f1,
        "ROUGE-2 F1": rouge_2_f1,
        "ROUGE-L F1": rouge_l_f1,
        "Final Score": final_score
    }
    
    return result
