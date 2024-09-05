import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

os.environ["HF_TOKEN"] = "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"

# model_id = "saltlux/Ko-Llama3-Luxia-8B"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id
)

def generate_response(user_message):
    system_message = """
    당신은 최고 수준의 영한 번역 전문가입니다. 주어진 영어 문자열를 한국어로 정확하고 자연스럽게 번역해야 합니다.
    이제 주어진 영어 텍스트를 위의 지침에 따라 한국어로 번역하세요. 오직 번역된 한국어 텍스트만 출력하세요.
    다음은 원하는 번역의 예시입니다. 이 스타일과 품질을 참고하여 번역하세요:

    영어: Mr. Smith's getting a check-up, and Doctor Hawkins advises him to have one every year. Hawkins'll give some information about their classes and medications to help Mr. Smith quit smoking.

    한국어: 스미스씨가 건강검진을 받고 있고, 호킨스 의사는 매년 건강검진을 받는 것을 권장합니다. 호킨스 의사는 스미스씨가 담배를 끊는 데 도움이 될 수 있는 수업과 약물에 대한 정보를 제공할 것입니다.
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.eos_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip()

def post_process(text):
    korean_sentences = re.findall(r'[가-힣\s]+[.!?]', text)
    if korean_sentences:
        return korean_sentences[0].strip()
    return text.strip()

def translate_summary(summary):
    translated = generate_response(summary)
    return post_process(translated)

if __name__ == "__main__":
    df = pd.read_csv('./prediction/output.csv')

    translated_summaries = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        summary = row['summary']
        print("*" * 100)
        print(f"{summary}\n\n")

        translated_summary = translate_summary(summary)
        print(f"{translated_summary}")
        print("*" * 100)

        # print()
        # translated_summaries.append(translated_summary)
        break

    # df['translated_summary'] = translated_summaries
    # df.to_csv('./prediction/translated_output.csv', index=False)