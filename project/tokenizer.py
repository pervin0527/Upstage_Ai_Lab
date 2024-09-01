import os
import random
import argparse
import tempfile
import httpimport
import pandas as pd
import sentencepiece as spm
from konlpy.tag import Mecab
from glob import glob
from transformers import PreTrainedTokenizerFast
from tokenizers import SentencePieceUnigramTokenizer

parser = argparse.ArgumentParser(prog="train_tokenizer", description="Training Huggingface Tokenizer with Mecab preprocessing")
parser.add_argument("--tokenizer-path", type=str, default="/home/pervinco/Upstage_Ai_Lab/project/tokenizer", help="path to save tokenizer")
parser.add_argument("--vocab-size", type=int, default=8105, help="vocab size of tokenizer")

special_words = [
    '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#',
    '#SSN#', '#Email#', '#Address#', '#CarNumber#', '#DateOfBirth#', '#CardNumber#', '#PhoneNumber#', '#PassportNumber#',
    '#Reaction#', '#Movietitle#'
]

SENTENCEPIECE_URI = "https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece/"

PAD, UNK, BOS, EOS, MASK, SEP = "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[SEP]"

def mecab_tokenize(text):
    mecab = Mecab()
    return ' '.join(mecab.morphs(text))

def main(args: argparse.Namespace):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_prefix = os.path.join(tmpdir, "tokenizer")

        # 데이터 파일 읽기
        train_df = pd.read_csv("./dataset/cleaned_train.csv")
        valid_df = pd.read_csv("./dataset/cleaned_dev.csv")
        # new_df = pd.read_csv("./dataset/new_data.csv")
        df = pd.concat([train_df, valid_df], ignore_index=True)
        
        # dialogue 컬럼의 데이터만 사용
        text_data = df['dialogue'].astype(str)
        
        # Mecab으로 분절화
        tokenized_text_data = text_data.apply(mecab_tokenize)
        
        # 텍스트 데이터를 임시 파일로 저장
        temp_input_file = os.path.join(tmpdir, "input.txt")
        with open(temp_input_file, 'w', encoding='utf-8') as f:
            for line in tokenized_text_data:
                f.write(line + "\n")

        spm.SentencePieceTrainer.train(
            input=temp_input_file,
            model_prefix=model_prefix,
            model_type="unigram",
            vocab_size=args.vocab_size,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece=PAD, unk_piece=UNK, bos_piece=BOS, eos_piece=EOS,
            user_defined_symbols=[MASK, SEP, *special_words],
        )

        with httpimport.remote_repo(SENTENCEPIECE_URI):
            import sentencepiece_model_pb2
            tokenizer = SentencePieceUnigramTokenizer.from_spm(model_prefix + ".model")

    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS, eos_token=EOS, cls_token=BOS,
        unk_token=UNK, sep_token=SEP, pad_token=PAD, mask_token=MASK,
        additional_special_tokens=special_words,
    )
    pretrained_tokenizer.save_pretrained(args.tokenizer_path)
    print(f"[+] Saved to {args.tokenizer_path}")

    # 샘플 대화 토큰화 및 결과 출력
    sample_dialogue = random.choice(df['dialogue'])
    print("\n샘플 대화:")
    print(sample_dialogue)
    
    # Mecab으로 분절화 후 토큰화
    mecab_tokenized = mecab_tokenize(sample_dialogue)
    tokens = pretrained_tokenizer.tokenize(mecab_tokenized)
    print("\n토큰화 결과 (Mecab + SentencePiece):")
    print(tokens)
    
    # 토큰 ID 출력
    token_ids = pretrained_tokenizer.encode(mecab_tokenized)
    print("\n토큰 ID:")
    print(token_ids)
    
    # 토큰 ID를 다시 텍스트로 디코딩
    decoded_text = pretrained_tokenizer.decode(token_ids)
    print("\n디코딩된 텍스트:")
    print(decoded_text)

if __name__ == "__main__":
    main(parser.parse_args())