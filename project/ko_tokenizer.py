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

SENTENCEPIECE_URI = "https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece/"
PAD, UNK, BOS, EOS, MASK, SEP = "<pad>", "<unk>", "<s>", "</s>", "<mask>", "<sep>"

parser = argparse.ArgumentParser(prog="train_tokenizer", description="Training Huggingface Tokenizer with optional Mecab preprocessing")
parser.add_argument("--tokenizer-path", type=str, default="./tokenizer/ko_sentencepiece", help="path to save tokenizer")
parser.add_argument("--vocab-size", type=int, default=8113, help="vocab size of tokenizer")
parser.add_argument("--use-mecab", action="store_true", help="whether to use Mecab for tokenization")

special_words = [
    '#Person1#',
    '#Person2#',
    '#Person3#',
    '#Person4#',
    '#Person5#',
    '#Person6#',
    '#Person7#',
    '#SSN#',
    '#Email#',
    '#Address#',
    '#Reaction#',
    '#CarNumber#',
    '#Movietitle#',
    '#DateOfBirth#',
    '#CardNumber#',
    '#PhoneNumber#',
    '#PassportNumber#',

    '#Person1#:',
    '#Person2#:',
    '#Person3#:',
    '#Person4#:',
    '#Person5#:',
    '#Person6#:',
    '#Person7#:',
]

def mecab_tokenize(text):
    mecab = Mecab()
    return ' '.join(mecab.morphs(text))

def main(args: argparse.Namespace):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_prefix = os.path.join(tmpdir, "tokenizer")

        train_df = pd.read_csv("./dataset/cleaned_train.csv")
        valid_df = pd.read_csv("./dataset/cleaned_dev.csv")
        test_df = pd.read_csv("./dataset/test.csv")
        df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        
        text_data = df['dialogue'].astype(str)

        # Mecab 토큰화를 선택적으로 적용
        if args.use_mecab:
            print("Using Mecab for tokenization.")
            tokenized_text_data = text_data.apply(mecab_tokenize)
        else:
            print("Skipping Mecab tokenization.")
            tokenized_text_data = text_data

        temp_input_file = os.path.join(tmpdir, "input.txt")
        with open(temp_input_file, 'w', encoding='utf-8') as f:
            for line in tokenized_text_data:
                f.write(line + "\n")

        spm.SentencePieceTrainer.train(input=temp_input_file,
                                       model_prefix=model_prefix,
                                       model_type="unigram",
                                       vocab_size=args.vocab_size,
                                       pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                                       pad_piece=PAD, unk_piece=UNK, bos_piece=BOS, eos_piece=EOS,
                                       user_defined_symbols=[MASK, SEP, *special_words])

        with httpimport.remote_repo(SENTENCEPIECE_URI):
            import sentencepiece_model_pb2
            tokenizer = SentencePieceUnigramTokenizer.from_spm(model_prefix + ".model")

    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                                   bos_token=BOS, eos_token=EOS, cls_token=BOS,
                                                   unk_token=UNK, sep_token=SEP, pad_token=PAD, mask_token=MASK,
                                                   additional_special_tokens=special_words)
    
    pretrained_tokenizer.save_pretrained(args.tokenizer_path)
    print(f"[+] Saved to {args.tokenizer_path}")
    print(pretrained_tokenizer.special_tokens_map)

    sample_dialogue = random.choice(df['dialogue'])
    print("\n샘플 대화:")
    print(sample_dialogue)
    
    if args.use_mecab:
        mecab_tokenized = mecab_tokenize(sample_dialogue)
        tokens = pretrained_tokenizer.tokenize(mecab_tokenized)
        print("\n토큰화 결과 (Mecab + SentencePiece):")
    else:
        tokens = pretrained_tokenizer.tokenize(sample_dialogue)
        print("\n토큰화 결과 (SentencePiece):")
    
    print(tokens)
    
    token_ids = pretrained_tokenizer.encode(mecab_tokenized if args.use_mecab else sample_dialogue)
    print("\n토큰 ID:")
    print(token_ids)
    
    decoded_text = pretrained_tokenizer.decode(token_ids)
    print("\n디코딩된 텍스트:")
    print(decoded_text)

if __name__ == "__main__":
    main(parser.parse_args())
