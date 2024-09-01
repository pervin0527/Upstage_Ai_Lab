import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import argparse

from transformers import EarlyStoppingCallback
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig

from utils.metrics import compute_metrics
from data.dataset import Preprocess, prepare_train_dataset
from utils.config_utils import load_config, save_config, make_save_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--config_path', type=str, default='./config.yaml', help='Path to the config file')
    args = parser.parse_args()

    return args


def load_tokenizer_and_model_for_train(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    bart_config = BartConfig(vocab_size=32000,
                             pad_token_id=0,
                             bos_token_id=2,
                             eos_token_id=3).from_pretrained(model_name)
    
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'], config=bart_config)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config['tokenizer']['path'], config=bart_config)
    # generate_model = BartForConditionalGeneration.from_pretrained('./pretrain')


    # special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    # tokenizer.add_special_tokens(special_tokens_dict)

    generate_model.resize_token_embeddings(len(tokenizer)) # 사전에 special token을 추가했으므로 재구성 해줍니다.
    generate_model.to(device)
    print(generate_model.config)

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model , tokenizer


def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    print('-'*10, 'Make training arguments', '-'*10,)
    
    # Optimizer 생성
    optimizer = torch.optim.AdamW(generate_model.parameters(), lr=config['training']['learning_rate'])
    
    # num_warmup_steps와 num_training_steps를 설정
    num_warmup_steps = config['training']['num_warmup_steps']
    num_training_steps = config['training']['num_training_steps']
    
    # 스케줄러 생성
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # TrainingArguments 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],  # 모델 출력 디렉토리
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],  # 학습 에폭 수
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],  # 학습 중 디바이스별 배치 크기
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],  # 평가 중 디바이스별 배치 크기
        warmup_ratio=config['training']['warmup_ratio'],  # 워밍업 스텝 비율
        weight_decay=config['training']['weight_decay'],  # 가중치 감쇠 (L2 정규화)
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],  # Gradient Accumulation Steps
        evaluation_strategy=config['training']['evaluation_strategy'],  # 평가 전략
        save_strategy=config['training']['save_strategy'],  # 체크포인트 저장 전략
        save_total_limit=config['training']['save_total_limit'],  # 총 저장할 체크포인트 개수
        fp16=config['training']['fp16'],  # FP16 Mixed Precision Training
        load_best_model_at_end=config['training']['load_best_model_at_end'],  # 가장 좋은 모델을 학습 종료 시점에 로드
        seed=config['training']['seed'],  # 랜덤 시드
        logging_dir=config['training']['logging_dir'],  # 로깅 디렉토리
        logging_strategy=config['training']['logging_strategy'],  # 로깅 전략
        predict_with_generate=config['training']['predict_with_generate'],  # 생성 후 평가 시 생성된 결과를 사용
        generation_max_length=config['training']['generation_max_length'],  # 생성 시 최대 길이
        do_train=config['training']['do_train'],  # 학습 여부
        do_eval=config['training']['do_eval'],  # 평가 여부
        report_to=config['training']['report_to']  # 로깅 툴 (예: wandb)
    )
    
    # EarlyStoppingCallback 설정 (Validation loss가 개선되지 않으면 학습 중단)
    my_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    
    print('-'*10, 'Make trainer', '-'*10,)
    
    # Trainer 인스턴스 생성
    trainer = Seq2SeqTrainer(
        model=generate_model,  # 학습할 모델
        args=training_args,  # 학습 인자
        train_dataset=train_inputs_dataset,  # 학습 데이터셋
        eval_dataset=val_inputs_dataset,  # 평가 데이터셋
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),  # 평가 지표 함수
        callbacks=[my_callback],  # 콜백 리스트 (EarlyStopping 포함)
        optimizers=(optimizer, lr_scheduler)  # 옵티마이저와 스케줄러
    )
    
    print('-'*10, 'Make trainer complete', '-'*10,)
    
    return trainer


def main(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    # 사용할 모델과 tokenizer를 불러옵니다.
    generate_model , tokenizer = load_tokenizer_and_model_for_train(cfg,device)
    print('-'*10,"tokenizer special tokens : ",tokenizer.special_tokens_map,'-'*10)

    # 학습에 사용할 데이터셋을 불러옵니다.
    preprocessor = Preprocess(cfg['tokenizer']['bos_token'], cfg['tokenizer']['eos_token'], cfg['tokenizer']['sep_token'])
    data_path = cfg['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(cfg,preprocessor, data_path, tokenizer)
    print(len(train_inputs_dataset))

    # Trainer 클래스를 불러옵니다.
    trainer = load_trainer_for_train(cfg, generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset)
    save_config(cfg, cfg['general']['output_dir'])
    trainer.train()   # 모델 학습을 시작합니다.

    # (선택) 모델 학습이 완료된 후 wandb를 종료합니다.
    # wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cfg = load_config(config_path)

    curr = make_save_dir(cfg['general']['output_dir'])
    cfg['general']['output_dir'] = curr
    cfg['training']['logging_dir'] = f"{curr}/logs"

    # pprint(cfg)
    main(cfg)