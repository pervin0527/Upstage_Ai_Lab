import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import argparse

from transformers import EarlyStoppingCallback
from transformers import PreTrainedTokenizerFast
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig

from utils.metrics import compute_metrics
from data.dataset import Preprocess, prepare_train_dataset
from utils.config_utils import load_config, save_config, make_save_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--config_path', type=str, default='./configs/config.yaml', help='Path to the config file')
    args = parser.parse_args()

    return args


def load_tokenizer_and_model_for_train(config,device):
    bart_config = BartConfig.from_pretrained(config['general']['model_cfg'])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config['tokenizer']['path'], config=bart_config)
    generate_model = BartForConditionalGeneration.from_pretrained('./pretrain')

    # model_name = config['general']['model_name']
    # bart_config = BartConfig().from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # generate_model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)

    special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)
    generate_model.resize_token_embeddings(len(tokenizer))

    generate_model.to(device)
    return generate_model , tokenizer


def load_trainer_for_train(config,generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset):
    # set training args
    training_args = Seq2SeqTrainingArguments(
                output_dir=config['general']['output_dir'], # model output directory
                overwrite_output_dir=config['training']['overwrite_output_dir'],
                num_train_epochs=config['training']['num_train_epochs'],  # total number of training epochs
                learning_rate=config['training']['learning_rate'], # learning_rate
                per_device_train_batch_size=config['training']['per_device_train_batch_size'], # batch size per device during training
                per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],# batch size for evaluation
                warmup_ratio=config['training']['warmup_ratio'],  # number of warmup steps for learning rate scheduler
                weight_decay=config['training']['weight_decay'],  # strength of weight decay
                lr_scheduler_type=config['training']['lr_scheduler_type'],
                optim =config['training']['optim'],
                gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
                evaluation_strategy=config['training']['evaluation_strategy'], # evaluation strategy to adopt during training
                save_strategy =config['training']['save_strategy'],
                save_total_limit=config['training']['save_total_limit'], # number of total save model.
                fp16=config['training']['fp16'],
                load_best_model_at_end=config['training']['load_best_model_at_end'], # 최종적으로 가장 높은 점수 저장
                seed=config['training']['seed'],
                logging_dir=config['training']['logging_dir'], # directory for storing logs
                logging_strategy=config['training']['logging_strategy'],
                predict_with_generate=config['training']['predict_with_generate'], #To use BLEU or ROUGE score
                generation_max_length=config['training']['generation_max_length'],
                do_train=config['training']['do_train'],
                do_eval=config['training']['do_eval'],
                report_to=config['training']['report_to'] # (선택) wandb를 사용할 때 설정합니다.
            )

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics = lambda pred: compute_metrics(config,tokenizer, pred),
        callbacks = [MyCallback]
    )

    return trainer


def main(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')

    generate_model, tokenizer = load_tokenizer_and_model_for_train(cfg, device)
    print(tokenizer.special_tokens_map)

    preprocessor = Preprocess(cfg['tokenizer']['bos_token'], cfg['tokenizer']['eos_token'])
    data_path = cfg['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(cfg, preprocessor, data_path, tokenizer)

    trainer = load_trainer_for_train(cfg, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    save_config(cfg, cfg['general']['output_dir'])
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cfg = load_config(config_path)

    curr = make_save_dir(cfg['general']['output_dir'])
    cfg['general']['output_dir'] = curr
    cfg['training']['logging_dir'] = f"{curr}/logs"

    main(cfg)