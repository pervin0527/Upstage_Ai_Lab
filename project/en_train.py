import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import random
import argparse
import numpy as np

from rouge import Rouge
from transformers import EarlyStoppingCallback
from transformers import PreTrainedTokenizerFast
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig

from data.dataset import Preprocess, prepare_train_dataset
from utils.config_utils import load_config, save_config, make_save_dir

def compute_metrics(config,tokenizer,pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    replaced_predictions = decoded_preds.copy()
    replaced_labels = labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    print('-'*150)
    print(f"PRED: {replaced_predictions[0]}")
    print(f"GOLD: {replaced_labels[0]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[1]}")
    print(f"GOLD: {replaced_labels[1]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[2]}")
    print(f"GOLD: {replaced_labels[2]}")

    results = rouge.get_scores(replaced_predictions, replaced_labels,avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--config_path', type=str, default='./configs/config.yaml', help='Path to the config file')
    args = parser.parse_args()

    return args


def load_tokenizer_and_model_for_train(config,device):
    model_name = config['general']['model_name']
    print(model_name)
    bart_config = BartConfig.from_pretrained(model_name)
    print(bart_config)

    ## Pretrained Tokenizer + Model
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(config['tokenizer']['path'], config=bart_config)
    # generate_model = BartForConditionalGeneration.from_pretrained('./pretrain')

    ## Huggingface Pretrained Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
    special_tokens_dict = {
        'sep_token': config['tokenizer']['sep_token'],
        'additional_special_tokens' : config['tokenizer']['special_tokens']
    }
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    set_seed(cfg['training']['seed'])

    generate_model, tokenizer = load_tokenizer_and_model_for_train(cfg, device)
    print(tokenizer.special_tokens_map)

    # preprocessor = Preprocess(cfg['tokenizer']['bos_token'], cfg['tokenizer']['eos_token'])
    preprocessor = Preprocess(cfg['tokenizer']['bos_token'], cfg['tokenizer']['eos_token'], cfg['tokenizer']['sep_token'])
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