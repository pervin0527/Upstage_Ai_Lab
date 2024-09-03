import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
nltk.download('punkt')

import torch
import random
import argparse
import numpy as np

from transformers import EarlyStoppingCallback, Trainer
from transformers import PreTrainedTokenizerFast
from transformers import Seq2SeqTrainingArguments
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig

from datasets import load_metric
from typing import Dict, List, Tuple
from transformers import EvalPrediction
from data.dataset import Preprocess, prepare_train_dataset
from utils.config_utils import load_config, save_config, make_save_dir

import torch.nn.functional as F
import pytorch_lightning as pl

def compute_metrics(config, tokenizer, pred: EvalPrediction) -> Dict:
    rouge_metric = load_metric("rouge")
    
    predictions, labels = pred.predictions, pred.label_ids
    
    # 예측값이 로짓인 경우 처리
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # -100을 패딩 토큰 ID로 변경
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 토큰 ID를 텍스트로 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(tokenizer.tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(tokenizer.tokenize(label.strip())) for label in decoded_labels]
    
    # Rouge 점수 계산
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # 중간값 계산
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # 정밀도를 2자리로 반올림
    result = {k: round(v, 2) for k, v in result.items()}
    
    return result

class R3FModule(pl.LightningModule):
    def __init__(
        self,
        model: BartForConditionalGeneration,
        r3f_lambda: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.r3f_lambda = r3f_lambda
        self.noise_sampler = torch.distributions.normal.Normal(loc=0.0, scale=1e-5)

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
        ) / noised_logits.size(0)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels=None):
        inputs_embeds = self.model.model.shared(input_ids)
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True,
        )

        if self.training:
            noise = self.noise_sampler.sample(sample_shape=inputs_embeds.shape).to(inputs_embeds)
            noise_embeds = inputs_embeds.detach().clone() + noise
            noise_output = self.model(
                inputs_embeds=noise_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                return_dict=True,
            )

            symm_kl = self._get_symm_kl(noise_output.logits, output.logits)
            output.loss += self.r3f_lambda * symm_kl

        return output

class R3FTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            labels=inputs["labels"]
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--config_path', type=str, default='./configs/config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

def load_tokenizer_and_model_for_train(config, device):
    model_name = config['general']['model_name']
    print(model_name)
    bart_config = BartConfig.from_pretrained(model_name)
    print(bart_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
    special_tokens_dict = {
        'sep_token': config['tokenizer']['sep_token'],
        'additional_special_tokens': config['tokenizer']['special_tokens']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    base_model.resize_token_embeddings(len(tokenizer))

    r3f_model = R3FModule(base_model, r3f_lambda=config['training'].get('r3f_lambda', 1.0))
    r3f_model.to(device)
    return r3f_model, tokenizer

def load_trainer_for_train(config, r3f_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
    )

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    trainer = R3FTrainer(
        model=r3f_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[MyCallback]
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['training']['seed'])

    r3f_model, tokenizer = load_tokenizer_and_model_for_train(cfg, device)
    print(tokenizer.special_tokens_map)

    preprocessor = Preprocess(cfg['tokenizer']['bos_token'], cfg['tokenizer']['eos_token'], cfg['tokenizer']['sep_token'])
    data_path = cfg['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(cfg, preprocessor, data_path, tokenizer)

    trainer = load_trainer_for_train(cfg, r3f_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
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