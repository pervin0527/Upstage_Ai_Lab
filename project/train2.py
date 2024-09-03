import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import argparse
from transformers import PreTrainedTokenizerFast, AutoTokenizer, BartForConditionalGeneration, BartConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from konlpy.tag import Mecab

from utils.metrics import compute_metrics
from data.dataset import Preprocess, prepare_train_dataset
from utils.config_utils import load_config, save_config, make_save_dir

mecab = Mecab()

def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--config_path', type=str, default='./configs/config.yaml', help='Path to the config file')
    args = parser.parse_args()

    return args

def load_tokenizer_and_model_for_train(config, device):
    model_name = config['general']['model_name']
    # bart_config = BartConfig().from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # generate_model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)

    bart_config = BartConfig.from_pretrained(config['general']['model_cfg'])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config['tokenizer']['path'], config=bart_config)
    generate_model = BartForConditionalGeneration.from_pretrained('./pretrain')

    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)
    generate_model.resize_token_embeddings(len(tokenizer))

    generate_model.to(device)
    return generate_model, tokenizer

def tokenize_text(text):
    # 빈 문자열 처리
    if not text.strip():
        return 'empty'
    return ' '.join(mecab.morphs(text))

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, tokenizer, config, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # `compute_metrics` 사용하여 ROUGE 점수 계산
    from transformers import EvalPrediction
    predictions = torch.tensor(all_predictions)
    labels = torch.tensor(all_labels)
    eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    rouge_scores = compute_metrics(config, tokenizer, eval_pred)
    
    return avg_loss, rouge_scores

def train(model, train_dataloader, val_dataloader, tokenizer, config, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    num_epochs = config['training']['num_train_epochs']

    best_val_loss = float('inf')
    early_stopping_patience = config['training'].get('early_stopping_patience', 3)
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss, rouge_scores = evaluate(model, val_dataloader, tokenizer, config, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"ROUGE-1 F1 Score: {rouge_scores['rouge-1']:.4f}, ROUGE-2 F1 Score: {rouge_scores['rouge-2']:.4f}, ROUGE-L F1 Score: {rouge_scores['rouge-l']:.4f}")
        print()

        # Best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(config['general']['output_dir'], 'best.pth'))
            print("Best model saved!")
        else:
            early_stopping_counter += 1

        # Early Stopping
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered!")
            break

    # Last model 저장
    torch.save(model.state_dict(), os.path.join(config['general']['output_dir'], 'last.pth'))
    print("Last model saved!")

def main(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    generate_model, tokenizer = load_tokenizer_and_model_for_train(cfg, device)
    print(tokenizer.special_tokens_map)

    preprocessor = Preprocess(cfg['tokenizer']['bos_token'], cfg['tokenizer']['eos_token'])
    data_path = cfg['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(cfg, preprocessor, data_path, tokenizer)

    train_dataloader = DataLoader(train_inputs_dataset, batch_size=cfg['training']['per_device_train_batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_inputs_dataset, batch_size=cfg['training']['per_device_eval_batch_size'])

    save_config(cfg, cfg['general']['output_dir'])
    train(generate_model, train_dataloader, val_dataloader, tokenizer, cfg, device)

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cfg = load_config(config_path)

    curr = make_save_dir(cfg['general']['output_dir'])
    cfg['general']['output_dir'] = curr
    cfg['training']['logging_dir'] = f"{curr}/logs"

    main(cfg)
