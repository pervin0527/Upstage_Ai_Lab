import torch
import random
import pandas as pd

from tqdm import tqdm
from rouge import Rouge
from konlpy.tag import Mecab
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizerFast, AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score

from utils.metrics import compute_metrics

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

    # ROUGE 점수 중 F-1 score를 통해 평가합니다.
    result = {key: value["f"] for key, value in results.items()}
    
    return result

class AugmentedDialogueDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dialogues = dataframe['dialogue'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def text_infilling(self, tokens, mask_token):
        output = []
        i = 0
        while i < len(tokens):
            if random.random() < 0.15:  # 15% 확률로 text infilling 적용
                span_length = min(random.randint(1, 5), len(tokens) - i)
                output.append(mask_token)
                i += span_length
            else:
                output.append(tokens[i])
                i += 1
        return output

    def sentence_permutation(self, dialogue):
        sentences = dialogue.split(self.tokenizer.sep_token)
        if len(sentences) > 1:
            random.shuffle(sentences)
        return f" {self.tokenizer.sep_token} ".join(sentences)

    def token_masking(self, tokens, mask_token):
        return [mask_token if random.random() < 0.15 else token for token in tokens]

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        dialogue = dialogue.replace('\n', f' {self.tokenizer.sep_token} ')

        # 문장 순서 섞기
        dialogue = self.sentence_permutation(dialogue)
        
        tokens = self.tokenizer.tokenize(dialogue)
        tokens = self.text_infilling(tokens, self.tokenizer.mask_token)
        tokens = self.token_masking(tokens, self.tokenizer.mask_token)
        augmented_dialogue = self.tokenizer.convert_tokens_to_string(tokens)
        inputs = self.tokenizer(augmented_dialogue, 
                                truncation=True, 
                                max_length=self.max_length, 
                                padding='max_length', 
                                return_tensors='pt')
        targets = self.tokenizer(dialogue, 
                                 truncation=True, 
                                 max_length=self.max_length, 
                                 padding='max_length', 
                                 return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def load_data(file_path):
    return pd.read_csv(file_path)

def train(model, train_loader, val_loader, optimizer, device, num_epochs, tokenizer, config):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels)
                
                val_loss += outputs.loss.item()
                all_predictions.extend(outputs.logits.argmax(dim=-1).cpu())
                all_labels.extend(labels.cpu())

        avg_val_loss = val_loss / len(val_loader)

        # metrics 계산
        from transformers import EvalPrediction
        predictions = torch.stack(all_predictions)
        labels = torch.stack(all_labels)
        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        rouge_scores = compute_metrics(config, tokenizer, eval_pred)
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"ROUGE-1 F1 Score: {rouge_scores['rouge-1']:.4f}, ROUGE-2 F1 Score: {rouge_scores['rouge-2']:.4f}, ROUGE-L F1 Score: {rouge_scores['rouge-l']:.4f}")
        print()

    return model

def main():
    MODEL_NAME = 'EbanLee/kobart-summary-v3'
    TOKENIZER_PATH = "./tokenizer"
    MAX_LENGTH = 1000
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = load_data('./dataset/cleaned_train.csv')
    val_df = load_data('./dataset/cleaned_dev.csv')

    tokenizer = BartTokenizerFast.from_pretrained(TOKENIZER_PATH)
    print(tokenizer.special_tokens_map)

    train_dataset = AugmentedDialogueDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = AugmentedDialogueDataset(val_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    config = {
        'inference': {
            'remove_tokens': tokenizer.all_special_tokens
        }
    }

    trained_model = train(model, train_loader, val_loader, optimizer, device, NUM_EPOCHS, tokenizer, config)
    trained_model.save_pretrained('./pretrain')
    tokenizer.save_pretrained('./pretrain/tokenizer')

if __name__ == "__main__":
    main()
