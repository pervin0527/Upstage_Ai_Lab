import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizerFast, AdamW
from tqdm import tqdm

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

    def permutation(self, tokens):
        n = len(tokens)
        if n <= 1:
            return tokens
        split_point = random.randint(1, n-1)
        return tokens[split_point:] + tokens[:split_point]

    def token_masking(self, tokens, mask_token):
        return [mask_token if random.random() < 0.15 else token for token in tokens]

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]

        # 대화에 [SEP] 토큰 추가
        dialogue = dialogue.replace('\n', f' {self.tokenizer.sep_token} ')

        # 토큰화
        tokens = self.tokenizer.tokenize(dialogue)

        # Text Infilling
        tokens = self.text_infilling(tokens, self.tokenizer.mask_token)

        # Permutation
        tokens = self.permutation(tokens)

        # Token Masking
        tokens = self.token_masking(tokens, self.tokenizer.mask_token)

        # 다시 텍스트로 변환
        augmented_dialogue = self.tokenizer.convert_tokens_to_string(tokens)

        # 입력 인코딩
        inputs = self.tokenizer(augmented_dialogue, 
                                truncation=True, 
                                max_length=self.max_length, 
                                padding='max_length', 
                                return_tensors='pt')

        # 타겟 인코딩 (원본 대화 사용)
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

def train(model, train_loader, val_loader, optimizer, device, num_epochs):
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
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels)
                
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print()

    return model

def main():
    MAX_LENGTH = 1000
    BATCH_SIZE = 12
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = load_data('./dataset/cleaned_train.csv')
    val_df = load_data('./dataset/cleaned_dev.csv')
    new_df = load_data('./dataset/new_data.csv')

    train_dialogues = train_df[['dialogue']]
    new_dialogues = new_df[['dialogue']]
    train_df = pd.concat([train_dialogues, new_dialogues], ignore_index=True)
    # train_df = train_df.sample(frac=0.45, random_state=42)

    tokenizer = BartTokenizerFast.from_pretrained('./tokenizer')

    train_dataset = AugmentedDialogueDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = AugmentedDialogueDataset(val_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    trained_model = train(model, train_loader, val_loader, optimizer, device, NUM_EPOCHS)
    trained_model.save_pretrained('./pretrain')
    tokenizer.save_pretrained('./pretrain/tokenizer')

if __name__ == "__main__":
    main()