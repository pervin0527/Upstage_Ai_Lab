import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, f1_score, precision_score
from datasets import Dataset, load_metric

df = pd.read_csv('./dataset/train.csv')
tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")

df['dialogue'] = df['dialogue'].apply(lambda x: ' [SEP] '.join(x.split('\n')))
df['dialogue'] = df['dialogue'].apply(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=512, return_tensors="pt"))
dataset = Dataset.from_pandas(df)

def preprocess_function(examples):
    inputs = examples['dialogue']
    return tokenizer(inputs, max_length=512, truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Masked Language Modeling을 위한 데이터 Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# 메트릭 계산 함수
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(dim=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision}

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 모델 학습
trainer.train()

# 평가
trainer.evaluate()
