import re
import os
import random
import pandas as pd

from torch.utils.data import Dataset

class Preprocess:
    def __init__(self, bos_token: str, eos_token: str, sep_token: str, mask_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.mask_token = mask_token

    @staticmethod
    def make_set_as_df(file_path, is_train = True):
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname','dialogue','summary']]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[['fname','dialogue']]
            return test_df

    def add_sep_tokens(self, dialogue):
        # 화자가 바뀔 때 SEP 토큰을 추가하는 함수
        pattern = r'(#Person\d+#)'
        parts = re.split(pattern, dialogue)
        result = []
        prev_speaker = None
        for part in parts:
            if re.match(pattern, part):
                if prev_speaker and prev_speaker != part:
                    result.append(self.sep_token)
                prev_speaker = part
            result.append(part)
        return ''.join(result)

    def sentence_infilling(self, text, mask_prob=0.15):
        words = text.split()
        masked_words = []
        mask_length = 0
        for word in words:
            if random.random() < mask_prob:
                if mask_length == 0:
                    masked_words.append(self.mask_token)
                mask_length += 1
            else:
                if mask_length > 0:
                    masked_words[-1] = self.mask_token
                    mask_length = 0
                masked_words.append(word)
        if mask_length > 0:
            masked_words[-1] = self.mask_token
        return ' '.join(masked_words)

    def permutation(self, text, max_perm=3):
        sentences = text.split('.')
        num_perm = min(len(sentences), max_perm)
        perm_indices = list(range(num_perm))
        random.shuffle(perm_indices)
        sentences[:num_perm] = [sentences[i] for i in perm_indices]
        return '.'.join(sentences)

    def apply_augmentations(self, text, infill_prob=0.5, perm_prob=0.5):
        if random.random() < infill_prob:
            text = self.sentence_infilling(text)
        if random.random() < perm_prob:
            text = self.permutation(text)
        return text

    def make_input(self, dataset, is_test=False, apply_aug=True):
        if is_test:
            encoder_input = dataset['dialogue'].apply(self.add_sep_tokens)
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue'].apply(self.add_sep_tokens)
            if apply_aug:
                encoder_input = encoder_input.apply(self.apply_augmentations)
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()
        

def prepare_train_dataset(config, preprocessor: Preprocess, data_path, tokenizer):
    train_file_path = os.path.join(data_path, config['general']['train_file'])
    val_file_path = os.path.join(data_path, config['general']['valid_file'])

    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    # new_file_path = os.path.join(data_path, config['general']['new_file'])
    # new_data = pd.read_csv(new_file_path)
    # start_index = len(train_data)
    # new_data['fname'] = ['train_' + str(i) for i in range(start_index, start_index + len(new_data))]
    # train_data = pd.concat([train_data, new_data[['fname', 'dialogue', 'summary']]], ignore_index=True)


    encoder_input_train , decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val , decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)

    tokenized_encoder_inputs = tokenizer(encoder_input_train, return_tensors="pt", padding=True,
                            add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_inputs = tokenizer(decoder_input_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_ouputs = tokenizer(decoder_output_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs,len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(encoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_inputs = tokenizer(decoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_ouputs = tokenizer(decoder_output_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs,len(encoder_input_val))

    return train_inputs_dataset, val_inputs_dataset
        
class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len

class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len

class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len