import torch
import re
import unicodedata
from torch.utils.data import Dataset, DataLoader

# 유니코드 문자열을 아스키 문자열로 변환하는 함수
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 문자열을 정규화하는 함수
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

class MNTDataset(Dataset):
    def __init__(self, txt_path, min_length, max_length):
        self.X, self.Y = [], []
        self.min_length = min_length
        self.max_length = max_length
        self.corpus = open(txt_path, 'r', encoding='utf-8').readlines()

        self.preprocess()
        self.build_vocab()

    def preprocess(self):
        for parallel in self.corpus:
            src, trg, _ = parallel.strip().split('\t')

            ## src 문장이나 trg 문장이 비어있는 경우 제외.
            if src.strip() == "" or trg.strip() == "":
                continue

            ## 문장을 정규화하고 단어 단위로 분리.
            normalized_src = normalize_string(src).split()
            normalized_trg = normalize_string(trg).split()

            if len(normalized_src) >= self.min_length and len(normalized_trg) <= self.max_length \
            and len(normalized_trg) >= self.min_length and len(normalized_src) <= self.max_length:
                self.X.append(normalized_src)
                self.Y.append(normalized_trg)

    def build_vocab(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        self.source_vocab = list(set(flatten(self.X)))
        self.target_vocab = list(set(flatten(self.Y)))
        print(len(self.source_vocab), len(self.target_vocab))

        self.source2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.target2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

        # 소스 사전을 구축
        for vocab in self.source_vocab:
            if vocab not in self.source2index:  # 단어가 사전에 없으면 추가
                self.source2index[vocab] = len(self.source2index)
        self.index2source = {v: k for k, v in self.source2index.items()}

        # 타겟 사전을 구축
        for vocab in self.target_vocab:
            if vocab not in self.target2index:  # 단어가 사전에 없으면 추가
                self.target2index[vocab] = len(self.target2index)
        self.index2target = {v: k for k, v in self.target2index.items()}


    def prepare_sequence(self, seq, to_index, max_len=None):
        idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
        if max_len is not None:
            # 패딩 추가
            idxs = idxs + [to_index['<PAD>']] * (max_len - len(idxs))
            idxs = idxs[:max_len]  # 최대 길이를 초과하지 않도록 자름
        return torch.LongTensor(idxs)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        src = self.X[idx]
        trg = self.Y[idx]
        return src, trg

    def collate_fn(self, batch):
        # 각 배치에서 소스와 타겟 시퀀스를 분리
        batch_src, batch_trg = zip(*batch)

        # 소스와 타겟 시퀀스의 최대 길이
        max_len_src = max([len(s) for s in batch_src]) + 1  ## src내에서 가장 긴 길이, +1 for <EOS>
        max_len_trg = max([len(t) for t in batch_trg]) + 1  ## trg내에서 가장 긴 길이, +1 for <EOS>

        # 시퀀스를 인덱스로 변환하고 패딩을 추가
        src_sequences = [self.prepare_sequence(s + ['<EOS>'], self.source2index, max_len_src) for s in batch_src]
        trg_sequences = [self.prepare_sequence(t + ['<EOS>'], self.target2index, max_len_trg) for t in batch_trg]

        # 텐서로 결합
        src_batch = torch.stack(src_sequences)
        trg_batch = torch.stack(trg_sequences)

        # 각 시퀀스의 길이 계산
        src_lengths = [len(s) for s in batch_src]
        trg_lengths = [len(t) for t in batch_trg]

        return src_batch, trg_batch, src_lengths, trg_lengths

    def get_dataloader(self, batch_size, shuffle, num_workers):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

