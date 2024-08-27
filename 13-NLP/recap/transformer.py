import math
import torch

from torch import nn

device = "cuda" if torch.cuda.is_available else None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        # Positional Encoding을 계산하기 위한 빈 텐서 생성
        # 이 텐서의 크기는 (max_seq_length, d_model)
        positional_encoding = torch.zeros(max_seq_length, d_model)

        # 각 위치(position)에 대한 정보를 담은 텐서를 생성
        # 이 텐서의 크기는 (max_seq_length, 1)임
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # 각 차원(dimension)에 대한 분모(div_term)를 계산
        # 이 텐서의 크기는 (d_model // 2,)임
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 짝수 인덱스에는 sin 함수를 적용
        # 짝수 인덱스에 대한 수식: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)

        # 홀수 인덱스에는 cos 함수를 적용
        # 홀수 인덱스에 대한 수식: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        positional_encoding[:, 1::2] = torch.cos(position * div_term)


        # 계산된 Positional Encoding을 모듈의 버퍼에 등록
        # 이를 통해 모듈의 state_dict에 포함되어 저장 및 로딩이 가능해짐
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))

    # 순전파 함수를 정의
    def forward(self, x):
        # 입력 x와 Positional Encoding을 더하여 반환
        # 이 때, 입력 x의 시퀀스 길이에 맞추어 Positional Encoding을 슬라이싱함
        return x + self.positional_encoding[:, :x.size(1)]


class FeedForwardNN(nn.Module):
    def __init__(self, d_model, feedforward_dim):
        super().__init__()
        # 선형 변환을 위한 Linear 레이어와 활성화 함수 정의
        self.fc1 = nn.Linear(d_model, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, d_model)
        self.relu = nn.ReLU()

    # 순전파 정의
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어떨어져야 한다."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        ## 8개의 head별로 Q, K, V를 독립적으로 만드는 것은 비효율적이니 한 번에 만들자.
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # Query, Key, Value에 각각 선형 변환을 적용하고 헤드를 분할
        query = self.split_heads(self.query_linear(query))
        key = self.split_heads(self.key_linear(key))
        value = self.split_heads(self.value_linear(value))

        # Scaled Dot-Product Attention을 계산
        attention_output = self.scaled_dot_product_attention(query, key, value, mask)

        # 분할된 헤드를 다시 합치고, 선형 변환을 적용
        output = self.output_linear(self.combine_heads(attention_output))

        return output

    # 입력 Tensor를 헤드 수만큼 분할
    def split_heads(self, tensor):
        batch_size, seq_length, d_model = tensor.size()
        return tensor.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    # 분할된 헤드를 다시 합침
    def combine_heads(self, tensor):
        batch_size, _, seq_length, d_k = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    # Scaled Dot-Product Attention을 계산하는 함수
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # Query와 Key의 행렬 곱을 계산하고, Key 차원의 제곱근으로 나눠줌
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.key_dim)

        # 마스크가 주어진 경우, 마스크가 0인 위치에 매우 작은 값을 채워줌
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Softmax를 적용해 Attention 확률을 계산
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Attention 확률과 Value의 행렬 곱을 계산
        output = torch.matmul(attention_probs, value)

        return output
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, feedforward_dim, dropout):
        super().__init__()
        # MultiHeadAttention과 FeedForwardNN 정의
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNN(d_model, feedforward_dim)
        # Layer Normalization과 Dropout 정의
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # 순전파 정의
    def forward(self, x, mask):
        # Self-Attention을 계산하고, 결과를 원래의 입력과 더한 후 Layer Normalization 적용
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        # Feedforward를 계산하고, 결과를 원래의 입력과 더한 후 Layer Normalization 적용
        feedforward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feedforward_output))

        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, feedforward_dim, dropout):
        super().__init__()
        # MultiHeadAttention과 FeedForwardNN 정의
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNN(d_model, feedforward_dim)
        # Layer Normalization과 Dropout을 정의
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # 순전파 정의
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Self-Attention을 계산하고, 결과를 원래의 입력과 더한 후 Layer Normalization을 적용
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_output))
        # Cross-Attention을 계산하고, 결과를 원래의 입력과 더한 후 Layer Normalization을 적용
        attention_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attention_output))
        # Feedforward를 계산하고, 결과를 원래의 입력과 더한 후 Layer Normalization을 적용
        feedforward_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feedforward_output))

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim, num_heads, num_layers, feedforward_dim, max_seq_length, dropout):
        super(Transformer, self).__init__()
        # Embedding과 Positional Encoding 정의
        self.encoder_embedding = nn.Embedding(src_vocab_size, model_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_seq_length)

        # Encoder와 Decoder 레이어를 정의
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, feedforward_dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, feedforward_dim, dropout) for _ in range(num_layers)])

        # 최종 출력을 위한 선형 변환 레이어와 Dropout을 정의
        self.fc = nn.Linear(model_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # 마스크를 생성하는 함수 (Decoder의 self-attention)
    def generate_mask(self, src, tgt):
        # 입력된 소스와 타겟에서 각각 0이 아닌 위치를 찾아 마스크를 생성
        # attention 스코어와 연산을 할 수 있게 하기 위해, unsqueeze를 사용하여 차원을 추가
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)

        # 타겟의 시퀀스 길이를 가져옴
        seq_length = tgt.size(1)

        # nopeak_mask는 디코더가 자신보다 미래의 단어를 참조하지 못하게 하는 마스크
        # 대각선 아래쪽은 1, 위쪽은 0인 상삼각행렬을 생성하고, 이를 불리언 타입으로 변환
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)

        # 타겟 마스크와 nopeak_mask를 AND 연산하여 최종 타겟 마스크를 생성
        # 이 마스크는 디코더가 패딩 위치 뿐 아니라 자신보다 미래의 단어를 참조하지 못하게 함
        tgt_mask = tgt_mask & nopeak_mask

        # 소스 마스크와 타겟 마스크를 반환
        return src_mask, tgt_mask


    # 순전파 정의
    def forward(self, src, tgt):
        # 마스크 생성
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # 소스와 타겟에 각각 Embedding과 Positional Encoding을 적용
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Encoder를 통과 (Encoder 순전파)
        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)

        # Decoder를 통과 (Decoder 순전파)
        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)

        # 최종 출력 계산
        output = self.fc(decoder_output)
        return output