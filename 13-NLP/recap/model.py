import torch

from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, bidirec=False, device=None):
        super().__init__()
        """
        input_size : src vocab 크기 [batch_size, seq_len]
        embedding_size : 임베딩 벡터 크기
        hidden_size : hidden state 크기
        """
        self.device = device
        self.input_size = input_size ## x_t의 차원
        self.hidden_size = hidden_size ## h_t의 차원
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, embedding_size)

        # 양방향 GRU를 사용할지 여부에 따라 GRU 레이어
        # 양방향 GRU를 사용하면, n_direction을 2로 설정하고, 그렇지 않으면 1로 설정
        if bidirec:
            self.n_direction = 2
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        else:
            self.n_direction = 1
            self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)

    # 은닉 상태를 초기화하는 함수
    # 입력의 크기에 따라 0으로 채워진 텐서를 생성하고, CUDA를 사용할 경우 GPU로 이동
    def init_hidden(self, inputs):
        hidden = torch.zeros(self.n_layers * self.n_direction, inputs.size(0), self.hidden_size)

        return hidden.cuda() if self.device else hidden

    # 가중치를 초기화하는 함수
    # 임베딩 레이어와 GRU 레이어의 가중치를 xavier_uniform 방식으로 초기화
    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)

    # 순전파 정의
    # 입력과 입력 길이를 받아서 임베딩 레이어와 GRU 레이어를 통과
    # GRU 레이어의 출력과 은닉 상태를 반환
    def forward(self, inputs, input_lengths):
        hidden = self.init_hidden(inputs) ## 최초 hidden state 초기화

        embedded = self.embedding(inputs) ## 입력 임베딩 [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]

        ## 입력된 시퀀스 데이터에서 padding을 제거한 원래 길이로 되돌린다.
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        
        ## GRU에 입력.
        outputs, hidden = self.gru(packed, hidden)

        ## 다시 padding을 추가한다.
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # 레이어 수가 1보다 크면, 마지막 두 레이어의 은닉 상태를 사용하고, 그렇지 않으면 마지막 레이어의 은닉 상태를 사용
        if self.n_layers > 1:
            if self.n_direction == 2:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]

        ## output : 모든 시점 t의 hidden state
        ## torch.cat : 각 배치별 마지막 hidden state를 하나로 concat
        return outputs, torch.cat([h for h in hidden], 1).unsqueeze(1)


class Decoder(nn.Module):
    # 초기화 함수
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1, dropout_p=0.1, device=None):
        super().__init__()
        """
        input_size : trg vocab 크기
        embedding_size : embedding 크기
        hidden_size : hidden state 크기
        """

        self.device = device
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)

        ## GRU 레이어를 생성. 임베딩 크기 + 은닉층 크기, 은닉층 크기, 레이어 수를 인자로 받음
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, n_layers, batch_first=True)

        # 선형 레이어를 생성. 은닉층 크기 * 2, 입력 크기를 인자로 받음
        self.linear = nn.Linear(hidden_size * 2, input_size)

        # Attention 레이어를 생성. 은닉층 크기를 인자로 받음
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    # 은닉 상태를 초기화하는 함수
    def init_hidden(self,inputs):
        hidden = torch.zeros(self.n_layers, inputs.size(0), self.hidden_size)

        return hidden.cuda() if self.device else hidden

    # 가중치를 초기화하는 함수
    def init_weight(self):
        # 각 레이어의 가중치를 Xavier 초기화를 사용하여 초기화
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)
        self.attn.weight = nn.init.xavier_uniform(self.attn.weight)

    # Attention 메커니즘을 구현하는 함수
    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """
        # hidden의 차원을 변경
        hidden = hidden[0].unsqueeze(2)  # (1,B,D) -> (B,D,1)

        # encoder_outputs의 크기를 가져옴
        batch_size = encoder_outputs.size(0) # B
        max_len = encoder_outputs.size(1) # T
        # attention 에너지를 계산
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1)) # B*T,D -> B*T,D
        energies = energies.view(batch_size,max_len, -1) # B,T,D
        attn_energies = energies.bmm(hidden).squeeze(2) # B,T,D * B,D,1 --> B,T

        # softmax를 사용하여 attention 가중치를 계산
        alpha = F.softmax(attn_energies,1) # B,T
        alpha = alpha.unsqueeze(1) # B,1,T
        # context 벡터를 계산
        context = alpha.bmm(encoder_outputs) # B,1,T * B,T,D => B,1,D

        return context, alpha

    # 순전파 함수
    def forward(self, inputs, context, max_length, encoder_outputs, encoder_maskings=None, is_training=False):
        """
        inputs : [batch_size, 1] (LongTensor, SOS)
        context : [batch_size, 1, hidden_dim] (FloatTensor, 마지막 인코더 은닉 상태)
        max_length : int, 디코딩할 최대 길이
        encoder_outputs : [batch_size, seq_len, hidden_dim]
        encoder_maskings : B,T # ByteTensor
        is_training : bool, 드롭아웃을 훈련 단계에서만 적용하기 위함
        """
        hidden = self.init_hidden(inputs)

        embedded = self.embedding(inputs)
        if is_training:
            embedded = self.dropout(embedded)

        decode = []
        for i in range(max_length):
            # GRU의 입력으로 embedded(디코더 입력)와 context(인코더 마지막 hidden state)를 연결한 것을 사용
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c)

            # hidden과 context를 연결하여 concated를 생성
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c)

            ## Bahdanau Attention
            # 선형 레이어를 통해 score를 계산
            score = self.linear(concated.squeeze(0))

            # score에 softmax를 적용하여 확률 분포를 얻음
            softmaxed = F.log_softmax(score,1)
            
            # softmaxed를 decode 리스트에 추가
            decode.append(softmaxed)
            
            # softmaxed의 최대값 인덱스를 decoded에 저장
            decoded = softmaxed.max(1)[1]

            # decoded를 임베딩하여 embedded를 업데이트
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1}
            
            # 훈련 단계에서만 드롭아웃 적용
            if is_training:
                embedded = self.dropout(embedded)

            # attention을 사용하여 다음 context 벡터를 계산
            context, alpha = self.Attention(hidden, encoder_outputs, encoder_maskings)

        # decode 리스트를 텐서로 변환
        scores = torch.cat(decode, 1)  # [batch_size, max_length, vocab_size]

        return scores  # [batch_size, max_length, vocab_size]



    # 디코딩 함수
    def decode(self, context, encoder_outputs):
        # 디코딩을 시작하는 심볼을 start_decode에 저장
        start_decode = torch.LongTensor([[2] * 1]).transpose(0, 1)

        # start_decode를 임베딩
        embedded = self.embedding(start_decode)
        
        # 은닉 상태를 초기화
        hidden = self.init_hidden(start_decode)

        decodes = []
        attentions = []
        decoded = embedded
        # decoded가 종료 심볼이 될 때까지 반복
        while decoded.data.tolist()[0] != 3: # </s>까지
            # GRU의 입력으로 embedded와 context를 연결한 것을 사용
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
        
            # hidden과 context를 연결하여 concated를 생성
            concated = torch.cat((hidden, context.transpose(0, 1)), 2) # y_t = g(h_t,y_{t-1},c)
        
            # 선형 레이어를 통해 score를 계산
            score = self.linear(concated.squeeze(0))
        
            # score에 softmax를 적용하여 확률 분포를 얻음
            softmaxed = F.log_softmax(score,1)
            
            # softmaxed를 decodes 리스트에 추가
            decodes.append(softmaxed)
            
            # softmaxed의 최대값 인덱스를 decoded에 저장
            decoded = softmaxed.max(1)[1]
            
            # decoded를 임베딩하여 embedded를 업데이트
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1}
            
            # attention을 사용하여 다음 context 벡터를 계산
            context, alpha = self.Attention(hidden, encoder_outputs,None)
            
            # alpha를 attentions 리스트에 추가
            attentions.append(alpha.squeeze(1))

        # decodes 리스트를 텐서로 변환하고 최대값 인덱스를 반환
        return torch.cat(decodes).max(1)[1], torch.cat(attentions)