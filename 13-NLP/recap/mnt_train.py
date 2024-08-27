import torch

from torch import nn
from torch import optim
from data import MNTDataset
from model import Encoder, Decoder

def main():
    min_seq_len = 3
    max_seq_len = 25
    epochs = 100
    batch_size = 256
    embedding_size = 256
    hidden_size = 512
    num_workers = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MNTDataset("/home/pervinco/Datasets/fra-eng/fra.txt", min_seq_len, max_seq_len)

    dataloader = dataset.get_dataloader(batch_size, True, num_workers)

    input_size = len(dataset.source2index)   # source vocabulary size
    output_size = len(dataset.target2index)  # target vocabulary size
    print(input_size, output_size)

    encoder = Encoder(input_size=input_size, 
                      embedding_size=embedding_size, 
                      hidden_size=hidden_size, 
                      n_layers=1, 
                      bidirec=False,
                      device=device)

    decoder = Decoder(input_size=output_size, 
                      embedding_size=embedding_size, 
                      hidden_size=hidden_size, 
                      n_layers=1, 
                      dropout_p=0.1,
                      device=device)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 손실 함수와 옵티마이저
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.target2index['<PAD>'])  # 패딩 인덱스를 무시
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        for batch in dataloader:
            source_batch, target_batch, source_lengths, target_lengths = batch
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            # 1) Encoder에 입력
            encoder_outputs, encoder_hidden = encoder(source_batch, source_lengths) 
            
            # 2) Decoder에 입력 - '<SOS>' 토큰 사용
            decoder_input = torch.LongTensor([[dataset.target2index['<SOS>']]] * source_batch.size(0)).to(device)
            context = encoder_hidden
            
            # 디코더에서 타겟 시퀀스의 길이를 사용하도록 함
            decoder_outputs = decoder(decoder_input, context, max_length=target_batch.size(1),
                                      encoder_outputs=encoder_outputs, is_training=True)
            
            # decoder_outputs: [batch_size, max_length, vocab_size]
            # target_batch: [batch_size, max_length]
            
            # decoder_outputs을 [batch_size * max_length, vocab_size]로 변경
            decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))

            # target_batch을 [batch_size * max_length]로 변경
            target_batch = target_batch.view(-1)

            # 손실 계산
            loss = criterion(decoder_outputs, target_batch)
            
            # 역전파 및 옵티마이저 스텝
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()
