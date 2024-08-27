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
    # print(len(dataset.X), len(dataset.Y))
    # print(dataset.X[0], dataset.Y[0])

    dataloader = dataset.get_dataloader(batch_size, True, num_workers)

    input_size = len(dataset.source2index)   # source vocabulary size
    output_size = len(dataset.target2index)  # target vocabulary size

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

    # 3. 손실 함수와 옵티마이저
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
            ## encoder에서 발생한 시점별 모든 hidden state, 마지막 hidden state
            encoder_outputs, encoder_hidden = encoder(source_batch, source_lengths) ## [batch_size, seq_len, hidden_dim], [batch_size, 1, hidden_dim]
            
            # 2) Decoder에 입력
            decoder_input = torch.LongTensor([[dataset.source2index['<EOS>']]] * source_batch.size(0)).to(device)
            context = encoder_hidden
            decoder_outputs = decoder(decoder_input, context, max_length=max(target_lengths), 
                                      encoder_outputs=encoder_outputs, is_training=True)
            
            # 3) 손실 계산
            loss = criterion(decoder_outputs, target_batch.view(-1))
            
            # 4) 역전파 및 옵티마이저 스텝
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()