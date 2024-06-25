# Week1

Deep Learning의 기본에 대해 공부하고 관련된 미션들을 풀어보자!

## Mission 1(난이도 하)

1. Pytorch Template [https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template) 을 이용하여 MNIST를 분류하는 MLP 모형 만들기

2. Pytorch 공식 튜토리얼 문서의 컴퓨터 비전 전이 학습 [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)을 이해하고 각 줄에 대한 주석 달기

## Mission 2(난이도 중)

1. Convolutional Neural Networks를 직접 구성하여 99% 이상의 성능을 내는 MNIST 분류기 만들기

2. Recurrent Neural Networks (RNN or LSTM or GRU)를 직접 구성하여 98% 이상의 성능을 내는 MNIST 분류기 만들기

3. Albumentation [https://albumentations.ai/](https://albumentations.ai/) 라이브러리를 이용하여 MNIST 데이터를 증강하여 99.5% 이상의 성능을 내는 MNIST 분류기 만들기

## Mission 3(난이도 상)

1. Convolution과 Activation 레이어만을 활용하여 MNIST 분류기 만들기

   - Flatten 연산 및 Fully Connected 레이어 없이 CNN을 만들기 위해서는 Global Average Pooling을 이용해 (b, 1, 1, dim)의 형태로 만든다.
   - 이후 1 x 1 conv를 사용해서 (b, 1, 1, num_classes) 형태로 바꿔준다.

2. Semi-supervised learning을 이용한 MNIST 분류기 만들기
   - 참고1 : [https://blog.est.ai/2020/11/ssl/](https://blog.est.ai/2020/11/ssl/)
   - 참고2 : [https://github.com/rubicco/mnist-semi-supervised](https://github.com/rubicco/mnist-semi-supervised)
