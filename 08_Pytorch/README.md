## 1.Installation

[https://pytorch.kr/get-started/locally/](https://pytorch.kr/get-started/locally/)

    conda create --name DL python=3.8
    conda install pytorch==2.0.1 torchvision==0.15.2 torchtext==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
    conda install pexpect jupyter
    pip install pexpect jupyter ipykernel
    pip uninstall pyzmq
    pip install pyzmq
    pip install timm transformers


    conda create --name DL_BASIC python=3.9
    conda install pytorch torchvision torchaudio torchtext=0.18.0 pytorch-cuda=11.8 -c pytorch -c nvidia

설치가 정상적으로 되었는지 검사

```python
import torch

torch.__version__ ## '2.3.1'
torch.cuda.is_available() ## True
```
