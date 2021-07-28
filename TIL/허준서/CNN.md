# Convolutional Neural Network (CNN)

수정날짜: 2021년 7월 28일 오후 1:53
작성날짜: 2021년 7월 26일 오후 10:04

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled.png)

위 그림과 같이 CNN은 Convolution layer와 Pooling layer로 구성된다.

# 1. Convolution layer

합성곱층에서는 합성곱 연산을 통해 이미지의 공간적 특징을 보존하면서 특성을 추출한다.

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%201.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%201.png)

합성곱 연산은 위와 같이 입력 데이터 위에 커널을 올려두고, 커널과 입력의 곱들을 합함으로써 수행된다.

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%202.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%202.png)

이런 과정을 통해 나온 결과를 특성 맵(feature map)이라고 하며, 특성 맵은 커널과 스트라이드(stride)에 의해 결정된다. 

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%203.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%203.png)

입력 이미지의 크기는 5x5 였지만 특성 맵의 크기는 3x3으로 작아졌다. 만약 입력의 크기를 그대로 유지하고 싶은 경우 패딩을 추가할 수 있다.

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%204.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%204.png)

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%205.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%205.png)

위의 두 그림은 다층 퍼셉트론을 사용한 경우와 CNN을 사용한 경우 인공 신경망에서 어떤 식으로 작동하는지 표현한 것이다.

다층 퍼셉트론을 사용하는 경우 9x4 크기의 가중치 행렬이 필요하지만 합성곱 연산을 사용하는 경우 커널의 크기인 2x2 크기의 가중치 행렬이 사용되었다.

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%206.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%206.png)

데이터의 채널이 여러 개인 경우 커널의 채널도 데이터의 채널 수와 같게 해야한다.

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%207.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%207.png)

이때 커널은 여러 개가 되는게 아니라, 여러 개의 채널을 가진 하나의 커널이 된다.

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%208.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%208.png)

하지만 여러 개의 커널을 쓸 수도 있는데 이런 경우 특성 맵은 커널의 수와 같은 수의 채널을 갖는다.

# 2. Pooling

합성곱 층에서 합성곱 연산, 활성화 함수 연산을 하고 난 뒤 보통 풀링층을 지나게 된다. 풀링층에서는 풀링 연산이 이루어지는데 일반적으로 max pooling과 average pooling으로 나뉜다.

![Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%209.png](Convolutional%20Neural%20Network%20(CNN)%205ccf3966974840bca58385c05ddf96db/Untitled%209.png)

위 그림은 스트라이드가 2이고 크기가 2x2인 커널로 max pooling 연산을 하는 과정을 보여준다. 풀링 연산을 통해 특성 맵의 크기를 줄임으로써 가중치의 개수를 줄일 수 있다.

# 3. 모델 구현

여기서는 합성곱(nn.Conv2d) + 활성화 함수(nn.ReLU) + 맥스풀링(nn.MaxPoold2d)을 하나의 합성곱 층으로 본다고 가정한다.

```python
import torch
import torch.nn as nn
```

모듈을 임포트하고

```python
# 배치 크기 × 채널 × 높이(height) × 너비(widht)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)
```

입력으로 사용할 임의의 텐서를 정의한다.

```python
conv1 = nn.Conv2d(1, 32, 3, padding=1)
#Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```

첫번째 합성곱 층이다. 1채널의 입력을 받아서 32채널을 내보낸다. 커널 사이즈는 3이고 패딩은 1이다.

```python
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```

두번째 합성곱 층이다. 32채널의 입력을 받고 64채널을 출력한다. 커널 사이즈는 3이고 패딩은 1이다.

```python
pool = nn.MaxPool2d(2)
#MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
```

커널 사이즈와 스트라이드가 2인 맥스풀링이다.

```python
out = conv1(inputs)
print(out.shape)
out = pool(out)
print(out.shape)

out = conv2(out)
print(out.shape)
out = pool(out)
print(out.shape)

out = out.view(out.size(0), -1) 
print(out.shape)

fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
out = fc(out)
print(out.shape)
```

입력을 합성곱 층에 넣고, 풀링하는 과정을 2번 반복한 후 리니어 함수에 통과시켜보면 아웃풋이 10차원의 텐서로 나오는 것을 볼 수 있다.

여기서는 ReLU 함수를 통과하는 과정을 생략했다.

# 4. MNIST 분류

```python
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
```

필요한 모듈을 임포트하고 하이퍼 파라미터를 설정한다.

```python
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)
```

토치비전 데이터셋을 정의해준다.

```python
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
```

위에서 정의한 데이터셋을 이용해 데이터로더를 정의해준다.

```python
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out
```

모델을 설계하자. 전체 코드는 위와 같다. 코드를 하나씩 살펴보자.

```python
self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
```

첫번째 레이어 부분이다. 합성곱, 렐루, 맥스풀을 하나의 층으로 합쳤다.

MNIST는 흑백 이미지 데이터이기 때문에 채널이 1개뿐이다. 따라서 입력은 1채널로 받고 출력은 32채널로 내보낸다.

```python
self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
```

두번째 레이어이다. 이전 레이어에서 32채널을 뽑아냈으므로 입력은 32채널로 받고 출력은 64채널로 내보낸다.

```python
self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
torch.nn.init.xavier_uniform_(self.fc.weight)
```

전결합층이다. 숫자가 10가지이기 때문에 출력으로 10차원 텐서를 내보낸다. 또한 전결합층의 가중치를 세이비어 초기화한다.

```python
def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out
```

forward연산 부분이다. 입력을 모든 층에 통과시키고 리턴한다.

```python
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(data_loader)
```

모델을 정의하고, 비용함수와 옵티마이저를 만든다.

```python
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
```

```python
[Epoch:    1] cost = 0.225698516
[Epoch:    2] cost = 0.0629934967
[Epoch:    3] cost = 0.0462695956
[Epoch:    4] cost = 0.0373955965
[Epoch:    5] cost = 0.0314149745
[Epoch:    6] cost = 0.0261254944
[Epoch:    7] cost = 0.0218432005
[Epoch:    8] cost = 0.0184181854
[Epoch:    9] cost = 0.0161410216
[Epoch:   10] cost = 0.0133445393
[Epoch:   11] cost = 0.0100770202
[Epoch:   12] cost = 0.0101299807
[Epoch:   13] cost = 0.00803094357
[Epoch:   14] cost = 0.00733287167
[Epoch:   15] cost = 0.00644240994
```

데이터로더를 이용해 미니배치 단위로 학습한다.

```python
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
```

```python
Accuracy: 0.9842999577522278
```

테스트 데이터로 모델을 평가한다.