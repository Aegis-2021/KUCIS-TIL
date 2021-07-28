# Recurrent Neural Network (RNN)

수정날짜: 2021년 7월 28일 오후 6:14
작성날짜: 2021년 7월 26일 오후 10:41

# 1. 순환 신경망

앞서 배웠던 신경망들은 모두 피드 포워드 신경망이었다. (은닉층에서 활성화 함수를 지난 값들은 오직 출력층 방향으로만 향했다.) 하지만 RNN은 은닉층에서 활성화 함수를 지나 나온 값들을 출력층 방향으로도 보내면서 은닉층의 다음 계산의 입력으로 보낸다.

# 2. 메모리 셀 (RNN Cell)

은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드를 셀이라고 한다. RNN의 셀은 이전의 값을 기억하려고 하는 경향이 있으므로 메모리 셀, 또는 RNN 셀이라고 한다.

현재 시점을 t라고 할 때, 셀은 다음 시점은 t+1 시점의 자신에게 계산된 값을 보내는데 이 값을 `은닉 상태`라고 한다. 즉 t 시점의 메모리 셀은 t-1 시점의 메모리 셀이 보낸 은닉 상태를 t 시점의 은닉 상태를 계산하기 위한 값으로 이미 사용한 것이다.

이를 뉴런 단위로 표현하면 아래 그림과 같다.

![Recurrent%20Neural%20Network%20(RNN)%207a7c9f370f254b30bfb3ce013a415e13/Untitled.png](Recurrent%20Neural%20Network%20(RNN)%207a7c9f370f254b30bfb3ce013a415e13/Untitled.png)

# 3. 아키텍처

![Recurrent%20Neural%20Network%20(RNN)%207a7c9f370f254b30bfb3ce013a415e13/Untitled%201.png](Recurrent%20Neural%20Network%20(RNN)%207a7c9f370f254b30bfb3ce013a415e13/Untitled%201.png)

RNN의 기본적인 아키텍처이다. 중요한 것은 RNN 아키텍처에서는 뉴런 단위가 잘 쓰이지 않는다는 것이다. 따라서 위의 그림에서 각 사각형은 뉴런이 아니라 벡터를 의미한다. 또한 왼쪽에서 오른쪽으로 진행될수록 시간도 같이 진행된다는 사실을 기억해야한다.

![Recurrent%20Neural%20Network%20(RNN)%207a7c9f370f254b30bfb3ce013a415e13/Untitled%202.png](Recurrent%20Neural%20Network%20(RNN)%207a7c9f370f254b30bfb3ce013a415e13/Untitled%202.png)

$입력벡터  \space x_t에 \space 대한 \space 가중치 \space  W_x \space 와 \space 은닉상태 \space h_{t-1}에 \space 대한  \space 가중치 \space W_h에 \space 대해 \space h_t=tanh(W_xx_t+W_hh_{t-1}+b) \space 이고  \space y_t=f(W_yh_t+b) \space 이다. \space (f는 \space 비선형 \space 활성화 \space 함수)$

$h_t$의 계산을 도식화하면 다음과 같다.

![Recurrent%20Neural%20Network%20(RNN)%207a7c9f370f254b30bfb3ce013a415e13/Untitled%203.png](Recurrent%20Neural%20Network%20(RNN)%207a7c9f370f254b30bfb3ce013a415e13/Untitled%203.png)

은닉상태를 계산하기 위한 활성화 함수로는 보통 tanh가 쓰인다.

# 4. 파이토치에서 구현

넘파이를 이용해 셀이 정확히 어떻게 작동하는지 보자.

```python
import numpy as np

timesteps = 10 # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_size = 4 # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_size)) # 입력에 해당되는 2D 텐서

hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태는 0(벡터)로 초기화
# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.
```

```python
Wx = np.random.random((hidden_size, input_size))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wh = np.random.random((hidden_size, hidden_size)) # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
b = np.random.random((hidden_size,)) # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).
```

```python
total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs: # 각 시점에 따라서 입력값이 입력됨.
  output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh,hidden_state_t) + b) # Wx * Xt + Wh * Ht-1 + b(bias)
  total_hidden_states.append(list(output_t)) # 각 시점의 은닉 상태의 값을 계속해서 축적
  print(np.shape(total_hidden_states)) # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
  hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis = 0) 
# 출력 시 값을 깔끔하게 해준다.

print(total_hidden_states) # (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.
```

inputs의 크기는 10x4이다. 따라서 위의 for문은 10번 반복되며 10개의 input 시점에 따라 output을 계산한다.

이제 파이토치가 제공하는 nn.RNN()을 이용해 RNN을 구현해보자.

```python
import torch
import torch.nn as nn

input_size = 5 # 입력의 크기
hidden_size = 8 # 은닉 상태의 크기

# (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)
```

```python
cell = nn.RNN(input_size, hidden_size, batch_first=True)
```

nn.RNN()을 이용해 입력의 크기, 은닉 상태의 크기를 설정해주고 batch_first=True로 설정해서 입력 텐서의 첫번째 차원이 배치 크기임을 알려준다.

```python
outputs, _status = cell(inputs)
print(outputs.shape) # 모든 time-step의 hidden_state
print(_status.shape) # 최종 time-step의 hidden_state
```

```
torch.Size([1, 10, 8])
torch.Size([1, 1, 8])
```

RNN셀은 두 개의 값을 리턴하는데 하나는 모든 timesteps에서의 은닉 상태들이고 하나는 마지막 timestep의 은닉 상태이다. 시점이 10개이고 은닉 상태가 8차원이므로 첫번째 값의 크기는 1x10x8이며, 두번째 값의 크기는 1x1x8이다.