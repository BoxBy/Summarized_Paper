# Attention is All You Need[(링크)](https://arxiv.org/pdf/1706.03762.pdf)

Seq2Seq + Attention 구조[(링크1)](https://arxiv.org/abs/1409.0473)[(링크2)](https://arxiv.org/abs/1409.0473)는 Seq2Se2의 한계점을 개선해주었지만, 근본적인 해결책이 되지는 못했음

RNN은 기본적으로 순차적인 연산을 진행하므로, 병렬연산을 사용할 수 없어 필연적으로 병목이 발생한다.

이를 Self-Attention 구조를 사용하여 해결하였음

## Self-Attention

자기 자신을 주목하는 Attention

Attention에서 가중치 행렬(W)가 추가되었고, 이를 총하여 Q(Query), k(key), V(Value) 벡터를 얻어 연산을 진행한다

> 일일히 벡터 연산을 진행할 필요 없이 행렬로 한번에 연산할 수 있다.

> 실제로도 행렬로 구현이 되어있으며, 논문에 기재된 수식과도 일치한다.

Transformers 모델에서는 더욱 확장된 Multi-Head Attention을 사용한다.

> 말 그대로 여러개의 Head를 가지고 있는 Attention

> n차원의 단어벡터를 n/num_head차원으로 변형하여 각각 Attention을 통과시켜 연결하고, 가중치 행렬(W)를 곱한다

> 의도적으로 input/output의 Size를 동일하게 만들어 놓았기 때문에, 층을 더 쌓아 깊은 모델을 만들거나 Skip Connection을 사용하기 용이하다

## Encoder

(Shortcut - Multi-Head Attention - Add - LayerNorm)

(Shortcut - {Dense - ReLU - Dense}(Feed Forward Layer) - Add - LayerNorm)

이 두 레이어를 N번 쌓은 간단한 구조

## Decoder

Encoder에서 사용하는 레이어에 추가적으로 Masked Multi-Head Attention이 추가됨

미래의 값을 볼 수 없도록 Masking 전처리 과정이 추가된 Attention

(Shortcut - Multi-Head Attention - Add - LayerNorm)

이후 Multi-Head Attention Layer에서는 Encoder에서 Key와 Value를 받아오고, Multi-Head Attention에서 넘겨준 데이터를 Query로 사용하여 연산한다.

## 성능

한줄요약 - 가볍고 빠르면서 더 정확하다

논문에서는 Traning Cost를 통하여 비교해주고 있는데, GNMT[(링크)](https://arxiv.org/pdf/1609.08144.pdf) 대비 1/50의 연산량만을 가지며 정확도도 더 높다

> GMNT의 연산량 1.4+20e, Transformer의 연산량 3.3+18e
