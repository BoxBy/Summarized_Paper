# 2017

## Learning To Ask : Neural question generation for reading comprehension [(링크)](https://www.aclweb.org/anthology/P17-1123.pdf)

[(github)](https://github.com/xinyadu/nqg)

특이하게 Lua로 짜여져 있음

### 모델

RNN Encoder-Decoder + Attention 구조 (Attention만 사용한 구조가 아님)

두종류의 모델 variation이 존재함(encoder 부분의 변형)
> sentence encode only
> 
> sentence + paragraph encode

#### Decoder

LSTM 사용

sentence only model
> sentence Encoder만 존재

sentence + paragraph model
> sentence Encoder, paragraph Encoder 따로 있음

> 각각의 Encoder를 통과한 hidden state를 Concatenate해서 사용합니다

> Y shaped Network라고 설명하고있음

#### Encoder

Bi-Directional LSTM with Attention

##### Paragraph Encoder

truncation

#### Training & Inference

Dataset
> Sentence(Sequence) - Question Pair

beam search를 사용하여 inference

target side dictionary에 존재하지 않는 단어들은 decoding시에 UNK Token으로 나오게 된다

이를 highest attention score를 가진 Token으로 replacing해준다
> attention score를 계산한 후, argmax를 취하여 Token을 구함

### Experimental Setup

#### Dataset

SQuAD dataset

<SOS>, <EOS> Token으로 Start/End Of Sentence를 구분
