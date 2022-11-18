# Graph Neural Networks for Natural Language Processing: A Survey[(링크)](https://arxiv.org/pdf/2106.06090.pdf)

Graph4NLP 논문  
130페이지짜리 논문(...)

## Abstract

현 NLP는 Token을 이용하여 text input을 넣어주는 방식으로 진행되는데, Graph의 구조를 이용하여 이를 잘 표현할 수 있는 NLP의 문제들이 존재한다.  
NLP에 사용되는 Graph Neural Networks를 설명하는 논문
> Graph 구조  
> Graph 표현 학습방법  
> Graph 기반 Encoder-Decoder Model

> Keyword : Graph Neural Networks, Natural Language Processing, Deep Learning on Graphs

## Introduction

대규모 말뭉치에서 작동하는 요즘의 NLP들은 일반적으로 Text Sequence를 BoW(Bag-of-Words)나 TF-IDF같은 토큰으로 간주한다.  
Word Embedding 테크닉으로 인하여, 문장들은 NLP에서 Token Sequence로 간주된다.  

하지만, Graph구조를 활용하는것이 자연어처리 문제를 더 잘 풀 수 있는 방법이 될 수 있다.  
> 문장의 구조를 보는것이 문장의 의미를 더 잘 파악할 수 있는 방법이 될 수 있음
> > Knowledge Graph를 구축하는것과 비슷하다고 볼 수 있지만, KG는 DL이 아니다  

비슷하게, Sequence data의 의미 정보는 Original Sequence의 데이터를 향상시키는데에 사용될수도 있다  
그래서 Graph구조로 이루어진 데이터들은 서로간의 관계정보를 더 많이 학습할 수 있다  

다만 현재 딥러닝에 사용되는 데이터에 사용하는 Technique들은 그래프 구조로 바로 변경할 수 없음  
> 이미지는 Euclidean Data, Text는 Sequence Data  
> Graph Data는 다른 데이터들과 달리 구조가 불규칙적이며 Node Neighbor의 크기가 다양함

이러한 이유로 인하여 Graph Neural Network라는 구조의 개발에 대한 흐름이 나타나게 되었다  
> 현재 존재하는 구조로는 이것을 써먹을 수 없으니, 새로운 구조를 만들어 써먹게 하자!  

Graph와 NLP의 결합은 많은 NLP task에서 성공을 거두었었음(ex: ERNIE, Knowledge Graph 구조를 활용하였음)  
이로 인하여 다양한 종류의 GNN들을 많은 NLP Task에 적용하는것에 대한 관심이 증가하였고, 괜찮은 성공을 거두었음  
> Classification Tack(sentence Classification)  
> Semantic role labeling  
> Relation Extraction  
> Machine Translation  
> Question Generation  
> Summarization  

이러한 Task에서 성공을 거두었지만, 많은 문제들이 존재한다.
> 자동으로 원본 Text Sequence Data를 Highly Graph-Structured Data로 변환해야 한다.  
> Graph Representation Learning Technique을 설정해야한다.  
> > Graph구조



