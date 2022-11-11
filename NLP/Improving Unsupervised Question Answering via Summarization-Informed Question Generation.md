# Improving Unsupervised Question Answering via Summarization-Informed Question Generation

Question Generation과정은 Passage(Context)와 Answer를 제공하여 Question을 만드는 과정  
여기에는 두가지 큰 줄기가 존재한다.

> Template-based QG  

미리 결정된 규칙을 가지고 선언문을 의문문으로 만든다.  
생성된 질문이 원 문장과 매우 비슷함

> Supervised QG

Question Answering 데이터셋을 사용하여 시스템을 학습시켜 Question을 생성한다.  
학습에 사용된 도메인이나 QA 데이터셋에 제한사항이 존재한다

이러한 문제점을 극복하기 위하여, 저자는 요약(Summary)에서 생성한 질문을 QG 시스템을 학습하기 위한 데이터로 사용하였다.  
요약데이터는 뉴스 요약 데이터를 사용했음, NER과 SRL(Semantic Role Labeling)을 사용하여 구문분석  
이렇게 학습한 QG모델을 QA모델과 쌍을 이루게 된다

## Introduction

QG의 목표는 주어진 Passage와 Answer를 가지고 의미있는 질문을 만들어내는것  
추가적으로, QG는 Data Augmentation의 목적으로 QA 과정에 적용될 수 있음  
마지막 목표는 unsupervised Question Answering 과정, QG의 출력만으로 QA모델을 학습한다.  

Seq2Seq model은 매우 효율적이지만, 데이터셋이 많이 필요함

저자가 제안한 방식 : QG과정을 summarization-questioning 과정으로 분할함

