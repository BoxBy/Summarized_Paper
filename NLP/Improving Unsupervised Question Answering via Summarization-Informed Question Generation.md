# Improving Unsupervised Question Answering via Summarization-Informed Question Generation[(링크)](https://arxiv.org/pdf/2109.07954.pdf)

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
dependency parsing 적용 : 기존에는 *original news article*을 input을 넣고 generated question을 받앗지만,  
현 모델은 parsed summaries에서 question을 생성한다

> 1번 이미지(나중에 추가할것)

news summary data로 QG모델 학습, 이 모델을 사용하여 QA모델의 비지도학습을 준비  
test를 위하여 SQuAD, Natural Questions, TriviaQA, NewsQA, BioASQ, DuoRC 데이터셋 사용

이러한 작업은 두개로 요약될 수 있음

> 1. Summary Data와 Syntatic/Semantic 분석을 사용하는 Unsupervised QG  
> Text Summarization과 Question Generation 과정을 연결한 첫번째 논문

> 2. QG 모델을 사용하여 인공 데이터를 생성한다.  
> 인공적으로 만든 학습 데이터의 양이 적어도 SOTA를 달성할 수 있다

## Related Work

### Question Generation

QG모델의 기원과 발전과정, RL을 접목한 논문, BLEU 스코어 등장, 

### Question Generation Evaluation

BLEU, ROUGE, Meteor metrics, .....

### Unsupervised QA

비지도 QA에서는 QG모델에서 기반된 인공 데이터를 사용하여 학습을 진행한다.  
NMT(Neural Machine Translation)같은 방법이 사용되었음  

## Methodology

논문에서 제안하는 방식은  
요약 데이터에서 휴리스틱을 사용하여 생성한 인공적인 QG 데이터를 사용해서 QG모델을 학습함  

> 2번 이미지(나중에 추가할것)  
> Context에서 Summary를 만들고, Summary를 Question Generation heuristics를 통과하여 Question과 정답을 만들어낸다. 
> Context + Answer를 Encoder에 넣어 Question을 생성하고 이를 Heuristics에서 만들어낸 Question과 비교하여 loss를 계산한다.

### Question Generation

사소한 질문들이 생성되는것을 방지하기 위하여 Summary를 Bridge로 사용하여 Generated Question과 Context를 연결한다  
이를 위하여 Dependency Parsing, Named Entity Recognition, Semantic Role Labeling을 사용한다.  
> Named Entity Recognition(NER) : 개체명 인식  
> 가장 적절한 질문을 생성하기 위하여 모든 Entity에 대하여 Tagging 진행
> Semantic Role Labeling(SRL) : 의미역 결정, 자연어 문장의 서술어와 그 서술어에 속하는 논항들 사이의 의미관계를 결정  
> 핵심 분석 요소, Frame단위로 모든 의미를 얻기 위하여 사용
> Parsing : 각 문장의 문법적인 구성 또는 구문을 분석하는 과정(구문분석 트리를 구성)  
> > Dependency Parsing(DP) : 단어간의 의존관계를 파악하여 구조를 분석하는 방법  
> > Main Verb(Root Verb)를 인식하기 위하여 사용


> 알고리즘 1 sudocode(나중에 추가할것)

Question Generation Heuristics에 대한 알고리즘  
> 1. 모든 Dependency Edge와 Label obtain(DPs)  
> 요약 문장에서 NER tag(NERs), SRL Frame(SRL_frames) 구성  
> 2. Frame에 존재하는 모든 Root Verb에 대하여 iteration을 수행
> 3. 적합한 Wh-Words, base_verb, auxiliary word(aux), Quesiton Argument(Q_arg)를 찾아 Question을 생성

end-to-end Seq2Seq Model를 사용하여 학습  
input 형태 : Context <SEP> Answer <SEP>  
target 형태 : Question  
NLL Loss를 사용하여 optimize됨
  
## Experiments 
  
본 논문은 BART-base 모델을 사용하였음  
AdamW optimizer, 3e-5 learning rate, 3epoch로 학습 진행  

### Unsupervised QA

#### Datasets
  
SQuAD 1.1, NewsQA, National Questions, TriviaQA, BioASQ, DuoRC
  
> Table 1 이미지(나중에 추가할것)
  
지도학습보다는 낮지만(Transformers 기반이 아닌 모델보다는 높은 성능을 기록중), 비지도학습 QA에서는 SOTA 달성  
비교에 사용한 Supervised Model은 BERT-base/large이며 1e-5의 lr로 학습
  
### Results
  
> Table 2 이미지(나중에 추가할것)
  
꾸준히 20정도희 EM과 F1 score의 차이가 나지만, 비지도학습에서는 SOTA
  
## Analysis
  
### Effect of Answer Extraction
  
논문에서 제안한 Unsupervised QA System은 생각보다 고려할만하지 않음  
다만, NER 등을 기반으로 한 자동 답변 추출 방법은 모델의 개선 여지가 존재한다.  
사람이 추출한 Answer와 NER로 추출한 Answer에는 간극이 분명히 존재하며(NER의 성능이 낮음), 이런 경우에는 NER이 (잘못된)Answer를 추출할 수 있다.  
> Table 5 이미지(나중에 추가할것)  

Answer Extraction은 인공 QA 데이터셋의 품질에 생각보다 더 큰 영향을 주게된다.  
> 잘못된 Answer를 추출할 수 있는 NER을 Human으로 바꿔주는것 만으로 F1-score가 5~10point 증가한다  
  
### Effect of Different Heuristics
  
인공 데이터를 사용하면 성능이 증가합니다  
  
### Effect of the Size of Synthetic QA Data
  
> Figure 3 이미지(나중에 추가할것)
  
인공 데이터의 양이 늘어날수록 성능이 감소합니다  
20k보다 낮은 데이터의 양이 권장
  
### Few-Shot Learning

> Figure 4 이미지(나중에 추가할것)
  
Few-Shot에서는 매우 효율적
  
### QG Error Analysis
  
50개를 랜덤을 뽑아서 비교해봄  
> 62%(31개) high quality  
> 12%(6개) wh-word와 answer mismatch  
> 8%(4개) missing information needed to locate the answer  
> 10%(5개)factual error
> 8%(4개) grammatical error
  
## Conclusion

이 논문은 QA 데이터셋을 사용하지 않고 Question을 생성하는 방법에 대한 연구  
이전방법보다 더 잘 작동하고, 이 방법으로 생성한 질문을 Unsupervised Learning에 접목할 수 있음
