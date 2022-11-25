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
> 복잡한 Graph 데이터들을 효과적으로 모델링 할 수 있어야 한다.

이 논문에서는 Graph Neural Network를 자연어처리에 접목하는 방법에 대한 개요를 설명하고 있음  
Automatic Graph Construction for NLP  
Graph representation learning for NLP  
GNN-based encoder-decoder model for NLP  
> Graph2Seq, Graph2Tree, Graph2Graph  

etc...  

논문에서는 GNN을 4종류로 구분함  
> Graph Construction  
> Graph Representation Learning  
> Graph-based Encoder-Decoder model  
> *3개로 되어있는데 Figure 1 표에서는 Addressing Tasks라는 구분을 추가적으로 제공하고 있음*  

포괄적인 구분을 제시하였으며 domain knowledge/semantic space/기타등등의 그래프 구성 접근방법에 대한 자세한 설명과 비교를 제공하고 있음  
> 그래프 구성, 그래프 표현학습 및 임베딩 초기화의 3가지 핵심 구성 요소를 따라 NLP작업을 처리하는 방법을 제공함
> > Benchmark dataset, evaluation metrics, open-source code

추가적으로 NLP에 GNN을 최대한 활용하기 위한 다양한 미해결 과제들을 대략적으로 설명한다.  
아직 연구되지 않은 방향에 대한 토론과 제안을 제공한다.  

Section 2는 Graph의 관점에서 NLP problem을 해결하기 위한 전통적인 Graph-Based method를 소개한다  
Section 3은 현대 신경망에서 사용하고 있는 Graph 구조인 Graph Neural Network에 대한 기본 지식과 방법을 설명한다  
Section 4에서는 Graph sturctured Input을 사용하는 방법인 Dymanic/Static Graph Constuction을 소개한다  
Section 5에서는 Graph-Representation Learning Technique을 소개한다  
Section 6에서는 Seq2Seq와 Graph-Based Encoder-Decoder model인 Graph2Tree/Graph2Graph Model을 소개한다  
Section 7에서는 GNN을 사용하는 12개의 NLP Program을 설명한다  
Section 8에서는 NLP에 대한 남아있는 GNN의 과제들과 향후 연구 방향을 소개한다  
Section 9는 논문을 요약해준다  

> Figure 1 이미지

## Graph-Based Algorithms for NLP

NLP Problem들을 Graph의 관점에서 소개하고, 몇가지 대표적인 Graph-based method를 소개함  

### Natural Language Precessing: A Graph Perspective

자연어를 표현하는 방식은 그것에 대한 우리의 특별한 관점을 반영하고, 우리가 그것을 처리하고 이해하는 방식에 근본적인 영향을 미친다 *번역기 그대로 가져옴*  
일반적으로 자연어를 표현하는 방식에는 3가지가 있다  

가장 단순한 방법은 자연어를 BoT(Bag-of-Tokens)로 표현하는것이다  
이러한 가장 대표적인 NLP 기법은 주제 모델링(Topic Modeling)[Blei et al., 2003]이며 각 입력 텍스트를 Mixture of Topic으로 모델링하는것을 목표로 한다  
각 Topic은 Mixture of Word로 추가적으로 모델링될 수 있다  

조금 더 자연스러운 방법은 단어를 Sequence of Token으로 표현하는 것이다  
보편적으로 사람이 자연어를 말하고 쓰는 방법이다  
이러한 관점으로 자연어를 보게 되면 텍스트의 정보를 풍부하게 Capture할 수 있다  
> 예를 들자면, 두 토큰이 연속적인 local context에서 쌍으로 몇번 등장하는지에 대한 정보를 Capture할 수 있다  

이러한 방법을 가지는 대표적인 NLP는 CRF와 Word2Vec 등이 존재한다  

세번째 방법은 자연어를 그래프로 표헌하는 것이다  
NLP 어디에나 그래프는 존재한다  
Text는 Sequential Data로 간주하는게 맞지만, NLP에서는 오랜 시간동안 Text를 다양한 종류의 그래프로 표현하였다  
일반적인 그래프 표현에서는 Dependency Graph, Constituency Graph, AMR Graph, IE Graph, Lexical Network(어휘 네트워크), Knowledge Graph 등이 존재한다  
추가적으로 Document, Passage, Sentence, Word같은 요소의 여러 계층을 포함하는 Text Graph를 구성할 수도 있다  

위에 두 관점 대비 Graph로 표현하는것은 Text요소들 사이의 관계를 더 많이 포착할 수 있다  
다음 Section에서 소개하는것처럼, Graph구조는 단어 의미 모호성, 이름 모호성, 공동 참조 해결(Co-reference resolution), 감정 분석, 텍스트 클러스터링같은 많은 Graph-based method(ex: random walk, label propagation)이 NLP Problem에서 성공적으로 적용되었다  

### Graph-Based Methods for Natural Language Processing

여기서는 성공적으로 NLP에 적용된 다양한 Classical Graph-based algorithm을 소개한다  
몇가지 대표적인 Graph-based algorithm을 소개하고, 이를 NLP와 연결하는 방법에 대하여 자세히 설명한다.  
적용 범위는 (Mihalcea and Radev, 2011)을 참조한다  

#### Random Walk Algorithms


**Approach** Random Walk는 Graph-based Algorithm의 일종으로 그래프에서 랜덤 경로를 생성한다  
그래프의 임의 노드에서 시작할 수 있으며, 특정 전환 확률에 따라 매번 랜덤한 인접 노드를 반복적으로 방분하도록 선책할 수 있다  
Random Walk에서 방문한 모든 노드는 고정된 경로를 형성하게 된다  
Random Walk가 수렴된 이후, 그래프에 있는 모든 노드에 대한 고정분포를 얻을 수 있으며, 이 분포는 Probability Score를 계산하여 구조적 중요성이 높은 그래프에서 가장 두드러진 노드를 선택하거나 두 Random Walk 분포 사이의 유사성을 계산하여 두 그래프의 관련성을 측정하는데 사용할 수 있다  

**Applications** Random Walk 알고리즘은 다양한 NLP에 적용되었다  
> Measure Semantic Similarity of texts(Ramage et al., 2009)  
> Semantic distance on Semantic Networks(Hughes and Ramage, 2007)  
> Word-Sense Disambiguation(Mihalcea, 2005; Tarau et al., 2005)  
> Name Disambiguation(Minkov et al., 2006)  
> Query Expansion(Collins-Thompson and Callan, 2005)  
> Keyword Extraction(Mihalcea and Tarau, 2004)  
> Cross-language information retrieval(Monz and Dorr, 2005)

Hughes and Ramage는 Random Walk Algorithm을 사용하여 단어별 고정분포를 계산하고, 주어진 단어 쌍의 각각의 단어에 편향된 그래프의 Random Walk Distribution 사이의 유사성으로 두 단어 사이의 거리를 측정하였다  
Minkov는 이메일 말뭉치에서 이메일 특정 항목(ex:발신자, 수신자, 제목)의 그래프를 구축하고 Random Walk가 주어진 노드에서 멈출 확률을 설명하는 "lazy" topic-sensitive Random Walk Algorithm을 제안하였다  
이메일 그래프와 모호한 이름이 주어지면 메일 텍스트에 편향된 Random Walk를 수행하고, 수렴 후 나오는 고정분포에서 점수가 가장 높은 사람 노드를 선택하여 이름을 올바른 참조로 변경한다  
Mihalcea and Tarau는 키워드 추출 과제를 해결하기 위하여 Co-occurrence Graph에 Random Walk를 수행하였고, 이를 통해 나온 고정분포에서의 확률 점수를 기반으로 텍스트에서 단어의 Probability Score를 계산하였다  

#### Graph Clustering Algorithm

**Approach** 일반적인 Graph Clustering Algorithm에는 Spectral Clustering, Random Walk Clustering, Min-Cut Clustering이 존재한다  
Spectral Clustering Algorithm은 K-means같은 기존 알고리즘을 사용하여 클러스터링을 수행하기 이전에 그래프의 라플라시안 행렬의 스펙트럼(고윳값)을 사용하여 차원을 축소시킨다  
Random Walk Algorithm은 그래프에서 T-step Random Walk를 사용하여 작동하며, 각 노드는 그래프의 모든 다른 노드에 대한 T-step 생성 확률을 나타내는 확률 벡터로 표현된다  
모든 Clustering Algorithm은 Generation-link Vector에 적용 가능하다  
Graph Clustering은 Global Structural Information(Random Walk 수렴 후 고정 분포로 인코딩함)대신 Local Structural Information을 캡쳐하는데에 더 관심이 있으므로 작은 T값이 선호된다  
Min-Cut Algorithm을 사용하여 그래프를 클러스터로 분할할수도 있다  

**Applications** Erkan(2006)은 n개의 문서 노드가 포함된 지시 생성 그래프의 T단계 Random Walk에서 파생된 n-dim 확률 분포를 말뭉치의 각 문서의 벡터 표현으로 사용할 것을 제안하였다  
이러한 문서 표현은 Graph Clustering Algorithm에 의하여 Document Cluster를 생성할 수 있었다  
Ponte and Croft(1998)이 제안한 언어 모델 접근법에 따라 Corpus에서 순서가 지정된 각 문서 쌍의 생성 확률을 계산하여 Generation Graph를 구성하였다  

#### Graph Matching Algorithm



