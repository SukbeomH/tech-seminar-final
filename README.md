# tech-seminar-final

Woori FIS Academy Tech Seminar Final

# TEXTGRAD

[Introduction to TextGrad, Code AI](https://youtu.be/Qks4UEsRwl0)
[Github Repository](https://github.com/zou-group/textgrad)
[HuggingFace](https://huggingface.co/TextGrad)
[Paper](https://arxiv.org/abs/2406.07496)

> Automatic "Differentiation" via Text

TextGrad는 LLM에서 제공하는 텍스트 피드백을 통해, 역전파를 구현하여 텍스트 데이터에 대한 미분을 수행할 수 있습니다.

## BackPropagation

각 뉴런들은 이전의 모든 뉴런들의 결과라는 점을 통해, 출력에서 부터 거꾸로 입력까지 오차를 전파시키는 방법입니다.
이를 통해 각 뉴런의 가중치와 편향치를 조정할 수 있습니다.

## TextGrad

그렇다면 LLM의 결과값을 피드백으로 사용하여 역전파를 구현할 수 있을까?
→ 더 큰 수준에서의 미분을 수행할 수 있을까?

### Automatic “Differentiation” via Text

이미 많은 성과들이 LLM을 통해 이루어졌습니다.
올림피아드 수준의 문제를 풀거나, 코드를 생성하는 등의 작업들, 분자 구조를 예측하는 등의 작업들이 가능해졌습니다.
하지만 이러한 성과는 모두 전문가의 직관과 휴리스틱한 방법을 통해 이루어진 것입니다.

TextGrad는 이러한 전문가의 직관과 휴리스틱한 방법을 대체하고, 자동화하여 LLM의 수준을 끌어올리는 프레임워크입니다.

#### 역전파와 TextGrad는 어떻게 다른가?

역전파는 모델 자체의 성능을 향상시키는 방법입니다.

- 모델의 가중치와 편향치를 조정하여 모델의 성능을 향상시킵니다.
- 모델의 성능을 향상시키기 위해, 모델의 출력값을 최적화합니다.

반면 TextGrad는 모델을 포함한 시스템을 최적화하는 방법입니다.

> Here we use differentiation and gradients as a metaphor for textual feedback from LLMs.

기존 결과값을 바탕으로 LLM은 텍스트 피드백을 제공하고, 이를 통해 TextGrad는 모델을 최적화합니다.
미분과 그레디언트는 이 과정을 비유하기 위한 요소입니다.

### TextGrad, 필수 요소

> Our framework is built on the assumption
> that the current state-of-the-art LLMs are able to reason
> about individual components and subtasks of the system
> that it tries to optimize.

LLM은 시스템의 개별 구성요소와 하위 작업에 대해 추론할 수 있어야 합니다.
이러한 가정을 바탕으로 TextGrad는 시스템을 최적화합니다.

### 아이디어

> Prediction = LLM _ (**"Prompt"** + Question)
> Evaluation = LLM _ (Evaluation Instruction + Prediction)

> "Prompt + Question" **--LLM-→** "Evaluation Instruction + Prediction" **--LLM-→** "Evaluation"
