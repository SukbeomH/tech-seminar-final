# tech-seminar-final

Woori FIS Academy Tech Seminar Final

# TEXTGRAD

[Introduction to TextGrad, Code AI](https://youtu.be/Qks4UEsRwl0)
[Github Repository](https://github.com/zou-group/textgrad)
[HuggingFace](https://huggingface.co/TextGrad)
[Paper](https://arxiv.org/abs/2406.07496)

> Automatic "Differentiation" via Text

TextGrad는 LLM에서 제공하는 텍스트 피드백을 통해 역전파를 구현하여 텍스트 데이터에 대한 미분을 수행할 수 있습니다.

## BackPropagation

각 뉴런들은 이전의 모든 뉴런들의 결과라는 점을 통해, 출력에서 부터 거꾸로 입력까지 오차를 전파시키는 방법이다. 이를 통해 각 뉴런의 가중치와 편향치를 조정할 수 있다.

## TextGrad

그렇다면 LLM의 결과값을 피드백으로 사용하여 역전파를 구현할 수 있을까?
→ 텍스트 데이터에 대한 미분을 수행할 수 있을까?

결론적으로 가장 올바른 답을 찾아내는 모든 하이퍼파라미터를 찾아내고 조정하는 것이 가능할까?

### Automatic “Differentiation” via Text
