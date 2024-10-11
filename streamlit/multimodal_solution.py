import streamlit as st
import textgrad as tg
from textgrad import Variable, autograd
from textgrad.loss import TextLoss, MultiFieldEvaluation
from dotenv import load_dotenv

load_dotenv()


## Set the Streamlit Page Configurations
st.set_page_config(
    page_title="Prompt Optimization with TextGrad",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

## Add a title
st.title("Prompt Optimization with TextGrad")

## Add tabs for different methods
multi_modal_tab, solution_optimization_tab = st.tabs(["MultiModal", "Solution Optimization"])

## MultiModal Tab
with multi_modal_tab:
    ## Get the engine, in this case we are using gpt-4o
    engine = tg.set_backward_engine("gpt-4o", override=True)
    st.markdown("# MultiModal: 이미지 설명")
    ## Add a text input
    question = st.text_input("질문을 입력해주세요.", "이 사진에서 무엇을 볼 수 있나요?", on_change=True)

    ## Add upload image section
    image = st.file_uploader("이미지를 업로드해주세요.", type=["jpg", "jpeg", "png"])
    ## image to variable
    image_variable = None
    if image is not None:
        image_data = image.getvalue()
        image_variable = Variable(image_data, role_description="image to answer a question about", requires_grad=False)
        ## Display the image
        st.image(image_data, use_column_width=True)

    ## Add a button
    response = None
    question_variable = None
    if st.button("결과 확인"):
        if image is None:
            st.error("이미지를 업로드해주세요.")
        else:
            with st.container():
                ## Create a Variable object for TextGrad
                question_variable = Variable(question, role_description="question to the LLM", requires_grad=False)
                ## Call the LLM
                response = autograd.MultimodalLLMCall(engine)([image_variable, question_variable])
                ## Display the response, Large size fonts
                st.markdown(f"## {response}")
            

    with st.container():
        ## Check if response is available
        if response is not None:
            ## Criticize the Response with TextGrad
            evaluation_instruction = Variable("""
                    Does this seem like a complete and good answer for the image? 
                    Criticize. Do not provide a new answer. write in korean.
                    """,
                    requires_grad=False,
                    role_description="evaluation instruction"
            )
            response_evaluator = TextLoss(evaluation_instruction, engine)
            evaluated = response_evaluator(response)
            
            ## Display the loss
            if evaluated is not None:
                st.markdown(f"# 내용 평가: \n ## {evaluated}")
                
            ## Create the Optimized response
            optimizer = tg.TGD(parameters=[response])
            evaluated.backward()
            optimizer.step()
            
            ## Display the optimized response
            st.markdown(f"# 최적화된 답변: \n ## {response.value}")

## Solution Optimization Tab
with solution_optimization_tab:
    ## Get the engine
    engine = tg.set_backward_engine("gpt-3.5-turbo", override=True)
    ## Add a title
    st.markdown("# Solution Optimization")
    
    ## Get image from file path
    
    ## Add a text input
    question = st.text_input(
        "질문을 입력해주세요.", 
        """
        최고차항의 계수가 1인 삼차함수 f(z)와 실수 전체의 집합에서 연속인 함수 g(2)가 아래 조건을 만족시킬 때, f (4) 의 값을 구하시오.
        [
            (가) 모든 실수 2에 대하여 f(x)=f(1)+(x-1)f'(g(x))이다.
            (나) 함수 g(x)의 최솟값은 5/2 이다.
            (다) f(0)=-3, f(g(1))= 6
        ]
        """, 
        on_change=True
        )
    
    ## Add a button
    response = None
    question_variable = None
    if st.button("개선 이전 결과 확인"):
            with st.container():
                ## Create a Variable object for TextGrad
                question = question + "\n When write LaTeX expressions, by wrapping them in '$' or '$$' (the '$$' must be on their own lines). \n For example: $3x^2 - 7x + 2 = 0$ \n  return the solutions in markdown format."
                question_variable = Variable(question, role_description="question to the LLM", requires_grad=False)
                ## Call the LLM
                initial_solution = autograd.LLMCall(engine)(question_variable)
                ## Display the response, Large size fonts
                st.markdown(f"## 초기 답변: \n {initial_solution}")
                
                ## Proceed to evaluate the response
                if initial_solution is not None:
                    ## Create a Variable object for TextGrad
                    solution = Variable(
                        initial_solution.value,
                        requires_grad=True,
                        role_description="solution to the math question"
                        )
                    loss_system_prompt = Variable(
                        """
                        You will evaluate a solution to a math question.
                        Do not attempt to solve it yourself, 
                        do not give a solution, 
                        only identify errors. 
                        Be super concise.
                        """,
                        requires_grad=False,
                        role_description="system prompt"
                        )
                    
                    ## Calculate the loss function
                    response_evaluator = TextLoss(loss_system_prompt)
                    evaluated = response_evaluator(solution)
                    ## Display the loss
                    st.markdown(f"# 내용 평가: \n ## {evaluated}")
                    
                    
                    # optimizer = tg.TGD([solution])