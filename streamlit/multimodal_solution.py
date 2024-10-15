import streamlit as st
import textgrad as tg
from textgrad import Variable, autograd, TGD
from textgrad.loss import TextLoss
from dotenv import load_dotenv
import random
import time
from IPython.core.interactiveshell import InteractiveShell

load_dotenv()

def generate_random_test_case(size, min_value, max_value):
    return [random.randint(min_value, max_value) for _ in range(size)]

def run_function_in_interpreter(func_code):
    interpreter = InteractiveShell.instance()
    interpreter.run_cell(func_code, store_history=False, silent=True)
    func_name = func_code.split("def ")[1].split("(")[0].strip()
    func = interpreter.user_ns[func_name]

    return func


def test_longest_increasing_subsequence(fn):
    nums = [10, 22, 9, 33, 21, 50, 41, 60]
    assert fn(nums) == 5
    nums = [7, 2, 1, 3, 8, 4, 9, 6, 5]
    assert fn(nums) == 4
    nums = [5, 4, 3, 2, 1]
    assert fn(nums) == 1
    nums = [1, 2, 3, 4, 5]
    assert fn(nums) == 5
    nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    assert fn(nums) == 4
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    assert fn(nums) == 4
    nums = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    assert fn(nums) == 6
    nums = [7, 7, 7, 7, 7, 7, 7]
    assert fn(nums) == 1
    nums = [20, 25, 47, 35, 56, 68, 98, 101, 212, 301, 415, 500]
    assert fn(nums) == 11
    nums = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert fn(nums) == 1

    print("All test cases passed!")

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
    ## Set the engine
    engine = tg.get_engine("gpt-4o", override=True)
    # engine = tg.set_backward_engine("gpt-4o", override=True)
    
    ## Add a title
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
        st.image(image_data, use_column_width=False)

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
                ## aggregate the question and image
                aggregate = autograd.sum([question_variable, image_variable])
                ## Call the LLM
                response = autograd.LLMCall(engine=engine)([question_variable, image_variable])
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
            optimizer = TGD(parameters=[response])
            evaluated.backward()
            optimizer.step()
            
            ## Display the optimized response
            st.markdown(f"# 최적화된 답변: \n ## {response.value}")

## Solution Optimization Tab
with solution_optimization_tab:
    st.markdown("# Solution Optimization with TextGrad")
    
    problem_text = """Longest Increasing Subsequence (LIS)

    Problem Statement:
    Given a sequence of integers, find the length of the longest subsequence that is strictly increasing. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

    Input:
    The input consists of a list of integers representing the sequence.

    Output:
    The output should be an integer representing the length of the longest increasing subsequence."""
    st.markdown("## Problem Statement")
    st.markdown("## 문제 설명: \n 정수로 이루어진 수열이 주어졌을 때, 가장 긴 '엄격하게 증가하는 부분 수열'의 길이를 구하시오. \n (부분 수열이란, 원래 수열에서 일부 혹은 아무 요소도 삭제하지 않고, 나머지 요소들의 순서를 유지한 채 얻을 수 있는 수열을 말합니다.) \n")
    st.markdown("## 입력: \n 입력은 수열을 나타내는 정수 리스트로 주어집니다. \n ")
    st.markdown("## 출력: \n 출력은 가장 긴 증가하는 부분 수열의 길이를 나타내는 정수입니다.")
    
    initial_solution = """
    def longest_increasing_subsequence(nums):
        n = len(nums)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        max_length = max(dp)
        lis = []

        for i in range(n - 1, -1, -1):
            if dp[i] == max_length:
                lis.append(nums[i])
                max_length -= 1

        return len(lis[::-1])
    """
    st.markdown("## Initial Solution")
    st.code(initial_solution, language="python")

    # Test the function with a random test case
    size = 10000  # Adjust the size as needed
    min_value = 1
    max_value = 1000

    nums = generate_random_test_case(size, min_value, max_value)
    longest_increasing_subsequence = run_function_in_interpreter(initial_solution)

    start_time = time.time()
    lis = longest_increasing_subsequence(nums)
    end_time = time.time()

    print(f"Test Case Size: {size}")
    st.markdown(f"Test Case Size: {size}")
    
    print(f"Longest Increasing Subsequence Length: {lis}")
    st.markdown(f"Longest Increasing Subsequence Length: {lis}")
    
    print(f"Runtime: {end_time - start_time:.5f} seconds")
    st.markdown(f"Runtime: {end_time - start_time:.5f} seconds")
    
    # Test for all test cases
    test_longest_increasing_subsequence(longest_increasing_subsequence)
    
    ## Add a button to run optimization
    if st.button("Run Optimization"):
        llm_engine = tg.get_engine("gpt-4o", override=True)
        tg.set_backward_engine(llm_engine, override=True)

        # Code is the variable of interest we want to optimize -- so requires_grad=True
        code = tg.Variable(value=initial_solution,
                        requires_grad=True,
                        role_description="code instance to optimize")

        # We are not interested in optimizing the problem -- so requires_grad=False
        problem = tg.Variable(problem_text,
                            requires_grad=False,
                            role_description="the coding problem")

        # Let TGD know to update code!
        optimizer = tg.TGD(parameters=[code])
        
            # The system prompt that will guide the behavior of the loss function.
        loss_system_prompt = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise feedback."
        st.markdown(f"## System Prompt Set up: {loss_system_prompt}")
        loss_system_prompt = tg.Variable(loss_system_prompt, requires_grad=False, role_description="system prompt to the loss function")

        # The instruction that will be the prefix
        instruction = """Think about the problem and the code snippet. Does the code solve the problem? What is the runtime complexity?"""
        st.markdown(f"## Instruction Set up: {instruction}")
        
        # The format string and setting up the call
        format_string = "{instruction}\nProblem: {{problem}}\nCurrent Code: {{code}}"
        format_string = format_string.format(instruction=instruction)

        fields = {"problem": None, "code": None}
        formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_engine,
                                                        format_string=format_string,
                                                        fields=fields,
                                                        system_prompt=loss_system_prompt)

        # Finally, the loss function
        def loss_fn(problem: tg.Variable, code: tg.Variable) -> tg.Variable:
            inputs = {"problem": problem, "code": code}

            return formatted_llm_call(inputs=inputs,
                                    response_role_description=f"evaluation of the {code.get_role_description()}")

        # Let's do the forward pass for the loss function.
        loss = loss_fn(problem, code)
        print(loss.value)
        if loss is not None:
            st.markdown(f"## Loss Value: \n {loss.value}")
        
        # Let's visualize our computation graph.
        # loss.generate_graph()
        
        # Let's look at the gradients!
        loss.backward()
        print(code.gradients)
        st.markdown(f"## Gradients: \n {code.gradients}")
        
        # Let's update the code
        optimizer.step()
        
        # Hopefully, we should get much better runtime!
        longest_increasing_subsequence = run_function_in_interpreter(code.value)

        start_time = time.time()
        lis = longest_increasing_subsequence(nums)
        end_time = time.time()

        print(f"Longest Increasing Subsequence Length: {lis}")
        st.markdown(f"Longest Increasing Subsequence Length: {lis}")
        
        print(f"Runtime: {end_time - start_time:.5f} seconds")
        st.markdown(f"Runtime: {end_time - start_time:.5f} seconds")

        test_longest_increasing_subsequence(longest_increasing_subsequence)
        
        st.markdown("# Run Another Iteration")
        # Let's do one more iteration
        optimizer.zero_grad()
        loss = loss_fn(problem, code)
        loss.backward()
        optimizer.step()
        
        longest_increasing_subsequence = run_function_in_interpreter(code.value)

        start_time = time.time()
        lis = longest_increasing_subsequence(nums)
        end_time = time.time()

        print(f"Longest Increasing Subsequence Length: {lis}")
        st.markdown(f"Longest Increasing Subsequence Length: {lis}")
        
        print(f"Runtime: {end_time - start_time:.5f} seconds")
        st.markdown(f"Runtime: {end_time - start_time:.5f} seconds")

        test_longest_increasing_subsequence(longest_increasing_subsequence)
        
        print(code.value)
        st.code(code.value, language="python")