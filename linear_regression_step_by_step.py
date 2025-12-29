# streamlit run C:\Users\孙冰\Desktop\AI助教25-12-07\linear_regression_step_by_step.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time
from learning_report import generate_report_step
from datetime import datetime

st.set_page_config(layout="wide")

def check_quiz(answers, correct_answers):
    """检查当前步骤所有题目是否都答对"""
    for ans, correct in zip(answers, correct_answers):
        if ans != correct:
            return False
    return True

def record_error(step_num, question, user_answer, correct_answer):
    """记录错误信息"""
    error_info = {
        'question': question,
        'user_answer': user_answer,
        'correct_answer': correct_answer,
        'time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    # 更新步骤错误记录
    st.session_state.linear_step_records['step_records'][f'step_{step_num}']['error_count'] += 1
    st.session_state.linear_step_records['step_records'][f'step_{step_num}']['error_details'].append(error_info)
    # 更新总错误次数
    st.session_state.linear_step_records['total_errors'] += 1


def record_answer(step_num, question, user_answer, correct_answer, is_correct):
    """记录答题详情"""
    answer_info = {
        'question': question,
        'user_answer': user_answer,
        'correct_answer': correct_answer,
        'is_correct': is_correct,
        'time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.linear_step_records['step_records'][f'step_{step_num}']['answer_details'].append(answer_info)


def start_step_timer(step_num):
    """记录步骤开始时间"""
    if st.session_state.linear_step_records['step_records'][f'step_{step_num}']['start_time'] is None:
        st.session_state.linear_step_records['step_records'][f'step_{step_num}']['start_time'] = time.time()


def complete_step(step_num):
    """标记步骤完成并计算耗时"""
    st.session_state.linear_step_records['step_records'][f'step_{step_num}']['end_time'] = time.time()
    start_time = st.session_state.linear_step_records['step_records'][f'step_{step_num}']['start_time'] or time.time()
    st.session_state.linear_step_records['step_records'][f'step_{step_num}']['duration'] = round(
        st.session_state.linear_step_records['step_records'][f'step_{step_num}']['end_time'] - start_time, 2)
    st.session_state.linear_step_records['step_records'][f'step_{step_num}']['is_completed'] = True
    st.session_state.linear_step_records['current_step'] = step_num

# 步骤1：项目说明（场景化包装）
def step0():
    st.subheader("分析医疗数据并预测患者病情进展")
    st.info("""
    **你的角色：**
    你是数据分析师，接到一项重要任务：帮助医生通过患者的生理数据预测糖尿病病情进展。
   
    **任务背景：**
    医院收集了442名糖尿病患者的信息，包括：
    - 10项关键生理指标（年龄、体质指数、血压等）📋
    - 1年后的病情进展评分（数值越高表示病情越严重）📈
    
    **你的目标：**
    用线性回归模型从数据中找到规律，让医生能根据新患者的生理指标，提前预测病情发展。🔍
    
    **任务拆解：**
    你需要完成6个步骤，一步步搭建预测模型：
    1. 熟悉数据：看看手里有什么样的数据
    2. 整理数据：为建模做准备
    3. 搭建模型：选择合适的预测工具
    4. 训练模型：让模型从数据中学习规律
    5. 检验效果：看看模型预测得准不准
    6. 总结改进：分析结果并提出优化方向
    """)   
    
    # 加载数据集用于展示
    diabetes = load_diabetes()
    
    st.subheader("数据集预览")
    # 定义中文特征名列表
    chinese_feature_names = [
        "年龄", "性别", "体质指数", "平均血压", 
        "血清总胆固醇", "低密度脂蛋白", "高密度脂蛋白",
        "甲状腺素", "促甲状腺激素", "血糖"
    ]
    st.session_state.chinese_feature_names = chinese_feature_names
    df = pd.DataFrame(
        data=diabetes.data,
        columns=chinese_feature_names
    )
    df['疾病预测评分'] = diabetes.target

    st.dataframe(df.head(10), use_container_width=True)

    
    # 知识小测验部分
    st.subheader("📌 知识小测验")
    questions = [
        "T1. 在数据分析中，我们通常将用于预测的变量称为？",
        "T2. 糖尿病数据集属于哪种类型的数据？",
        "T3. 我们的最终目标是预测患者的什么指标？"
    ]
    options = [
        ["目标变量", "特征变量", "标签变量", "预测变量"],
        ["图像数据", "文本数据", "结构化数据", "时序数据"],
        ["年龄", "性别", "血压", "疾病预测评分"]
    ]
    correct_answers = ['特征变量', '结构化数据', '疾病预测评分']    
    q0_1 = st.radio(questions[0], options[0], key="q0_1", index=None)
    q0_2 = st.radio(questions[1], options[1], key="q0_2", index=None)
    q0_3 = st.radio(questions[2], options[2], key="q0_3", index=None)    
    current_answers = [q0_1, q0_2, q0_3]
    
    # 显示每个问题的即时反馈并记录答题情况
    for i, (q, ans, correct) in enumerate(zip(questions, current_answers, correct_answers)):
        if ans is not None:
            # 记录答题详情
            record_answer(1, q, ans, correct, ans == correct)
            
            if ans == correct:
                st.success(f"{i+1}. 回答正确")
            else:
                st.error(f"{i+1}. 回答错误，正确答案是：{correct}")
                # 记录错误信息
                record_error(1, q, ans, correct)
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：特征变量）",
        key="step0_reflection",
        autocomplete="off"
    )    

    if reflection:
        st.session_state.linear_step_records['reflection']['step_0'] = reflection
    
    # 下一步按钮
    all_answered = all(ans is not None for ans in current_answers)
    if all_answered and check_quiz(current_answers, correct_answers):
        st.info("太棒了！🎉 你已掌握基础概念，这是超棒的开始！准备好深入分析吧！🚀")
        if st.button("进入下一步：数据观察与理解", key="next_step0"):
            complete_step(1)  # 标记步骤1完成
            st.session_state.step += 1
            st.rerun()
    elif all_answered:
        st.warning("请先回答正确所有问题才能继续")
    else:
        st.info("请完成所有问题的回答")
        

# 步骤1：数据观察与理解
def step1():
    st.header("数据观察与理解")
    st.subheader("目标：加载糖尿病数据集，观察基本信息")
    st.info("""
    **任务说明**：  
    数据探索是建模分析的基础环节。需系统考察数据集规模（样本量与特征维度）、数据分布特征及关键统计量。  
    具体包括：
    1. 明确样本数量与特征构成
    2. 观察特征数据的原始分布形态
    3. 计算目标变量的集中趋势与离散程度。  
    """)    
    
    left,mid,right = st.columns([13,0.2,6])
    
    with left:
        # 代码填空区域
        code_template = """
# 加载糖尿病数据集
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X_raw = diabetes.data      # 特征数据
y_raw = diabetes.target    # 目标变量（疾病预测评分）

# 观察数据
print("特征数据形状：", X_raw.___Q1___) # 查看特征数据形状
print("目标变量形状：", y_raw.shape) # 查看目标变量形状
print("前3行特征：\\n", X_raw[___Q2___]) # 查看前3行特征

import numpy as np
# 计算统计量
print("疾病预测评分 均值：", np.___Q3___(y_raw)) # 计算目标变量均值
print("疾病预测评分 标准差：", np.___Q4___(y_raw)) # 计算目标变量标准差
        """.strip()
        st.code(code_template, language="python")
    with right:        
        # 代码选择填空组件
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 选择获取数组形状的属性",
            "Q2. 选择正确的切片语法",
            "Q3. 选择计算均值的函数",
            "Q4. 选择计算标准差的函数"
        ]
        options = [
            [".shape", ".size", ".dim", ".shape()"],
            ["0:3", "3:", "0,3", "3"],
            ["mean", "average", "median", "sum"],
            ["std", "var", "stddev", "deviation"]
        ]
        correct_answers = [".shape", "0:3", "mean", "std"]
        
        shape_attr = st.selectbox(questions[0], options[0], key="fill1", index=None)
        slice_syntax = st.selectbox(questions[1], options[1], key="fill2", index=None)
        mean_func = st.selectbox(questions[2], options[2], key="fill3", index=None)
        std_func = st.selectbox(questions[3], options[3], key="fill4", index=None)
    
    # 会话状态保存运行成功的标志
    if 'step1_success' not in st.session_state:
        st.session_state.step1_success = False
    
    # 验证答案并展示结果
    if st.button("运行代码", key="run_step1"):
        current_answers = [shape_attr, slice_syntax, mean_func, std_func]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(2, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(2, q, ans, correct_ans)
        
        if all(correct):
            st.success("代码运行成功！输出结果：")
            diabetes = load_diabetes()
            X_raw = diabetes.data
            y_raw = diabetes.target
            
            with st.expander("查看输出"):                
                st.write("特征数据形状：", X_raw.shape)
                st.write("目标变量形状：", y_raw.shape)
                st.write("前3行特征：")
                st.write(X_raw[0:3].tolist())
                st.write("疾病预测评分 均值：", np.mean(y_raw))
                st.write("疾病预测评分 标准差：", np.std(y_raw))
            
            # 保存数据到会话状态
            st.session_state.data = X_raw
            st.session_state.y_raw = y_raw
            st.session_state.step1_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step1_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：mean）",
        key="step1_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.linear_step_records['reflection']['step_1'] = reflection
    
    if st.session_state.step1_success:
        st.info("哇！✨ 数据观察任务完美完成，太厉害啦！为后续分析打下好基础，继续加油！💪")
        if st.button("进入下一步：数据预处理", key="to_step2"):
            complete_step(2)  # 标记步骤2完成
            st.session_state.step += 1
            st.session_state.step1_success = False
            st.rerun()



# 步骤2：数据预处理
def step2():
    st.header("数据预处理")
    st.subheader("目标：划分训练集/测试集，标准化特征")
    st.info("""
    **任务说明**：  
    1. 数据集拆分：将样本划分为训练集（用于模型参数学习）与测试集（用于评估泛化能力），通常采用8:2的拆分比例  
    2. 特征标准化：通过均值-标准差转换消除量纲影响，使不同指标处于同一数量级，确保模型学习过程的公平性  
    """)    
    left,mid,right = st.columns([13,0.2,6])
    
    with left:
        # 代码填空区域
        code_template = """
# 划分训练集和测试集
from sklearn.model_selection import train_test_split

# 测试集数据占20%，随机数种子为42
X_train, X_test, y_train, y_test=train_test_split(
                                                  X_raw,
                                                  y_raw,
                                                  test_size=___Q1___,
                                                  random_state=___Q2___)

# 特征标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.___Q3___(X_train)   # 训练集用fit_transform
X_test_scaled = scaler.___Q4___(X_test)    # 测试集用transform

print("训练集特征形状：", X_train_scaled.shape)
print("测试集特征形状：", X_test_scaled.shape)
        """.strip()
        st.code(code_template, language="python")

    with right:        
        # 代码选择填空组件
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 测试集占比参数值",
            "Q2. 随机数种子参数值",
            "Q3. 训练集标准化方法",
            "Q4. 测试集标准化方法"
        ]
        options = [
            ["0.1", "0.2", "0.8", "1.0"],
            ["0", "10", "42", "100"],
            ["fit", "transform", "fit_transform", "predict"],
            ["fit", "transform", "fit_transform", "predict"]
        ]
        correct_answers = ["0.2", "42", "fit_transform", "transform"]
        
        test_size = st.selectbox(questions[0], options[0], key="fill1", index=None)
        random_state = st.selectbox(questions[1], options[1], key="fill2", index=None)
        train_method = st.selectbox(questions[2], options[2], key="fill3", index=None)
        test_method = st.selectbox(questions[3], options[3], key="fill4", index=None)
    
    # 会话状态保存运行成功的标志
    if 'step2_success' not in st.session_state:
        st.session_state.step2_success = False
    
    # 验证答案并展示结果
    if st.button("运行代码", key="run_step2"):
        current_answers = [test_size, random_state, train_method, test_method]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(3, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(3, q, ans, correct_ans)
        
        if all(correct):
            st.success("代码运行成功！输出结果：")
            # 获取步骤1保存的数据
            X = st.session_state.data
            y = st.session_state.y_raw
                
            # 执行数据预处理
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
                
            # 展示执行结果
              
            st.write(f"##### 📍 训练集特征形状：{X_train_scaled.shape}")
            st.write(f"##### 📍 测试集特征形状：{X_test_scaled.shape}" )
                
            # 保存到会话状态
            st.session_state.X_train = X_train_scaled
            st.session_state.X_test = X_test_scaled
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.step2_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step2_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：fit_transform）",
        key="step2_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.linear_step_records['reflection']['step_2'] = reflection

    # 下一步按钮
    if st.session_state.step2_success:
        st.info("太出色了！🌟 数据预处理滴水不漏，真了不起！赶紧进入模型构建环节吧！冲呀！")
        if st.button("进入下一步：构建线性回归模型", key="to_step3"):
            complete_step(3)  # 标记步骤3完成
            st.session_state.step = 3
            st.session_state.step2_success = False
            st.rerun()

# 步骤3：构建线性回归模型
def step3():
    st.header("构建线性回归模型")
    st.subheader("目标：实例化LinearRegression模型")
    st.info("""
    **任务说明**：  
    1. 模型训练（fit）：基于训练集求解最优参数（权重与截距），使模型对已知样本的预测误差最小化  
    2. 预测推理（predict）：使用训练好的模型对测试集样本进行病情预测，验证模型的实际应用效果  
    特征权重绝对值的大小可反映该指标对病情进展的影响强度。  
    """)    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        # 代码填空区域
        code_template = """
# 导入线性回归模型
from sklearn.linear_model import ___Q1___

# 实例化模型
model = ___Q2___

# 查看模型参数
print("模型参数：", model.___Q3___())
        """.strip()
        st.code(code_template, language="python")

    with right:        
        # 代码选择填空组件
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 选择线性回归模型类",
            "Q2. 选择正确的实例化代码",
            "Q3. 选择获取模型参数的方法"
        ]
        options = [
            ["Linear", "Regression", "LinearRegression", "LinearModel"],
            ["LinearRegression", "LinearRegression()", "new LinearRegression()", "LinearRegression.create()"],
            ["params", "get_params", "get_parameters", "show_params"]
        ]
        correct_answers = ["LinearRegression", "LinearRegression()", "get_params"]
        
        model_class = st.selectbox(questions[0], options[0], key="fill1", index=None)
        instantiate_code = st.selectbox(questions[1], options[1], key="fill2", index=None)
        get_params_method = st.selectbox(questions[2], options[2], key="fill3", index=None)
    
    # 会话状态保存运行成功的标志
    if 'step3_success' not in st.session_state:
        st.session_state.step3_success = False
    
    # 验证答案并展示结果
    if st.button("运行代码", key="run_step3"):
        current_answers = [model_class, instantiate_code, get_params_method]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(4, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(4, q, ans, correct_ans)
        
        if all(correct):
            st.success("代码运行成功！输出结果：")
            # 执行模型构建代码
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            
            # 展示执行结果
            with st.expander("查看输出"):                
                st.write("模型参数：", model.get_params())
            
            # 保存到会话状态
            st.session_state.model = model
            st.session_state.step3_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step3_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：实例化）",
        key="step3_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.linear_step_records['reflection']['step_3'] = reflection

    # 下一步按钮
    if st.session_state.step3_success:
        st.info("不可思议！🤩 模型构建成功，每一步都精准！就等你来训练模型啦！🔥")
        if st.button("进入下一步：模型训练与预测", key="to_step4"):
            complete_step(4)  # 标记步骤4完成
            st.session_state.step = 4
            st.session_state.step3_success = False
            st.rerun()

# 步骤4：模型训练与预测
def step4():
    st.header("模型训练与预测")
    st.subheader("目标：训练模型并预测患者病情")
    st.info("""
    **任务说明**：  
    1. 用训练集让模型学习（fit）：就像医生学习病例  
    2. 用测试集让模型预测（predict）：就像医生判断新患者病情  
    请完成以下代码填空，实现模型训练与预测
    """)    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        # 代码填空区域
        code_template = """
# 用训练集训练模型
model.___Q1___(X_train_scaled, y_train)

# 查看模型参数
print("特征系数（权重）：", model.___Q2___)
print("截距：", model.intercept_)

# 用测试集预测
y_pred = model.___Q3___(X_test_scaled)

# 查看预测结果
print("前5个预测值：", y_pred[:5])
print("前5个实际值：", y_test[:5])
        """.strip()
        st.code(code_template, language="python")

    with right:        
        # 代码选择填空组件
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 选择模型训练方法",
            "Q2. 选择特征系数属性",
            "Q3. 选择预测方法"
        ]
        options = [
            ["train", "fit", "learn", "predict"],
            ["coef", "coef_", "coefficients", "weights"],
            ["predict", "forecast", "estimate", "guess"]
        ]
        correct_answers = ["fit", "coef_", "predict"]
        
        train_method = st.selectbox(questions[0], options[0], key="fill1", index=None)
        coef_attr = st.selectbox(questions[1], options[1], key="fill2", index=None)
        predict_method = st.selectbox(questions[2], options[2], key="fill3", index=None)
    
    # 会话状态保存运行成功的标志
    if 'step4_success' not in st.session_state:
        st.session_state.step4_success = False
    
    # 验证答案并展示结果
    if st.button("运行代码", key="run_step4"):
        current_answers = [train_method, coef_attr, predict_method]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(5, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(5, q, ans, correct_ans)
        
        if all(correct):
            # 执行模型训练代码
            model = st.session_state.model
            X_train_scaled = st.session_state.X_train
            y_train = st.session_state.y_train
            X_test_scaled = st.session_state.X_test
            y_test = st.session_state.y_test
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # 展示执行结果
            with st.expander("查看输出"):                
                st.write("特征系数（权重）：", model.coef_.tolist())
                st.write("截距：", model.intercept_)
                st.write("前5个预测值：", y_pred[:5].tolist())
                st.write("前5个实际值：", y_test[:5].tolist())
            
            # 特征重要性可视化
            st.subheader("🔍 特征对病情的影响程度")
            coef_df = pd.DataFrame({
                "特征": st.session_state.chinese_feature_names,
                "影响系数": model.coef_
            })
            coef_df = coef_df.sort_values("影响系数", ascending=False)
            cols = st.columns([1,5,1])
            with cols[1]:
                plt.figure(figsize=(10, 6))
                plt.barh(coef_df["特征"], coef_df["影响系数"], color='lightblue')
                plt.xlabel("影响系数（正值表示加剧病情，负值表示缓解）")
                plt.title("各生理指标对糖尿病进展的影响")
                st.pyplot(plt)
            
            # 保存到会话状态
            st.session_state.y_pred = y_pred
            st.session_state.step4_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step4_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：coef_）",
        key="step4_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.linear_step_records['reflection']['step_4'] = reflection

    # 下一步按钮
    if st.session_state.step4_success:
        st.success("太棒了！🚀 模型训练和预测成功，每一步都很精准！赶紧看看结果吧～")
        if st.button("进入下一步：模型评估", key="to_step5"):
            complete_step(5)  # 标记步骤5完成
            st.session_state.step = 5
            st.session_state.step4_success = False
            st.rerun()

# 步骤5：模型评估
def step5():
    st.header("模型评估")
    st.subheader("目标：评估模型预测效果")
    st.info("""
    **任务说明**：  
    用两个指标评估模型好坏：  
    1. 均方误差（MSE）：预测值与实际值的平均平方差，越小越好  
    2. 决定系数（R²）：模型能解释的 variance 比例，越接近1越好  
    请完成以下代码填空，实现模型评估
    """)    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        # 代码填空区域
        code_template = """
# 导入评估指标
from sklearn.metrics import ___Q1___, ___Q2___

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差（MSE）：{mse:.2f}")
print(f"决定系数（R²）：{r2:.2f}")
        """.strip()
        st.code(code_template, language="python")

    with right:        
        # 代码选择填空组件
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 选择均方误差函数",
            "Q2. 选择决定系数函数"
        ]
        options = [
            ["mean_squared_error", "mse", "mean_square_error", "ms_error"],
            ["r2", "r2_score", "r_squared", "r2_function"]
        ]
        correct_answers = ["mean_squared_error", "r2_score"]
        
        metric1 = st.selectbox(questions[0], options[0], key="fill1", index=None)
        metric2 = st.selectbox(questions[1], options[1], key="fill2", index=None)
    
    # 会话状态保存运行成功标志
    if 'step5_success' not in st.session_state:
        st.session_state.step5_success = False
    
    # 验证答案并展示结果
    if st.button("运行代码", key="run_step5"):
        current_answers = [metric1, metric2]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(6, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(6, q, ans, correct_ans)
        
        if all(correct):            
            # 执行评估代码
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 展示执行结果
            st.write(f"##### 📍 均方误差（MSE）：{mse:.2f}")
            st.write(f"##### 📍 决定系数（R²）：{r2:.2f}")
            
            # 预测vs实际值可视化
            st.subheader("📈 预测效果对比")
            cols = st.columns([1,5,1])
            with cols[1]:
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred, alpha=0.6)
                plt.plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 'r--')
                plt.xlabel("实际病情进展")
                plt.ylabel("预测病情进展")
                plt.title("预测值 vs 实际值（越靠近红线越准确）")
                st.pyplot(plt)
            
            # 保存到会话状态
            st.session_state.mse = mse
            st.session_state.r2 = r2
            st.session_state.step5_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step5_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：r2_score）",
        key="step5_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.linear_step_records['reflection']['step_5'] = reflection

    # 下一步按钮
    if st.session_state.step5_success:
        st.success("太出色了！🌟 模型评估完成，指标计算准确无误！快去看看模型表现吧～")
        if st.button("进入下一步：总结与思考", key="to_step6"):
            complete_step(6)  # 标记步骤6完成
            st.session_state.step = 6
            st.session_state.step5_success = False
            st.rerun()

# 步骤6：总结与思考
def step6():
    st.header("总结与思考")
    st.subheader("目标：分析模型结果并提出改进方向")
    st.info("""
    **任务说明**：  
    基于模型评估结果，理解模型表现并思考改进方向  
    1. 分析评估指标的含义和模型表现  
    2. 结合特征影响分析，解释关键发现  
    3. 提出至少2点可行的改进建议  
    """)   
   
    # 显示上一步得到的评估指标
    st.subheader("📊 模型评估关键结果")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("均方误差（MSE）", f"{st.session_state.mse:.2f}")
        st.caption("说明：预测值与实际值的平均平方差，值越小表示预测越准确")
    
    with col2:
        st.metric("决定系数（R²）", f"{st.session_state.r2:.2f}")
        st.caption("说明：模型可解释的方差比例，越接近1表示模型拟合效果越好")
    
    # 特征影响分析可视化
    st.subheader("🔍 特征影响程度回顾")
    model = st.session_state.model
    coef_df = pd.DataFrame({
        "特征": st.session_state.chinese_feature_names,
        "影响系数": model.coef_,
        "影响程度（绝对值）": abs(model.coef_)
    }).sort_values("影响程度（绝对值）", ascending=False)
    cols = st.columns([1,4,1])
    with cols[1]:
        plt.figure(figsize=(10, 6))
        plt.barh(coef_df["特征"], coef_df["影响系数"], color=['red' if x > 0 else 'green' for x in coef_df["影响系数"]])
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.xlabel("影响系数（正值加剧病情，负值缓解病情）")
        plt.title("各生理指标对糖尿病进展的影响程度")
        st.pyplot(plt)
    
    st.dataframe(coef_df, use_container_width=True)
    
    # 知识小测验
    st.subheader("📌 理解测试")
    questions = [
        "T1. 若模型的R²为0.3，说明什么？",
        "T2. 特征系数为正值表示该特征与病情进展的关系是？",
        "T3. 以下哪项不属于有效的模型改进方法？"
    ]
    options = [
        [
            "模型可解释30%的方差变化，拟合效果一般",
            "模型准确率为30%，表现较差",
            "模型错误率为30%，需要优化",
            "模型稳定性为30%，可靠性低"
        ],
        ["正相关（特征值越高，病情可能越严重）", "负相关（特征值越高，病情可能越轻）", "无相关关系", "非线性关系"],
        ["尝试多项式回归捕捉非线性关系", "增加更多患者特征数据", "删除所有影响系数为负的特征", "使用正则化减少过拟合"]
    ]
    correct_answers = [
        "模型可解释30%的方差变化，拟合效果一般",
        "正相关（特征值越高，病情可能越严重）",
        "删除所有影响系数为负的特征"
    ]
    
    # 生成测验选项
    q6_1 = st.radio(questions[0], options[0], key="q6_1", index=None)
    q6_2 = st.radio(questions[1], options[1], key="q6_2", index=None)
    q6_3 = st.radio(questions[2], options[2], key="q6_3", index=None)
    current_answers = [q6_1, q6_2, q6_3]

    # 初始化状态变量
    if 'analysis_submitted' not in st.session_state:
        st.session_state.analysis_submitted = False
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False  # 新增报告显示状态
    
    # 学生分析输入区域
    st.subheader("📌 分析与改进")
    reflection = st.text_input(
        "请结合线性回归评估指标，对模型的回归效果做出评价和分析，并给出改进意见",
        key="step6_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.linear_step_records['reflection']['step_6'] = reflection
   
    # 记录答题详情和错误信息
    if st.button("提交理解测试与我的分析改进意见", key="submit_analysis"):
        # 验证测验答案
        quiz_correct = [a == b for a, b in zip(current_answers, correct_answers)]
        all_answered = all(ans is not None for ans in current_answers)
        
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, quiz_correct):
            record_answer(6, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(6, q, ans, correct_ans)
        
        # 验证分析内容
        if not all(quiz_correct):
            st.error("理解测试存在错误，请修正后再提交")
            for i, is_correct in enumerate(quiz_correct):
                if not is_correct:
                    st.warning(f"第{i+1}题回答错误，正确答案：{correct_answers[i]}")
        elif not all_answered:
            st.error("请完成所有理解测试题")
        elif not reflection.strip():
            st.error("请填写你的分析改进意见")
        else:
            st.session_state.analysis_submitted = True
            st.success("分析提交成功！")

    # 完成流程与报告生成逻辑
    if st.session_state.analysis_submitted:
        # 显示完成流程按钮
        if st.button("1.完成全部流程", key="finish_all"):
            complete_step(6)
            st.balloons()
            st.success("🎉 恭喜你完成所有步骤！你已成功掌握线性回归分析的完整流程～")
            st.info("""
                本次实践总结：
                1. 你完成了从数据加载到模型评估的完整机器学习流程
                2. 掌握了线性回归模型的构建、训练和评估方法
                3. 学会了分析模型结果并提出改进方向
                        
                后续可以尝试：
                - 使用其他回归模型（如决策树、随机森林）进行对比
                - 尝试特征工程提升模型表现
                - 调整模型参数优化预测效果
                """)
                
        # 生成报告按钮 - 核心修改点
        if st.button("2.生成线性回归分步编程学习报告", key="generate_report"):
            st.session_state.show_report = True  # 切换状态
            st.rerun()  # 刷新页面
        if st.session_state.show_report:
            # 显示报告内容
            report = generate_report_step(
                raw_records=st.session_state.linear_step_records,steps=6
            )
            st.subheader("📊 线性回归分步编程学习报告")
            st.caption(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(report)
            st.session_state.show_report = False
          

# 主程序
def main():
    st.title("🏥 实战：糖尿病病情预测（线性回归）")

    # 初始化用户操作记录
    if 'linear_step_records' not in st.session_state:
        st.session_state.linear_step_records = {
            'total_steps': 7,  # 总步骤数
            'current_step': 0,  # 当前步骤
            'step_records': {},  # 各步骤详细记录
            'total_errors': 0,  # 总错误次数
            'reflection': {}  # 各步骤反思
        }

    # 初始化步骤记录结构
    for step_num in range(0, 7):
        if f'step_{step_num}' not in st.session_state.linear_step_records['step_records']:
            st.session_state.linear_step_records['step_records'][f'step_{step_num}'] = {
                'start_time': None,
                'end_time': None,
                'duration': 0,
                'error_count': 0,
                'error_details': [],
                'is_completed': False,
                'answer_details': []  
            }

    # 初始化会话状态
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'y_raw' not in st.session_state:
        st.session_state.y_raw = None
    if 'chinese_feature_names' not in st.session_state:
        st.session_state.chinese_feature_names = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'y_pred' not in st.session_state:
        st.session_state.y_pred = None
    if 'mse' not in st.session_state:
        st.session_state.mse = None
    if 'r2' not in st.session_state:
        st.session_state.r2 = None

   
    # 侧边栏步骤进度显示
    st.sidebar.title("步骤进度")
    steps = [
        "0. 项目说明",
        "1. 数据观察", "2. 数据预处理",
        "3. 模型构建", "4. 训练预测", "5. 模型评估", "6. 总结与思考"  
    ]
    
    for i, step in enumerate(steps):
        if st.session_state.step > i:
            st.sidebar.markdown(f"✔️ **{step}**")
        elif st.session_state.step == i:
            st.sidebar.markdown(f"🌟 **{step}**")
        else:
            st.sidebar.markdown(f"⭕ {step}")
    
    # 步骤处理逻辑
    if st.session_state.step == 0:
        step0()
    elif st.session_state.step == 1:
        step1()
    elif st.session_state.step == 2:
        step2()
    elif st.session_state.step == 3:
        step3()
    elif st.session_state.step == 4:
        step4()
    elif st.session_state.step == 5:
        step5()
    elif st.session_state.step == 6:
        step6()

if __name__ == "__main__":
    main()
