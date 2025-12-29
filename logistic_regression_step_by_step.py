# streamlit run C:\Users\孙冰\Desktop\AI助教25-12-07\logistic_regression_step_by_step.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time
from learning_report import generate_report_step
from datetime import datetime
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")

# 中文特征名定义
FEATURE_NAMES_CHINESE = [
    "平均半径", "平均纹理", "平均周长", "平均面积", "平均光滑度",
    "平均紧凑度", "平均凹度", "平均凹点", "平均对称性", "平均分形维数",
    "半径误差", "纹理误差", "周长误差", "面积误差", "光滑度误差",
    "紧凑度误差", "凹度误差", "凹点误差", "对称性误差", "分形维数误差",
    "最大半径", "最大纹理", "最大周长", "最大面积", "最大光滑度",
    "最大紧凑度", "最大凹度", "最大凹点", "最大对称性", "最大分形维数"
]
def record_error(step_num, question, user_answer, correct_answer):
    """记录错误信息"""
    error_info = {
        'question': question,
        'user_answer': user_answer,
        'correct_answer': correct_answer,
        'time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['error_count'] += 1
    st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['error_details'].append(error_info)
    st.session_state.logistic_step_records['total_errors'] += 1

def record_answer(step_num, question, user_answer, correct_answer, is_correct):
    """记录答题详情"""
    answer_info = {
        'question': question,
        'user_answer': user_answer,
        'correct_answer': correct_answer,
        'is_correct': is_correct,
        'time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['answer_details'].append(answer_info)

def start_step_timer(step_num):
    """记录步骤开始时间"""
    if st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['start_time'] is None:
        st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['start_time'] = time.time()

def complete_step(step_num):
    """标记步骤完成并计算耗时"""
    st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['end_time'] = time.time()
    start_time = st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['start_time'] or time.time()
    st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['duration'] = round(
        st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['end_time'] - start_time, 2)
    st.session_state.logistic_step_records['step_records'][f'step_{step_num}']['is_completed'] = True
    st.session_state.logistic_step_records['current_step'] = step_num

# 初始化会话状态
def init_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'logistic_step_records' not in st.session_state:
        st.session_state.logistic_step_records = {
            'total_errors': 0,
            'step_records': {
                f'step_{i}': {
                    'error_count': 0, 
                    'error_details': [], 
                    'answer_details': [],
                    'start_time': None,
                    'end_time': None,
                    'duration': 0,
                    'is_completed': False
                } for i in range(7)
            },
            'reflection': {f'step_{i}': '' for i in range(7)},
            'analysis': ''
        }
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False        
    if 'analysis_submitted' not in st.session_state:
        st.session_state.analysis_submitted = False        

# 步骤0：项目说明
def step0():
    st.subheader("乳腺癌检测与诊断分析")
    st.info("""
    **你的角色：**
    你是医疗数据分析师，需要帮助医生通过肿瘤特征数据判断肿瘤的良恶性。
   
    **任务背景：**
    医院收集了569名患者的乳腺肿瘤信息，包括：
    - 30项肿瘤特征（半径、纹理、周长等）📋
    - 诊断结果👉（**0=良性，1=恶性**）📊
    
    **你的目标：**
    用逻辑回归模型构建分类器，根据肿瘤特征准确区分良性和恶性肿瘤。🔍
    
    **任务拆解：**
    你需要完成7个步骤，一步步搭建分类模型：
    1. 数据观察：了解肿瘤特征数据的基本情况
    2. 数据预处理：划分训练集/测试集并标准化
    3. 搭建模型：实例化逻辑回归模型
    4. 训练预测：用数据训练模型并进行预测
    5. 模型评估：分析模型表现
    6. 改进建议：提出模型优化方向
    7. 反思总结：梳理完整流程与学习收获
    """)   
    
    # 加载数据集用于展示
    cancer = load_breast_cancer()
    
    st.subheader("数据集预览")
    df = pd.DataFrame(
        data=cancer.data,
        columns=FEATURE_NAMES_CHINESE
    )
    df['诊断结果'] = ['良性' if x == 0 else '恶性' for x in cancer.target]
    st.dataframe(df.head(10), use_container_width=True)
    
    # 知识小测验部分
    st.subheader("📌 知识小测验")
    questions = [
        "T1. 逻辑回归主要用于解决什么类型的问题？",
        "T2. 在本项目中，诊断结果（良性/恶性）属于什么变量？",
        "T3. 以下哪项是分类问题常用的评估指标？"
    ]
    options = [
        ["回归预测", "分类判断", "聚类分析", "降维处理"],
        ["特征变量", "输入变量", "目标变量", "解释变量"],
        ["均方误差", "决定系数", "准确率", "方差"]
    ]
    correct_answers = ['分类判断', '目标变量', '准确率']    
    q0_1 = st.radio(questions[0], options[0], key="q0_1", index=None)
    q0_2 = st.radio(questions[1], options[1], key="q0_2", index=None)
    q0_3 = st.radio(questions[2], options[2], key="q0_3", index=None)    
    current_answers = [q0_1, q0_2, q0_3]
    
    # 显示每个问题的即时反馈并记录答题情况
    for i, (q, ans, correct) in enumerate(zip(questions, current_answers, correct_answers)):
        if ans is not None:
            record_answer(0, q, ans, correct, ans == correct)
            if ans == correct:
                st.success(f"{i+1}. 回答正确")
            else:
                st.error(f"{i+1}. 回答错误，正确答案是：{correct}")
                record_error(0, q, ans, correct)
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：准确率）",
        key="step0_reflection",
        autocomplete="off",
    )    
    if reflection:
        st.session_state.logistic_step_records['reflection']['step_0'] = reflection
    
    # 下一步按钮
    all_answered = all(ans is not None for ans in current_answers)
    if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
        st.info("太棒了！🎉 你已掌握基础概念，准备好深入分析吧！")
        if st.button("进入下一步：数据观察与理解", key="next_step0"):
            complete_step(0)
            st.session_state.step = 1
            st.rerun()
    elif all_answered:
        st.warning("请先回答正确所有问题才能继续")
    else:
        st.info("请完成所有问题的回答")

# 步骤1：数据观察与理解（整合特征与目标变量划分）
def step1():
    st.header("数据观察与理解")
    st.subheader("目标：加载乳腺癌数据集，观察基本信息并划分变量")
    st.info("""
    **任务说明**：  
    数据探索是建模分析的基础环节。需系统考察数据集规模、特征分布及关键统计量，
    并明确特征变量与目标变量：
    1. 特征变量（X_raw）：用于预测的输入数据（肿瘤的各项特征）
    2. 目标变量（y_raw）：需要预测的结果（肿瘤的良恶性诊断）
    """)    
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        # 代码填空区域（整合特征与目标变量划分）
        code_template = """
# 加载乳腺癌数据集
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_raw = cancer.data      # 特征数据
y_raw = cancer.target    # 目标变量（0=良性，1=恶性）

# 观察数据
print("特征数据形状：", X_raw.shape) # 查看特征数据形状
print("目标变量形状：", y_raw.shape) # 查看目标变量形状
print("前3行特征：", X_raw[___Q1___]) # 查看前3行特征

import numpy as np
# 按列计算每个特征均值
print("每个特征的均值：", np.___Q2___(X_raw, axis=__Q3___))

# 按列计算每个特征方差
print("每个特征的方差：", np.___Q4___(X_raw, axis=__Q3___))  
        """.strip()
        st.code(code_template, language="python")
    
    with right:        
        # 代码选择填空组件
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 选择正确的切片语法",
            "Q2. 选择计算均值的函数",
            "Q3. 设置按列计算均值",
            "Q4. 选择计算方差的函数"
        ]
        options = [
            ["0:3", "3:", "0,3", "3"],
            ["mean", "average", "median", "sum"],
            ["0", "1", "[0]", "[1]"],
            ["var", "value_counts", "std", "bincount"]
        ]
        correct_answers = ["0:3", "mean","0", "var"]
        
        shape_attr = st.selectbox(questions[0], options[0], key="fill1", index=None)
        slice_syntax = st.selectbox(questions[1], options[1], key="fill2", index=None)
        mean_func = st.selectbox(questions[2], options[2], key="fill3", index=None)
        count_func = st.selectbox(questions[3], options[3], key="fill4", index=None)
    
    # 会话状态保存运行成功的标志
    if 'step1_success' not in st.session_state:
        st.session_state.step1_success = False
    
    # 验证答案并展示结果
    if st.button("运行代码", key="run_step1"):
        current_answers = [shape_attr, slice_syntax, mean_func, count_func]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(1, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(1, q, ans, correct_ans)
        
        if all(correct):
            st.success("代码运行成功！输出结果：")
            cancer = load_breast_cancer()
            X_raw = cancer.data
            y_raw = cancer.target
            
            with st.expander("查看输出"):                
                st.write("特征数据形状：", X_raw.shape)
                st.write("目标变量形状：", y_raw.shape)
                st.write("前3行特征：")
                st.write(X_raw[0:3].tolist())
          
            # 特征均值和方差显示
            data = {
                "特征名称":FEATURE_NAMES_CHINESE,
                "均值": np.mean(X_raw, axis=0),
                "方差": np.var(X_raw, axis=0)
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # 保存数据到会话状态
            st.session_state.X_raw = X_raw
            st.session_state.y_raw = y_raw
            st.session_state.cancer = cancer
            st.session_state.step1_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step1_success = False
            
    # 会话状态保存运行成功的标志
    if 'step1_que_success' not in st.session_state:
        st.session_state.step1_que_success = False
        
    if st.session_state.step1_success:          
    # 知识小测验部分
        st.subheader("📌 观察各特征均值与方差，回答以下问题：")
        questions = [
            "T1. 观察特征的均值数据，发现不同特征的均值数值差异很大（例如有的特征均值为几十，有的为几千），这种差异可能会对模型产生什么影响？",
            "T2. 若某些特征的方差极大（数值波动范围很大），而另一些特征的方差极小（数值几乎不变），这种情况可能会导致什么问题？"
        ]
        options = [
            ["导致模型更关注数值大的特征，忽略数值小的特征",
             "使模型训练速度加快", "提高模型预测的准确率", "对模型无任何影响"],
            ["方差大的特征对模型的影响被削弱",
             "方差小的特征更容易被模型捕捉到关键信息",
             "模型可能被方差大的特征主导，影响学习效果",
             "特征间的关联性增强"]
        ]
        correct_answers = ['导致模型更关注数值大的特征，忽略数值小的特征', '模型可能被方差大的特征主导，影响学习效果']    
        q1_1 = st.radio(questions[0], options[0], key="q1_1", index=None)
        q1_2 = st.radio(questions[1], options[1], key="q1_2", index=None)
        current_answers = [q1_1, q1_2]
        
        # 显示每个问题的即时反馈并记录答题情况
        for i, (q, ans, correct) in enumerate(zip(questions, current_answers, correct_answers)):
            if ans is not None:
                record_answer(1, q, ans, correct, ans == correct)
                if ans == correct:
                    st.success(f"{i+1}. 回答正确")
                else:
                    st.error(f"{i+1}. 回答错误，正确答案是：{correct}")
                    record_error(1, q, ans, correct)
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：特征变量）",
        key="step1_reflection",
        autocomplete="off"
    )
    if reflection:
        st.session_state.logistic_step_records['reflection']['step_1'] = reflection

    if st.session_state.step1_success:     
        all_answered = all(ans is not None for ans in current_answers)
        if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
            st.info("哇！✨ 数据观察任务完美完成，太厉害啦！为后续分析打下好基础，继续加油！💪")
            if st.button("进入下一步：数据预处理", key="to_step2"):
                complete_step(1)
                st.session_state.step = 2
                st.session_state.step1_success = False
                st.rerun()
        elif all_answered:
            st.warning("请先回答正确所有问题才能继续")
        else:
            st.info("请完成所有问题的回答")
        
# 步骤2：数据预处理
def step2():
    st.header("数据预处理")
    st.subheader("目标：划分训练集/测试集，标准化特征")
    st.info("""
    **任务说明**：  
    1. 数据集拆分：将样本划分为训练集（用于模型学习）与测试集（用于评估），采用8:2的比例  
    2. 特征标准化：通过均值-标准差转换消除量纲影响，使不同特征处于同一数量级  
    """)    
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        # 代码填空区域（使用更新后的模板）
        code_template = """
# 划分训练集和测试集
from sklearn.model_selection import train_test_split

# 测试集数据占20%，随机数种子为42
X_train, X_test, y_train, y_test = train_test_split(
                                            X_raw,
                                            y_raw,
                                            test_size=0.2,
                                            random_state=42)

# 特征标准化
from sklearn.preprocessing import ___Q1___
scaler = StandardScaler()

# 训练集用fit_transform
X_train_scaled = ___Q2___.fit_transform(___Q3___)

# 测试集用transform
X_test_scaled = scaler.transform(___Q4___)    
        """.strip()
        st.code(code_template, language="python")
    
    with right:        
        # 代码选择填空组件（匹配Q1-Q4的新含义）
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 需要导入的标准化类名",
            "Q2. 调用fit_transform方法的对象",
            "Q3. 训练集特征数据变量名",
            "Q4. 测试集特征数据变量名"
        ]
        options = [
            ["StandardScaler", "MinMaxScaler", "LabelEncoder", "OneHotEncoder"],  # Q1正确答案
            ["X_train", "scaler", "StandardScaler", "X_test"],  # Q2正确答案
            ["X_test", "y_train", "X_train", "y_test"],  # Q3正确答案
            ["X_test", "y_test", "X_train", "y_train"]   # Q4正确答案
        ]
        correct_answers = ["StandardScaler", "scaler", "X_train", "X_test"]
        
        # 对应Q1-Q4的选择框
        q1_answer = st.selectbox(questions[0], options[0], key="fill1", index=None)
        q2_answer = st.selectbox(questions[1], options[1], key="fill2", index=None)
        q3_answer = st.selectbox(questions[2], options[2], key="fill3", index=None)
        q4_answer = st.selectbox(questions[3], options[3], key="fill4", index=None)
    
    # 会话状态保存运行成功的标志
    if 'step2_success' not in st.session_state:
        st.session_state.step2_success = False
    
    # 验证答案并展示结果
    if st.button("运行代码", key="run_step2"):
        current_answers = [q1_answer, q2_answer, q3_answer, q4_answer]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(2, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(2, q, ans, correct_ans)
        
        if all(correct):
            st.success("代码运行成功！输出结果：")
            # 获取数据
            if 'X_raw' not in st.session_state or 'y_raw' not in st.session_state:
                st.error("请先完成数据加载步骤")
                return
            X = st.session_state.X_raw
            y = st.session_state.y_raw
                
            # 执行数据预处理
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
                
            # 输出每个特征标准化后的统计量（使用中文特征名）
            st.subheader("各特征标准化后统计量（训练集）")
            stats_data = []
            for i, feature_name in enumerate(FEATURE_NAMES_CHINESE):
                feature_data = X_train_scaled[:, i]
                max_val = round(feature_data.max(), 4)
                min_val = round(feature_data.min(), 4)
                mean_val = round(feature_data.mean(), 4)
                var_val = round(feature_data.var(), 4)
                stats_data.append({
                    "特征名称": feature_name,
                    "最大值": max_val,
                    "最小值": min_val,
                    "均值": mean_val,
                    "方差": var_val
                })
                
            # 用DataFrame展示统计结果
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # 保存到会话状态
            st.session_state.X_train = X_train_scaled
            st.session_state.X_test = X_test_scaled
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.step2_success = True

        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step2_success = False
            
    if st.session_state.step2_success:
        st.subheader("📌 观察各特征均值与方差，回答以下问题：")
        questions = [
            "T1. 以下关于归一化（Min-Max Scaling）和标准化（Z-Score）的说法，❌ 错误的是？",
            "T2. 在乳腺癌特征数据预处理中，若某特征存在大量极端异常值（如个别样本的 “最大面积” 特征值远高于其他样本），此时应优先选择哪种处理方式来消除量纲影响？"
        ]
        options = [
            ["异常值通常不会对归一化和标准化产生影响",
             "归一化会将特征缩放到固定的 [0,1]（或 [-1,1]）区间，标准化无固定取值范围",
             "标准化适用于数据分布近似正态分布的场景，归一化适用于需要特征在固定范围的场景",
             "两者都能消除特征量纲的影响，使不同特征具有可比性"],
            ["归一化，将特征缩放到 [0,1] 区间",
             "标准化，将特征转换为均值为 0、方差为 1 的分布",
             "先移除异常值，再使用标准化 / 归一化（根据模型需求选择）",
             "直接使用原始数据，异常值不影响预处理效果"]
        ]
        correct_answers = ['异常值通常不会对归一化和标准化产生影响',
                           '先移除异常值，再使用标准化 / 归一化（根据模型需求选择）']    
        q2_1 = st.radio(questions[0], options[0], key="q2_1", index=None)
        q2_2 = st.radio(questions[1], options[1], key="q2_2", index=None)
        current_answers = [q2_1, q2_2]
        
        # 显示每个问题的即时反馈并记录答题情况
        for i, (q, ans, correct) in enumerate(zip(questions, current_answers, correct_answers)):
            if ans is not None:
                record_answer(1, q, ans, correct, ans == correct)
                if ans == correct:
                    st.success(f"{i+1}. 回答正确")
                else:
                    st.error(f"{i+1}. 回答错误，正确答案是：{correct}")
                    record_error(1, q, ans, correct)
            
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：标准化）",
        key="step2_reflection",
        autocomplete="off",
    )
    if reflection:
        if 'logistic_step_records' not in st.session_state:
            st.session_state.logistic_step_records = {'reflection': {}}
        st.session_state.logistic_step_records['reflection']['step_2'] = reflection

    if st.session_state.step2_success:     
        all_answered = all(ans is not None for ans in current_answers)
        if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
            st.info("数据预处理完美收官啦🎉！特征们已经整整齐齐站好队，就等模型大显身手咯🚀 ")
            if st.button("进入下一步：构建逻辑回归模型", key="to_step3"):
                # 假设complete_step函数已定义
                complete_step(2)
                st.session_state.step = 3
                st.session_state.step2_success = False
                st.rerun()
                
# 步骤3：构建逻辑回归模型
def step3():
    st.header("构建逻辑回归模型")
    st.subheader("目标：导入并实例化逻辑回归模型")
    st.info("""
    **任务说明**：  
    逻辑回归是分类任务的常用模型，需完成：
    1. 从sklearn正确导入逻辑回归模型类
    2. 实例化模型（使用默认参数即可）
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        # 代码填空区域
        code_template = """
# 从sklearn.linear_model导入逻辑回归模型类
from sklearn.linear_model import ___Q1___

# 实例化逻辑回归模型
model = LogisticRegression()
        """.strip()
        st.code(code_template, language="python")
    
    with right:        
        # 代码选择填空组件
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 逻辑回归模型的类名"
        ]
        options = [
            ["LogisticRegression", "Logistic", "Regression", "LinearRegression"]
        ]
        correct_answers = ["LogisticRegression"]
        
        class_name = st.selectbox(questions[0], options[0], key="fill1", index=None)
    
    # 会话状态保存运行成功的标志
    if 'step3_success' not in st.session_state:
        st.session_state.step3_success = False
    
    # 验证答案并展示结果
    if st.button("运行代码", key="run_step3"):
        current_answers = [class_name]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(3, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(3, q, ans, correct_ans)
        
        if all(correct):
            try:
                # 执行正确的模型实例化代码

                model = LogisticRegression()
                
                st.session_state.model = model
                st.success("模型构建成功！")
                st.session_state.step3_success = True
            except Exception as e:
                st.error(f"执行错误：{str(e)}")
                st.session_state.step3_success = False
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step3_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：模型类的导入）",
        key="step3_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.logistic_step_records['reflection']['step_3'] = reflection
    
    if st.session_state.step3_success:
        st.info("逻辑回归模型已经组装完毕啦🔧！参数们都各就各位，💪 准备启动训练")
        if st.button("进入下一步：模型训练与预测", key="to_step4"):
            complete_step(3)
            st.session_state.step = 4
            st.session_state.step3_success = False
            st.rerun()

# 步骤4：模型训练与预测
def step4():
    st.header("模型训练与预测")
    st.subheader("目标：训练逻辑回归模型并进行预测")
    st.info("""
    **任务说明**：  
    1. 使用训练集训练模型（fit）：学习特征与肿瘤良恶性之间的关系  
    2. 使用测试集进行预测（predict）：对未知样本进行分类判断  
    请完成代码实现模型训练与预测功能。
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
y_pred = model.___Q3___(X_test_scaled) # 预测类别
y_pred_proba = model.___Q4___(X_test_scaled) # 预测概率

# 查看预测结果
print("前5个预测类别：", y_pred[:5])
print("前5个实际类别：", y_test[:5])
print("前5个预测概率：", y_pred_proba[:5])
        """.strip()
        st.code(code_template, language="python")
    
    with right:        
        # 代码选择填空组件
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 选择模型训练方法",
            "Q2. 选择特征系数属性",
            "Q3. 选择预测类别方法",
            "Q4. 选择预测概率方法"
        ]
        options = [
            ["train", "fit", "learn", "estimate"],
            ["coef", "coef_", "coefficients", "weights"],
            ["predict", "forecast", "classify", "guess"],
            ["predict_proba", "probability", "predict_prob", "get_proba"]
        ]
        correct_answers = ["fit", "coef_", "predict", "predict_proba"]
        
        train_method = st.selectbox(questions[0], options[0], key="fill1", index=None)
        coef_attr = st.selectbox(questions[1], options[1], key="fill2", index=None)
        predict_method = st.selectbox(questions[2], options[2], key="fill3", index=None)
        proba_method = st.selectbox(questions[3], options[3], key="fill4", index=None)
    
    # 会话状态保存运行成功的标志
    if 'step4_success' not in st.session_state:
        st.session_state.step4_success = False
    
    # 验证答案并展示结果
    if st.button("运行代码", key="run_step4"):
        current_answers = [train_method, coef_attr, predict_method, proba_method]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(4, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(4, q, ans, correct_ans)
        
        if all(correct):
            # 执行模型训练代码
            model = st.session_state.model
            X_train_scaled = st.session_state.X_train
            y_train = st.session_state.y_train
            X_test_scaled = st.session_state.X_test
            y_test = st.session_state.y_test
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # 展示执行结果
            with st.expander("查看输出"):
                st.write("特征系数（权重）：", model.coef_.tolist())
                st.write("截距：", model.intercept_)
                st.write("前5个预测类别：", y_pred[:5].tolist())
                st.write("前5个实际类别：", y_test[:5].tolist())
                st.write("前5个预测概率：\n", y_pred_proba[:5].tolist())

            st.subheader("前5个样本预测结果对比")
            comparison_df = pd.DataFrame({
                "样本序号": [f"样本{i+1}" for i in range(5)],
                "预测类别": y_pred[:5].tolist(),
                "预测概率": y_pred_proba[:5].tolist(),
                "实际类别": y_test[:5].tolist()           
                })
            st.dataframe(comparison_df, use_container_width=True)

            # 特征重要性可视化
            st.subheader("🔍 特征对分类的影响程度（影响最大的10各特征）")
            coef_df = pd.DataFrame({
                "特征": FEATURE_NAMES_CHINESE,
                "影响系数": model.coef_[0],  # 逻辑回归系数
                "系数绝对值": abs(model.coef_[0])  # 用于排序的重要性指标
            })
            # 按系数绝对值降序排序，取前10个最重要特征
            top10_coef_df = coef_df.sort_values("系数绝对值", ascending=False).head(10)
            # 为了可视化时保持从大到小的顺序（按原始系数值）
            top10_coef_df = top10_coef_df.sort_values("影响系数", ascending=False)

            cols = st.columns([1,5,1])
            with cols[1]:
                # 绘图
                plt.figure(figsize=(10, 6))
                # 为正负系数设置不同颜色（正向：蓝色，负向：橙色）
                colors = ['lightblue' if x > 0 else 'orange' for x in top10_coef_df["影响系数"]]
                plt.barh(top10_coef_df["特征"], top10_coef_df["影响系数"], color=colors)

                # 添加参考线（y=0），更直观区分正负影响
                plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

                plt.xlabel("影响系数（正值倾向恶性，负值倾向良性）")
                plt.title("各特征对肿瘤分类的影响（重要性前10，含正负向）")
                st.pyplot(plt)
            
            # 保存到会话状态
            st.session_state.y_pred = y_pred
            st.session_state.y_pred_proba = y_pred_proba
            st.session_state.step4_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step4_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：预测概率）",
        key="step4_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.logistic_step_records['reflection']['step_4'] = reflection

    if st.session_state.step4_success:
        st.info("咯📊～ 准备计算各项性能指标，看看你的模型到底表现有多棒吧✨！")
        if st.button("进入下一步：模型评估与改进", key="to_step5"):
            complete_step(4)
            st.session_state.step = 5
            st.session_state.step4_success = False
            st.rerun()


# 步骤5：模型评估与改进
def step5():
    st.header("模型评估与改进")
    st.subheader("目标：分析模型分类效果并提出改进方向")
    st.info("""
    **任务说明**：  
    基于模型评估结果，理解模型表现并思考改进方向  
    1. 分析分类评估指标的含义和模型表现
    2. 结合具体应用场景，理解评估指标的意义
    """)   
   
    # 代码填空区域（新增）
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 计算模型评估指标
from sklearn.metrics import accuracy_score, precision_score,
                    recall_score, f1_score, confusion_matrix

accuracy = ___Q1___(y_test, y_pred) # 计算准确率

precision = ___Q2___(y_test, y_pred) # 计算精确率

recall = ___Q3___(y_test, y_pred) # 计算召回率

f1 = ___Q4___(y_test, y_pred) # 计算F1分数

cm = ___Q5___(y_test, y_pred) # 计算混淆矩阵

print(f"准确率: {accuracy:.2f}")
print(f"精确率: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1分数: {f1:.2f}")
print("混淆矩阵:", cm)
        """.strip()
        st.code(code_template, language="python")
    
    with right:        
        # 代码选择填空组件（新增）
        st.write("请选择正确的评估指标函数填空:")
        questions = [
            "Q1. 准确率计算函数",
            "Q2. 精确率计算函数",
            "Q3. 召回率计算函数",
            "Q4. F1分数计算函数",
            "Q5. 混淆矩阵计算函数",            
        ]
        options = [
            ["accuracy_score", "precision", "acc_score", "accuracy"],
            ["precision", "precision_score", "prec_score", "precise"],
            ["recall", "recall_score", "rec_score", "sensitivity"],
            ["f1", "f1_measure", "f1_score", "f_measure"],
            ["confusion_matrix", "matrix", "cm", "f_cm"]
        ]
        correct_answers = ["accuracy_score", "precision_score", "recall_score", "f1_score","confusion_matrix"]
        
        q1_answer = st.selectbox(questions[0], options[0], key="fill1", index=None)
        q2_answer = st.selectbox(questions[1], options[1], key="fill2", index=None)
        q3_answer = st.selectbox(questions[2], options[2], key="fill3", index=None)
        q4_answer = st.selectbox(questions[3], options[3], key="fill4", index=None)
        q5_answer = st.selectbox(questions[4], options[4], key="fill5", index=None)
    # 会话状态保存运行成功的标志（新增）
    if 'step5_success' not in st.session_state:
        st.session_state.step5_success = False
    
    # 验证答案并展示结果（新增）
    if st.button("运行代码", key="run_step5"):
        current_answers = [q1_answer, q2_answer, q3_answer, q4_answer,q5_answer]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(5, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(5, q, ans, correct_ans)
        
        if all(correct):
            st.success("代码运行成功！输出结果：")
            # 获取预测结果
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            st.session_state.step5_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step5_success = False
    
    # 显示评估指标（原内容）
    if st.session_state.step5_success:  # 仅在代码运行成功后显示详细评估
        st.subheader("📊 模型评估关键结果")
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("准确率（Accuracy）", f"{accuracy:.2f}")
            st.caption("说明：正确分类的样本占比")
        with col2:
            st.metric("精确率（Precision）", f"{precision:.2f}")
            st.caption("说明：预测为恶性的样本中实际为恶性的比例")
        with col3:
            st.metric("召回率（Recall）", f"{recall:.2f}")
            st.caption("说明：实际为恶性的样本中被正确预测的比例")
        with col4:
            st.metric("F1分数", f"{f1:.2f}")
            st.caption("说明：精确率和召回率的调和平均")
        cols = st.columns([2,3])
        with cols[0]:
        # 混淆矩阵可视化
            st.subheader("混淆矩阵")
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('混淆矩阵',fontsize=18)
            plt.colorbar()
            classes = ['良性', '恶性']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, fontsize=16)
            plt.yticks(tick_marks, classes, fontsize=16)
        
            # 在矩阵中标记数值
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=40)
        
            plt.ylabel('实际类别', fontsize=16)
            plt.xlabel('预测类别', fontsize=16)
            st.pyplot(plt)
        with cols[1]:
            st.subheader("详细分类报告")
            report_dict = classification_report(y_test, y_pred,
                                          target_names=['良性', '恶性'],
                                          output_dict=True)# 将报告转为字典格式
            # 转换为DataFrame
            report_df = pd.DataFrame(report_dict).transpose()

            # 保留必要的列并格式化显示
            report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
            report_df['support'] = report_df['support'].astype(int)  

            # 显示表格
            st.dataframe(report_df.style.format({
                'precision': '{:.4f}',
                'recall': '{:.4f}',
                'f1-score': '{:.4f}'
            }), use_container_width=True)  
            st.info("""
                    👉**accuracy**：整体准确率：所有样本中预测正确的比例。
                    👉**macro avg**：宏平均，直接计算两个类别的指标平均值，用于平衡评估样本量较少的类别（如恶性肿瘤可能样本更少）。
                    👉**weighted avg**：加权平均，按每个类别的样本数量（support）加权计算指标平均值，更贴合实际样本分布的综合评估（样本多的类别影响更大）。
                    """)

        # 知识小测验
        st.subheader("📌 理解混淆矩阵")
        questions = [
            "T1. 在混淆矩阵中，漏诊（恶性肿瘤被误判为良性）的样本数是多少？",
            "T2. 若混淆矩阵中，“实际良性却被预测为恶性” 的数值较高，说明模型存在什么问题？",
            "T3. 混淆矩阵中对角线元素（左上角和右下角）的数值之和代表什么？"
        ]
        options = [["41","2","1","70"],
            [
                "漏诊率高（恶性肿瘤被误判为良性）",
                "误诊率高（良性肿瘤被误判为恶性）",
                "整体准确率低",
                "对良性肿瘤的识别能力强"
            ],
            [
                "所有被错误分类的样本数",
                "所有被正确分类的样本数",
                "实际为良性的总样本数",
                "预测为恶性的总样本数"
            ]
        ]
        correct_answers = [
            "1",
            "误诊率高（良性肿瘤被误判为恶性）",
            "所有被正确分类的样本数"
        ]
        
        # 生成测验选项
        q5_1 = st.radio(questions[0], options[0], key="q5_1", index=None)
        q5_2 = st.radio(questions[1], options[1], key="q5_2", index=None)
        q5_3 = st.radio(questions[2], options[2], key="q5_3", index=None)
        current_answers = [q5_1, q5_2, q5_3]

        # 显示每个问题的即时反馈并记录答题情况
        for i, (q, ans, correct) in enumerate(zip(questions, current_answers, correct_answers)):
            if ans is not None:
                record_answer(5, q, ans, correct, ans == correct)
                if ans == correct:
                    st.success(f"{i+1}. 回答正确")
                else:
                    st.error(f"{i+1}. 回答错误，正确答案是：{correct}")
                    record_error(5, q, ans, correct)
        # 反思输入
        reflection = st.text_input(
            "【反思】在本步骤中，你有什么不太理解的内容？（例如：F1分数）",
            key="step5_reflection",
            autocomplete="off",
        )    
        if reflection:
            st.session_state.logistic_step_records['reflection']['step_5'] = reflection
        
        # 下一步按钮
        all_answered = all(ans is not None for ans in current_answers)
        if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
            st.info("🎉 太棒啦！你已经把各项评估指标的含义和模型表现摸得透透的啦！简直超厉害！")
            if st.button("进入下一步：反思与总结", key="next_step5"):
                complete_step(5)
                st.session_state.step = 6
                st.rerun()
        elif all_answered:
            st.warning("请先回答正确所有问题才能继续")
        else:
            st.info("请完成所有问题的回答")

# 步骤6：反思与总结
def step6():
    st.header("反思与总结")
    st.subheader("目标：梳理逻辑回归完整流程与学习收获")
    st.info("""
    **任务说明**：  
    1. 总结逻辑回归模型的核心原理与应用场景  
    2. 回顾本次实践的关键发现与遇到的问题  
    3. 整理学习收获与未来可探索的方向  
    """)   
   
    # 1. 流程回顾
    st.subheader("📝 完整流程回顾")
    st.info("""
        1. 项目说明：明确乳腺癌诊断的分类任务目标
        2. 数据观察：理解肿瘤特征数据分布与变量划分
        3. 数据预处理：完成训练集/测试集拆分与标准化
        4. 模型构建：实例化逻辑回归分类模型
        5. 训练预测：通过训练数据学习模型参数并进行预测
        6. 模型评估：使用准确率、召回率等指标分析模型表现 
        """)

    
    # 2. 核心结果展示
    st.subheader("📊 模型核心结果摘要")
    if 'y_test' in st.session_state and 'y_pred' in st.session_state:
        accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
        st.info(f"最终模型准确率：{accuracy:.2f}")
        st.info("关键发现：通过特征系数分析，我们识别出对肿瘤良恶性判断影响最大的特征")
    
    # 3. 知识理解测试
    st.subheader("📌 理解测试")
    questions = [
        "T1. 逻辑回归与线性回归的本质区别是什么？",
        "T2. 为什么在分类任务中需要进行特征标准化？",
        "T3. 当模型在测试集上表现不佳时，可能的原因是什么？"
    ]
    options = [
        [
            "损失函数不同（逻辑回归用对数损失，线性回归用均方误差）",
            "逻辑回归只能处理二分类，线性回归只能处理回归",
            "逻辑回归不需要截距项，线性回归需要",
            "逻辑回归不能使用梯度下降优化"
        ],
        [
            "使特征具有相同量纲，避免某一特征主导模型",
            "提高模型运行速度",
            "减少特征数量",
            "使预测结果在[0,1]范围内"
        ],
        [
            "模型过拟合训练数据",
            "特征与目标变量无相关性",
            "训练数据量不足",
            "以上都是"
        ]
    ]
    correct_answers = [
        "损失函数不同（逻辑回归用对数损失，线性回归用均方误差）",
        "使特征具有相同量纲，避免某一特征主导模型",
        "以上都是"
    ]
    
    # 生成测验选项
    q6_1 = st.radio(questions[0], options[0], key="q6_1", index=None)
    q6_2 = st.radio(questions[1], options[1], key="q6_2", index=None)
    q6_3 = st.radio(questions[2], options[2], key="q6_3", index=None)
    current_answers = [q6_1, q6_2, q6_3]
    
    # 4. 学习反思输入
    st.subheader("📌 分析与改进")
    reflection = st.text_input(
        "请总结本次逻辑回归实践的主要收获、遇到的问题及解决方法",
        key="step6_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.logistic_step_records['reflection']['step_6'] = reflection
     
    # 提交与验证逻辑
    if st.button("提交理解测试与我的分析改进意见", key="submit_summary"):
        # 验证测验答案
        quiz_correct = [a == b for a, b in zip(current_answers, correct_answers)]
        all_answered = all(ans is not None for ans in current_answers)
        
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, quiz_correct):
            record_answer(6, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(6, q, ans, correct_ans)
        
        if not all(quiz_correct):
            st.error("理解测试存在错误，请修正后再提交")
            for i, is_correct in enumerate(quiz_correct):
                if not is_correct:
                    st.warning(f"第{i+1}题回答错误，正确答案：{correct_answers[i]}")
        elif not all_answered:
            st.error("请完成所有综合理解测试题")
        elif not reflection.strip():
            st.error("请填写你的分析改进意见")
        else:
            st.session_state.analysis_submitted = True
            st.success("反思与总结提交成功！")
            
    # 完成流程与报告生成逻辑
    if st.session_state.analysis_submitted:
        # 显示完成流程按钮
        if st.button("1.完成全部流程", key="finish_all"):
            complete_step(6)
            st.balloons()
            st.success("🎉 恭喜完成逻辑回归全流程实践！")
            st.info("""
                本次实践总结：
                1. 掌握了逻辑回归分类模型的完整构建流程
                2. 学会了使用准确率、召回率等指标评估分类模型
                3. 理解了特征重要性分析在实际问题中的应用
                            
                后续探索方向：
                - 尝试调整正则化参数优化模型性能
                - 对比不同分类模型（如决策树、SVM）的表现
                - 进行特征选择以简化模型并提高泛化能力
            """)
                
        # 生成报告按钮 - 核心修改点
        if st.button("2.生成逻辑回归分步编程学习报告", key="generate_report"):
            st.session_state.show_report = True  # 切换状态
            st.rerun()  # 刷新页面
        if st.session_state.show_report:
            # 显示报告内容
            report = generate_report_step(
                raw_records=st.session_state.logistic_step_records,steps=6
            )
            st.subheader("📊 逻辑回归分步编程学习报告")
            st.caption(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(report)
            st.session_state.show_report = False
                    

# 主函数
def main():
    st.title("逻辑回归分步实践：乳腺癌诊断分析")
    init_session_state()

    # 侧边栏步骤进度显示
    st.sidebar.title("步骤进度")
    steps = [
        "0. 项目说明",
        "1. 数据观察与理解",
        "2. 数据预处理",
        "3. 构建逻辑回归模型",
        "4. 模型训练与预测",
        "5. 模型评估与改进",
        "6. 反思与总结" 
    ]
    for i, step in enumerate(steps):
        if st.session_state.step > i:
            st.sidebar.markdown(f"✔️ **{step}**")
        elif st.session_state.step == i:
            st.sidebar.markdown(f"🌟 **{step}**")
        else:
            st.sidebar.markdown(f"⭕ {step}")
    
    # 根据当前步骤显示相应内容
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
