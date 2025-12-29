import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json
import time
from datetime import datetime
from learning_report import generate_report_step
# 特征名称中英文映射
FEATURE_NAME_MAP = {
    'MedInc': '收入中位数',
    'HouseAge': '房屋平均年龄',
    'AveRooms': '平均房间数',
    'AveBedrms': '平均卧室数',
    'Population': '人口数',
    'AveOccup': '平均住户人数',
    'Latitude': '纬度',
    'Longitude': '经度'
}

# 初始化会话状态
def init_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'y' not in st.session_state:
        st.session_state.y = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'X_train_scaled' not in st.session_state:
        st.session_state.X_train_scaled = None
    if 'X_test_scaled' not in st.session_state:
        st.session_state.X_test_scaled = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'linear_model' not in st.session_state:
        st.session_state.linear_model = None
    if 'y_pred_linear' not in st.session_state:
        st.session_state.y_pred_linear = None
    if 'nn_model' not in st.session_state:
        st.session_state.nn_model = None
    if 'y_pred_nn' not in st.session_state:
        st.session_state.y_pred_nn = None
    if 'linear_mse' not in st.session_state:
        st.session_state.linear_mse = None
    if 'linear_r2' not in st.session_state:
        st.session_state.linear_r2 = None
    if 'nn_mse' not in st.session_state:
        st.session_state.nn_mse = None
    if 'nn_r2' not in st.session_state:
        st.session_state.nn_r2 = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'chinese_feature_names' not in st.session_state:
        st.session_state.chinese_feature_names = None
    if 'step1_success' not in st.session_state:
        st.session_state.step1_success = False
    if 'analysis_submitted' not in st.session_state:
        st.session_state.analysis_submitted = False
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False

    if 'nn_step_records' not in st.session_state:
        st.session_state.nn_step_records = {
            'step_records': {
                f'step_{i}': {'error_count': 0, 'error_details': []} for i in range(8)
            },
            'total_errors': 0,
            'reflection': {f'step_{i}': '' for i in range(8)}
        }
    
# 初始化记录存储
def init_records():
    if 'nn_step_records' not in st.session_state:
        st.session_state.nn_step_records = {
            'answers': {},       # 存储各步骤答题情况
            'errors': {},        # 存储错误记录
            'reflection': {},   # 存储反思内容
            'analysis': '',      # 存储总结分析
            'progress': 0,       # 完成进度
            'completed_steps': [] # 已完成步骤
        }

# 记录答案
def record_answer(step_num, question, user_answer, correct_answer, is_correct):
    st.session_state.nn_step_records['step_records'][f'step_{step_num}'].setdefault('answers', []).append({
        'question': question,
        'user_answer': user_answer,
        'correct_answer': correct_answer,
        'is_correct': is_correct,
        'time': time.strftime("%Y-%m-%d %H:%M:%S")
    })

# 记录错误
def record_error(step_num, question, user_answer, correct_answer):
    error_info = {
        'question': question,
        'user_answer': user_answer,
        'correct_answer': correct_answer,
        'time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.nn_step_records['step_records'][f'step_{step_num}']['error_count'] += 1
    st.session_state.nn_step_records['step_records'][f'step_{step_num}']['error_details'].append(error_info)
    st.session_state.nn_step_records['total_errors'] += 1


# 标记步骤完成
def complete_step(step_num):
    st.session_state.nn_step_records['step_records'][f'step_{step_num}']['completed'] = True
    st.session_state.nn_step_records['step_records'][f'step_{step_num}']['completed_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

# 步骤0：项目说明
def step0():
    st.subheader("项目说明：神经网络 vs 线性回归（加州房价预测）")
    st.info("""
    **学习目标**
    1. 掌握回归问题的完整解决流程
    2. 理解线性回归与神经网络的原理差异
    3. 学会使用scikit-learn库实现两种模型
    4. 掌握模型评估指标（MSE、R²）的应用
    5. 能够对比分析不同模型的优缺点
    
    **数据集介绍**：
    
    加州房价数据集包含加州各地区的房价中位数以及相关特征，如收入中位数、房屋年龄、平均房间数等，共8个特征，用于预测该地区的房价中位数。
    """)
    # 数据集展示
    # 加载数据集
    housing = fetch_california_housing()
    st.session_state.data = housing
    
    st.subheader("数据集介绍")
    st.write("""
    该数据集包含20640个样本，8个特征，目标变量为房屋中位数价格。
    以下是部分样本数据：
    """)
    
    # 构建特征数据DataFrame，使用中文列名
    df = pd.DataFrame(
        data=housing.data,
        columns=[FEATURE_NAME_MAP[name] for name in housing.feature_names]  # 使用中文特征名称
    )
    # 添加目标值列（房价）
    df['房价（10万美元）'] = housing.target
    
    # 显示前10条数据，隐藏索引列
    st.set_page_config(layout="wide")
    st.dataframe(df.head(10), use_container_width=True)

    # 知识小测验部分
    st.subheader("📌 知识小测验")
    questions = [
        "T1. 在加州房价预测任务中，线性回归与神经网络的核心区别是什么？",
        "T2. 为什么在训练神经网络前需要对加州房价数据集的特征进行标准化处理？",
        "T3. 以下关于加州房价数据集的描述，正确的是？"
    ]
    options = [
        ["线性回归只能处理数值型特征，神经网络可以处理类别型特征",
         "线性回归假设特征与房价呈线性关系，神经网络可捕捉非线性关系",
         "线性回归需要大量数据，神经网络对数据量要求较低",
         "线性回归无法评估模型性能，神经网络可以"],
        
        ["标准化能消除异常值对房价预测的影响",
         "标准化可将所有特征值压缩到[0,1]区间，方便计算",
         "神经网络对特征尺度敏感，标准化能提高训练效率和精度",
         "标准化是 sklearn 库的强制要求，不标准化会报错"],
        
        ["特征包括收入中位数、房屋平均年龄等，目标变量是房价中位数",
         "特征包括房价中位数，目标变量是收入中位数、房屋平均年龄等",
         "经纬度属于目标变量，人口数属于特征",
         "所有数据都是特征，没有目标变量"]
    ]
    correct_answers = ['线性回归假设特征与房价呈线性关系，神经网络可捕捉非线性关系',
                       '神经网络对特征尺度敏感，标准化能提高训练效率和精度',
                       '特征包括收入中位数、房屋平均年龄等，目标变量是房价中位数']    
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
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：特征与目标变量的区别）",
        key="step0_reflection",
        autocomplete="off",
    )    
    if reflection:
        st.session_state.nn_step_records['reflection']['step_0'] = reflection

    # 下一步按钮
    all_answered = all(ans is not None for ans in current_answers)
    if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
        st.info("太棒了！🎉 你已掌握基础概念，准备好深入分析吧！")
        if st.button("进入下一步骤：数据观察与理解", key="next_step0"):
            complete_step(0)
            st.session_state.step = 1
            st.rerun()
    elif all_answered:
        st.warning("请先回答正确所有问题才能继续")
    else:
        st.info("请完成所有问题的回答")

# 步骤1：数据观察与理解
def step1():
    st.header("数据观察与理解")
    st.subheader("目标：加载数据集，观察基本信息及特征相关性")
    
    st.info("""
    **数据集说明**：  
    加州房价数据集包含20640个样本，8个特征，目标变量为房屋中位数价格（单位：10万美元）。
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 加载数据并定义特征中文名称
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

housing = fetch_california_housing()
X = housing.___Q1___  # 特征数据
y = housing.___Q2___  # 目标变量（房价）
feature_names_en = housing.feature_names  # 英文特征名

# 特征名称中英文映射
feature_name_map = {
    'MedInc': '收入中位数',
    'HouseAge': '房屋平均年龄',
    'AveRooms': '平均房间数',
    'AveBedrms': '平均卧室数',
    'Population': '人口数',
    'AveOccup': '平均住户人数',
    'Latitude': '纬度',
    'Longitude': '经度'
}

# 中文特征名称
chinese_feature_names = [feature_name_map[name] for name in feature_names_en]

# 2. 计算特征的统计信息
feature_means = np.mean(X, axis=0)  # 计算列均值
feature_stds = np.std(X, axis=0)  # 计算列标准差
feature_mins = np.min(X, axis=0)  # 计算列最小值
feature_maxs = np.max(X, axis=0)  # 计算列最大值
feature_medians = np.___Q3___(X, axis=0)  # 计算列中位数

print("每个特征的统计信息：")
for i in range(len(chinese_feature_names)):
    print(f"{chinese_feature_names[i]}:")
    print(f"  均值: {feature_means[i]:.4f}")
    print(f"  标准差: {feature_stds[i]:.4f}")
    print(f"  最小值: {feature_mins[i]:.4f}")
    print(f"  最大值: {feature_maxs[i]:.4f}")
    print(f"  中位数: {feature_medians[i]:.4f}")

# 3. 计算特征相关性并绘制热力图
# 合并特征和目标变量用于相关性计算
data_with_target = np.___Q4___((X, y))

# 计算相关系数矩阵（控制行/列变量设置）
correlation = np.corrcoef(data_with_target, rowvar=False)

# 与目标变量的相关性
target_corr = correlation[-1, :-1]  # 排除与自身的相关性
print("与目标变量（房价）的相关性：")
for name, corr in zip(chinese_feature_names, target_corr):
    print(f"{name}: {corr:.4f}")

# 绘制相关性热力图
plt.figure(figsize=(10, 8))
im = plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, label='相关系数')

# 添加特征名称（含目标变量）
names = chinese_feature_names + ['房价']
plt.xticks(range(len(names)), names, rotation=45)
plt.yticks(range(len(names)), names)

# 在热力图上标注相关系数
for i in range(len(names)):
    for j in range(len(names)):
        plt.text(j, i, f"{correlation[i, j]:.2f}", 
                 ha='center', va='center', color='white')

plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 获取特征数据",
            "Q2. 获取目标变量",
            "Q3. 计算特征中位数的函数",
            "Q4. 数组按「列」的方向拼接",
        ]
        options = [
            ["data", "X", "features", "values"],
            ["target", "y", "price", "label"],
            ["mean", "average", "median", "std"],
            ["column_stack", "row_stack", "column_append", "column_sord"],
        ]
        correct_answers = ["data", "target", "median", "column_stack"]
        
        q1_ans = st.selectbox(questions[0], options[0], key="s1_q1", index=None)
        q2_ans = st.selectbox(questions[1], options[1], key="s1_q2", index=None)
        q3_ans = st.selectbox(questions[2], options[2], key="s1_q3", index=None)
        q4_ans = st.selectbox(questions[3], options[3], key="s1_q4", index=None)

        # 相关性系数概念解释
        st.write("#### 皮尔逊相关系数")
        st.info("""
            **核心定义**💥
            - 皮尔逊相关系数是衡量两个连续变量之间线性相关程度的统计指标
            - 取值范围为 **[-1, 1]**
            - 核心反映变量间 **同向 / 反向** 变化的线性紧密程度💪
        """)
        st.info("""
            **数值含义**💥
            - ✅ 系数→1：完全正线性相关（一个变量增长，另一个同步等比例增长）
            - ❌ 系数→-1：完全负线性相关（一个变量增长，另一个同步等比例下降）
            - ➖ 系数→0：无线性相关（变量间无明显线性变化规律，不代表无其他非线性关联）
        """)
        st.info("""
            **补充说明**💥
            - 系数绝对值越接近 1，线性相关性越强；越接近 0，线性相关性越弱
            - 仅衡量线性关系，无法捕捉曲线、分段等非线性关联
            - 对异常值敏感，极端值易扭曲相关系数结果
            -相关性≠因果性：系数显著仅代表变量间有线性关联，不代表一方导致另一方变化
        """)
        st.info("""
            **可视化（热力图）解读**💥
            - 🔴 红色系（系数＞0）：代表正相关，颜色越深（越红），正相关性越强
            - 🔵 蓝色系（系数＜0）：代表负相关，颜色越深（越蓝），负相关性越强
            - ⚪ 浅灰 / 白色（系数≈0）：代表无线性相关。
        """)       
    if 'step1_success' not in st.session_state:
        st.session_state.step1_success = False
    
    if st.button("运行代码", key="run_step1"):
        current_answers = [q1_ans, q2_ans, q3_ans, q4_ans]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(1, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(1, q, ans, correct_ans)
        
        if all(correct):
            st.success("代码运行成功！输出结果：")

            # 加载数据
            housing = fetch_california_housing()
            X = housing.data
            y = housing.target

            chinese_feature_names = [FEATURE_NAME_MAP[name] for name in housing.feature_names]

            # 保存到会话状态
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.feature_names = housing.feature_names
            st.session_state.chinese_feature_names = chinese_feature_names

            # 显示输出结果
            st.subheader("查看特征统计信息表格")
             
            # 计算统计信息
            feature_means = np.mean(X, axis=0)
            feature_stds = np.std(X, axis=0)
            feature_vars = np.var(X, axis=0)
            feature_mins = np.min(X, axis=0)
            feature_maxs = np.max(X, axis=0)
            feature_medians = np.median(X, axis=0)
                
            # 显示特征统计信息表格
            stats_data = {
                "特征名称": chinese_feature_names,
                "均值": [f"{v:.4f}" for v in feature_means],
                "标准差": [f"{v:.4f}" for v in feature_stds],
                "方差": [f"{v:.4f}" for v in feature_vars],
                "最小值": [f"{v:.4f}" for v in feature_mins],
                "最大值": [f"{v:.4f}" for v in feature_maxs],
                "中位数": [f"{v:.4f}" for v in feature_medians]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

              
            # 相关性计算与显示
            data_with_target = np.column_stack((X, y))
            correlation = np.corrcoef(data_with_target, rowvar=False)
            target_corr = correlation[-1, :-1]
                
            corr_data = {
                "特征名称": chinese_feature_names,
                "与房价相关性": [f"{v:.4f}" for v in target_corr]
            }
            st.subheader("与目标变量（房价）的相关性：")
            st.dataframe(pd.DataFrame(corr_data), use_container_width=True)
            st.info("""
                **相关性表格解读**💡 ：
                - 收入中位数与房价呈强正相关（0.6881），是影响房价的核心因素
                - 房屋年龄、平均房间数与房价弱正相关，平均卧室数、人口数、经纬度等与房价仅呈极弱负相关，线性影响几乎可忽略。
            """)
            cols=st.columns([1,5,1])
            with cols[1]:
                # 绘制相关性热力图
                plt.figure(figsize=(10, 8))
                im = plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(im, label='相关系数')
                names = chinese_feature_names + ['房价']
                plt.xticks(range(len(names)), names, rotation=45)
                plt.yticks(range(len(names)), names)
                for i in range(len(names)):
                    for j in range(len(names)):
                        plt.text(j, i, f"{correlation[i, j]:.2f}", 
                                ha='center', va='center', color='white')
                plt.title('特征相关性热力图')
                plt.tight_layout()
                st.pyplot(plt)
            st.info("""
            **相关性热力图解读**💡 ：
            - 平均房间数和平均卧室数呈现深红色🔴（0.85），强正相关
            - 纬度和经度呈现深蓝色🔵（-0.92），强负相关
            - 仅收入中位数与房价为红色🧡（0.69），显著正相关，
            其余特征与房价、特征间多为浅灰 / 浅蓝 / 浅橙🟤，无强线性关联。
            """)
            st.session_state.step1_success = True

        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step1_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：特征相关性）",
        key="step1_reflection",
        autocomplete="off",
    )
    if reflection:
        # 假设已初始化相关状态变量
        st.session_state.nn_step_records['reflection']['step_1'] = reflection
    
    # 下一步按钮
    if st.session_state.step1_success: 
        st.info("哇！✨ 数据观察任务完美完成，太厉害啦！为后续分析打下好基础，继续加油！💪")
        if st.button("进入下一步骤：数据集划分", key="to_step2"):
            complete_step(1)  # 假设已定义该函数
            st.session_state.step = 2
            st.session_state.step1_success = False
            st.rerun()
            

# 步骤2：数据集划分
def step2():
    st.header("数据集划分")
    st.subheader("目标：将数据集划分为训练集和测试集")
    
    if st.session_state.X is None:
        st.warning("请先完成步骤1！")
        st.button("返回步骤1", on_click=lambda: setattr(st.session_state, 'step', 1))
        return
    
    st.info("""
    **任务说明**：
    1. 将数据集划分为训练集（80%）和测试集（20%）
    2. 训练集用于模型训练，测试集用于评估模型泛化能力
    3. 设置random_state保证结果可复现
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 导入数据集划分工具
from sklearn.model_selection import train_test_split

# 2. 划分训练集和测试集（测试集占20%）
X_train, X_test, y_train, y_test = ___Q1___(
    X, y, 
    test_size=___Q2___,  # 测试集比例
    random_state=42  # 随机种子，保证结果可复现
)

# 3. 查看划分后的数据集大小
print("训练集样本数：", X_train.shape[0])
print("测试集样本数：", X_test.shape[0])
print("特征数：", X_train.shape[1])
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 划分训练集与测试集函数",
            "Q2. 测试集占比参数值"
        ]
        options = [
            ["train_test_split", "train_split", "test_split", "data_split"],
            ["0.1", "0.2", "0.3", "0.4"]
        ]
        correct_answers = ["train_test_split", "0.2"]
        
        q1_ans = st.selectbox(questions[0], options[0], key="s2_q1", index=None)
        q2_ans = st.selectbox(questions[1], options[1], key="s2_q2", index=None)

    if 'step2_success' not in st.session_state:
        st.session_state.step2_success = False    

    if st.button("运行代码", key="run_step2"):
        current_answers = [q1_ans, q2_ans]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(2, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(2, q, ans, correct_ans)
        
        if all(correct):

            # 执行数据集划分
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.X, st.session_state.y,
                test_size=0.2,
                random_state=42
            )
           
            # 保存划分后的数据集
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
              
            st.success("数据集划分成功！输出结果：")
                
            # 显示划分结果
            split_data = {
                "数据集类型": ["训练集", "测试集"],
                "样本数量": [X_train.shape[0], X_test.shape[0]],
                "特征数量": [X_train.shape[1], X_test.shape[1]]
            }
            st.dataframe(pd.DataFrame(split_data), use_container_width=True)
                
            st.info("""
                **结果解读**💡：
                - 训练集样本数约为总样本的80%，用于模型训练
                - 测试集样本数约为总样本的20%，用于评估模型泛化能力
                - 特征数量保持一致，与原始数据集特征数相同
            """)

            st.session_state.step2_success = True

        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step2_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：随机种子）",
        key="step2_reflection",
        autocomplete="off",
    )
    if reflection:
        # 假设已初始化相关状态变量
        st.session_state.nn_step_records['reflection']['step_2'] = reflection
    
    # 下一步按钮
    if st.session_state.step2_success: 
        st.info("✨ 数据集划分任务完美完成！成功将数据分为训练集和测试集，为后续模型训练和评估做好了准备，继续加油！💪")
        if st.button("进入下一步骤：特征标准化", key="to_step3"):
            complete_step(2)  # 假设已定义该函数
            st.session_state.step = 3
            st.session_state.step2_success = False
            st.rerun()

# 步骤3：特征标准化
def step3():
    st.header("特征标准化")
    st.subheader("目标：对特征进行标准化处理（尤其对神经网络重要）")
    
    if st.session_state.X_train is None:
        st.warning("请先完成步骤2！")
        st.button("返回步骤2", on_click=lambda: setattr(st.session_state, 'step', 2))
        return
    
    st.info("""
    **任务说明**：
    1. 特征标准化可以使不同量级的特征具有相同的尺度
    2. 对线性回归影响较小，但对神经网络模型非常重要
    3. 使用StandardScaler将特征转换为均值为0，方差为1的分布
    4. 注意：只用训练集拟合标准化器，再分别转换训练集和测试集
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 导入标准化工具
from sklearn.preprocessing import StandardScaler

# 2. 初始化标准化器
scaler = StandardScaler()

# 3. 用训练集拟合标准化器，并转换训练集
X_train_scaled = scaler.___Q1___(X_train)

# 4. 用同样的标准化器转换测试集（不要重新拟合）
X_test_scaled = scaler.___Q2___(X_test)
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 拟合并转换训练集的方法",
            "Q2. 转换测试集的方法（不重新拟合）"
        ]
        options = [
            ["fit", "transform", "fit_transform", "fit_transfer"],
            ["fit", "transform", "fit_transform", "reuse_transform"]
        ]
        correct_answers = ["fit_transform", "transform"]
        
        q1_ans = st.selectbox(questions[0], options[0], key="s3_q1", index=None)
        q2_ans = st.selectbox(questions[1], options[1], key="s3_q2", index=None)
    
    if 'step3_success' not in st.session_state:
        st.session_state.step3_success = False
    
    if st.button("运行代码", key="run_step3"):
        current_answers = [q1_ans, q2_ans]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(3, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(3, q, ans, correct_ans)
        
        if all(correct):

            # 执行标准化流程
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(st.session_state.X_train)
            X_test_scaled = scaler.transform(st.session_state.X_test)
                
            # 保存标准化后的数据
            st.session_state.X_train_scaled = X_train_scaled
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.scaler = scaler
                
            st.success("特征标准化完成！输出结果：")
                
            # 显示所有特征的标准化效果对比
            all_stats = []
            for i, feature_name in enumerate(st.session_state.chinese_feature_names):
                # 计算原始特征的均值和标准差
                orig_mean = st.session_state.X_train[:, i].mean()
                orig_std = st.session_state.X_train[:, i].std()
                
                # 计算标准化后特征的均值和标准差
                scaled_mean = X_train_scaled[:, i].mean()
                scaled_std = X_train_scaled[:, i].std()
                
                # 收集统计信息
                all_stats.append({
                    "特征名称": feature_name,
                    "原始均值": f"{orig_mean:.4f}",
                    "原始标准差": f"{orig_std:.4f}",
                    "标准化后均值": f"{abs(scaled_mean.round(4))}",
                    "标准化后标准差": f"{scaled_std.round(4)}",
                })

            # 显示所有特征的统计信息表格
            st.dataframe(pd.DataFrame(all_stats), use_container_width=True)
            st.info(f"""
                **结果解读**💡：
                - 标准化后特征均值接近0，标准差接近1，符合标准化预期
                - 所有特征将保持原有分布形态，但处于相同量级
                - 测试集使用与训练集相同的标准化参数，保证数据分布一致性
            """)
            # 增加特征分布直方图对比
            st.subheader("🎉 各特征标准化前后分布对比")
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()

            for i, feature_name in enumerate(st.session_state.chinese_feature_names):
                # 绘制原始特征分布
                axes[i].hist(st.session_state.X_train[:, i], bins=30, alpha=0.5, label='原始特征')
                # 绘制标准化后特征分布
                axes[i].hist(X_train_scaled[:, i], bins=30, alpha=0.5, label='标准化后')
                axes[i].set_title(f'{feature_name}', fontsize=12)
                axes[i].legend()
                axes[i].set_xlabel('特征值')
                axes[i].set_ylabel('频数')

            plt.tight_layout()
            st.pyplot(fig)                
            st.info(f"""
                **图表构成**👇 ：每个子图对应一个特征
                - **浅蓝色** 直方图：原始特征的数值分布（横轴为特征值，纵轴为该值出现的频数）
                - **橙色**直方图：标准化后特征的数值分布

                **核心解读**👇 ：
                - 分布形状：标准化后特征的直方图形状与原始特征基本一致（仅左右平移和缩放），说明标准化保留了数据的分布模式
                - 数值范围：原始特征的横轴范围可能差异很大（例如 “人口数” 可能从 0 到几万，“平均房间数” 可能从 1 到 10），标准化后所有特征的数值范围集中在 0 附近
                - 对比意义：验证标准化是否 **只改变尺度，不改变分布** ，确保模型学习的是特征的分布规律而非原始尺度差异
            """)

            # 下一步按钮
            st.session_state.step3_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step3_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：标准化原理）",
        key="step3_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.nn_step_records['reflection']['step_3'] = reflection
    
    # 下一步按钮
    if st.session_state.step3_success: 
        st.info("太棒了！✨ 特征标准化任务顺利完成，这为模型训练做好了关键准备，继续前进吧！💪")
        if st.button("进入下一步骤：线性回归模型", key="to_step4"):
            complete_step(3)
            st.session_state.step = 4
            st.session_state.step3_success = False
            st.rerun()
            
# 步骤4：线性回归模型
def step4():
    st.header("线性回归模型")
    st.subheader("目标：构建并训练线性回归模型")
    
    if st.session_state.X_train_scaled is None:
        st.warning("请先完成步骤3！")
        st.button("返回步骤3", on_click=lambda: setattr(st.session_state, 'step', 3))
        return
    
    st.info("""
    **任务说明**：  
    1. 线性回归是一种简单的回归模型，可作为复杂模型的基准  
    2. 模型表达式：y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b（w为权重，b为偏置）  
    3. 训练模型并在测试集上进行预测，观察特征对房价的线性影响
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 导入线性回归模型
from sklearn.linear_model import ____Q1____

# 2. 实例化线性回归模型
linear_model = LinearRegression()

# 3. 训练模型（使用标准化后的特征）
linear_model.____Q2____(____Q3____, ____Q4____)

# 4. 在测试集上进行预测
y_pred_linear = linear_model.____Q5____(X_test_scaled)
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 用于构建线性回归模型的类",
            "Q2. 训练模型",
            "Q3. 传入标准化后的训练集数据",
            "Q4. 传入的目标变量",
            "Q5. 在测试集上进行预测"
            
        ]
        options = [
            ["LinearRegression", "LogisticRegression", "DecisionTreeRegressor", "SVR"],
            ["fit", "train", "fit_transform", "predict"],
            ["X_train_scaled", "X_test_scaled", "X_train", "y_train_scaled"],
            ["y_train", "y_test", "X_train", "X_test"],
            ["predict", "forecast", "estimate", "calculate"]
        ]
        correct_answers = ["LinearRegression",
                           "fit",
                           "X_train_scaled",
                           "y_train",
                           "predict"]
        
        q4_1 = st.selectbox(questions[0], options[0], key="s4_q1", index=None)
        q4_2 = st.selectbox(questions[1], options[1], key="s4_q2", index=None)
        q4_3 = st.selectbox(questions[2], options[2], key="s4_q3", index=None)
        q4_4 = st.selectbox(questions[3], options[3], key="s4_q4", index=None)    
        q4_5 = st.selectbox(questions[4], options[4], key="s4_q5", index=None)    

    if 'step4_success' not in st.session_state:
        st.session_state.step4_success = False

    if st.button("运行代码", key="run_step4"):
        current_answers = [q4_1, q4_2, q4_3, q4_4, q4_5]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题情况
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(4, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(4, q, ans, correct_ans)
        
        if all(correct):
            # 显示训练完成信息
            st.success("线性回归模型训练完成！输出结果：")
            
            X_train_scaled = st.session_state.X_train_scaled
            X_test_scaled = st.session_state.X_test_scaled
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
                 
            # 训练线性回归模型
            linear_model = LinearRegression()
            linear_model.fit(X_train_scaled, y_train)

            # 生成预测结果
            y_pred_linear = linear_model.predict(X_test_scaled)

            # 计算评估指标
            mse = mean_squared_error(y_test, y_pred_linear)
            r2 = r2_score(y_test, y_pred_linear)

            # 保存结果到会话状态
            st.session_state.linear_model = linear_model
            st.session_state.y_pred_linear = y_pred_linear
            st.session_state.linear_mse = mse
            st.session_state.linear_r2 = r2

            # 显示部分预测结果
            st.subheader("部分预测结果对比：")
            result_data = {
                "实际房价（10万美元）": [f"{y_test[i]:.4f}" for i in range(10)],
                "预测房价（10万美元）": [f"{y_pred_linear[i]:.4f}" for i in range(10)],
                "误差值": [f"{y_test[i]-y_pred_linear[i]:.4f}" for i in range(10)]
            }
            st.dataframe(pd.DataFrame(result_data), use_container_width=True)

            st.session_state.step4_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step4_success = False
    
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：误差值）",
        key="step4_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.nn_step_records['reflection']['step_4'] = reflection

    # 下一步按钮
    if st.session_state.step4_success: 
        st.info("想知道能不能更精准？😉 立刻开启神经网络模型来捕捉更复杂的关系吧！🚀 ")
        if st.button("进入下一步骤：神经网络模型", key="to_step5"):
            complete_step(4)
            st.session_state.step = 5
            st.session_state.step4_success = False
            st.rerun()
            
# 步骤5：神经网络模型
def step5():
    st.header("步骤5：神经网络模型")
    st.subheader("目标：构建并训练神经网络回归模型，捕捉非线性关系")
    
    if 'linear_model' not in st.session_state:
        st.warning("请先完成步骤4！")
        st.button("返回步骤4", on_click=lambda: setattr(st.session_state, 'step', 4))
        return
    
    st.info("""
    **任务说明**：  
    1. 神经网络通过多层非线性变换，可捕捉特征与房价间的复杂关系  
    2. 使用MLPRegressor实现多层感知器，通过隐藏层提取高级特征  
    3. 需设置网络结构、激活函数等关键参数，观察训练过程与预测效果
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 导入神经网络回归模型
from sklearn.neural_network import MLPRegressor

# 2. 实例化神经网络模型
# hidden_layer_sizes指定隐藏层结构，(64, 32)表示2个隐藏层，分别含64和32个神经元
nn_model = MLPRegressor(
    hidden_layer_sizes=___Q1___,  # 隐藏层结构（如(64, 32)）
    activation=___Q2___,  # 激活函数（如'relu'）
    solver='adam',  # 优化器
    max_iter=200,  # 最大迭代次数
    random_state=42,  # 随机种子，保证结果可复现
    verbose=False  # 不打印训练过程
)

# 3. 训练模型（使用标准化后的特征）
nn_model.___Q3___(X_train_scaled, y_train)  # 训练方法

# 4. 在测试集上进行预测
y_pred_nn = nn_model.___Q4___(X_test_scaled)  # 预测方法

# 5. 查看部分预测结果
print("部分预测结果（实际值 vs 神经网络预测值）：")
for i in range(5):
    print(f"实际值: {y_test[i]:.4f}, 预测值: {y_pred_nn[i]:.4f}")

# 6. 绘制神经网络的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(nn_model.___Q5___)  # 损失曲线属性
plt.title('神经网络训练损失曲线')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.grid(True)
plt.show()
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 选择2层隐藏层结构(64, 32)",
            "Q2. 选择relu激活函数",
            "Q3. 模型训练的方法",
            "Q4. 模型预测的方法",
            "Q5. 存储损失曲线的属性"
        ]
        options = [
            ["(64, 32)", "(10)", "(8, 4, 2, 1)", "(1000)"],
            ["'relu'", "'linear'", "'sigmoid'", "'tanh'"],
            ["fit", "train", "fit_transform", "predict"],
            ["predict", "forecast", "estimate", "calculate"],
            ["loss_curve_", "losses_", "error_curve_", "training_loss_"]
        ]
        correct_answers = ["(64, 32)", "'relu'","fit", "predict", "loss_curve_"]
        
        q5_1 = st.selectbox(questions[0], options[0], key="s5_q1", index=None)
        q5_2 = st.selectbox(questions[1], options[1], key="s5_q2", index=None)
        q5_3 = st.selectbox(questions[2], options[2], key="s5_q3", index=None)
        q5_4 = st.selectbox(questions[3], options[3], key="s5_q4", index=None)
        q5_5 = st.selectbox(questions[4], options[4], key="s5_q5", index=None)

    if 'step5_success' not in st.session_state:
        st.session_state.step5_success = False
   
    if st.button("运行代码", key="run_step5"):
        current_answers = [q5_1, q5_2, q5_3, q5_4, q5_5]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题情况
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(5, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(5, q, ans, correct_ans)
        
        if all(correct):
            X_train_scaled = st.session_state.X_train_scaled
            X_test_scaled = st.session_state.X_test_scaled
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
           
            # 神经网络模型定义与训练
            with st.spinner("神经网络训练中，请稍候..."):

                nn_model = MLPRegressor(
                    hidden_layer_sizes=(64, 32),  # 隐藏层结构
                    activation='relu',            # 激活函数
                    solver='adam',                # 优化器
                    max_iter=200,                 # 最大迭代次数
                    random_state=42,              # 随机种子确保可复现
                    verbose=False                 # 不打印训练过程
                )
                
                # 训练模型
                nn_model.fit(X_train_scaled, y_train)
                
                # 模型预测
                y_pred_nn = nn_model.predict(X_test_scaled)
                
                nn_mse = mean_squared_error(y_test, y_pred_nn)  # 均方误差
                nn_r2 = r2_score(y_test, y_pred_nn)
                
                st.session_state.nn_model = nn_model
                st.session_state.y_pred_nn = y_pred_nn
                st.session_state.nn_mse = nn_mse
                st.session_state.nn_r2 = nn_r2
                
                # 绘制损失曲线
                plt.figure(figsize=(10, 6))
                plt.plot(nn_model.loss_curve_)
                plt.title('神经网络训练损失曲线')
                plt.xlabel('迭代次数')
                plt.ylabel('损失值')
                plt.grid(True)
                    
                st.success("神经网络模型训练完成！输出结果：")
                
                # 显示部分预测结果
                st.subheader("部分预测结果对比：")
                result_data = {
                    "实际房价（10万美元）": [f"{y_test[i]:.4f}" for i in range(10)],
                    "神经网络预测值": [f"{y_pred_nn[i]:.4f}" for i in range(10)],
                    "线性回归预测值": [f"{st.session_state.y_pred_linear[i]:.4f}" for i in range(10)],
                    "神经网络误差": [f"{abs(y_test[i]-y_pred_nn[i]):.4f}" for i in range(10)],
                    "线性回归误差": [f"{abs(y_test[i]-st.session_state.y_pred_linear[i]):.4f}" for i in range(10)]
                }
                st.dataframe(pd.DataFrame(result_data), use_container_width=True)
               
                # 显示模型训练信息
                st.subheader("神经网络训练信息：")
                st.info(f"""
                    **最终训练损失**：{nn_model.loss_:.6f}                    
                    **实际迭代次数**：{nn_model.n_iter_}/{nn_model.max_iter}                    
                    **是否收敛**：{'是' if nn_model.n_iter_ < nn_model.max_iter else '否'}
                    """)
                    
                st.subheader("训练损失曲线：")
                cols = st.columns([1, 5, 1])
                with cols[1]:
                    st.pyplot(plt)
                        
                st.info("""
                    **损失曲线解读**💡：  
                    - 曲线持续下降表明模型在不断学习  
                    - 后期趋于平缓说明模型逐渐收敛  
                    - 若曲线波动较大，可尝试减小学习率或增加迭代次数  
                """)
            st.session_state.step5_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step5_success = False
            
    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：优化器）",
        key="step5_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.nn_step_records['reflection']['step_5'] = reflection

    # 下一步按钮
    if st.session_state.step5_success: 
        st.info("神经网络像精密的预测大师，一层层拆解数据的奥秘🛠️，用非线性的智慧捕捉房价背后藏着的复杂密码！🚀 ")
        if st.button("进入下一步骤：模型评估与对比", key="to_step6"):
            complete_step(5)
            st.session_state.step = 6
            st.session_state.step5_success = False
            st.rerun()    


# 步骤6：模型评估与对比
def step6():
    st.header("模型评估与对比")
    st.subheader("目标：评估两种模型的性能并进行对比分析")
    
    if 'nn_model' not in st.session_state:
        st.warning("请先完成步骤5！")
        st.button("返回步骤5", on_click=lambda: setattr(st.session_state, 'step', 5))
        return
    
    st.info("""
    **任务说明**：
    1. 使用均方误差（MSE）和R²分数评估模型性能
    2. 均方误差越小越好，R²分数越接近1越好
    3. 对比线性回归和神经网络的性能差异
    4. 可视化预测结果与实际值的关系
    """)
    
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 导入评估指标
from sklearn.metrics import mean_squared_error, r2_score

# 2. 评估线性回归模型
linear_mse = ___Q1___(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

# 3. 评估神经网络模型
nn_mse = mean_squared_error(y_test, y_pred_nn)
nn_r2 = ___Q2___(y_test, y_pred_nn)

# 4. 打印评估结果
print("线性回归模型评估：")
print(f"均方误差（MSE）：{linear_mse:.4f}")
print(f"R²分数：{linear_r2:.4f}")

print("神经网络模型评估：")
print(f"均方误差（MSE）：{nn_mse:.4f}")
print(f"R²分数：{nn_r2:.4f}")

# 5. 可视化预测结果
plt.figure(figsize=(12, 5))

# 线性回归预测 vs 实际值
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('线性回归：预测值 vs 实际值')
plt.xlabel('实际房价')
plt.ylabel('预测房价')

# 神经网络预测 vs 实际值
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_nn, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('神经网络：预测值 vs 实际值')
plt.xlabel('实际房价')
plt.ylabel('预测房价')

plt.tight_layout()
plt.show()
        """.strip()
        st.code(code_template, language="python")    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 计算线性回归模型均方误差（MSE）",
            "Q2. 计算神经网络模型R²分数",
        ]
        options = [
            ["mean_squared_error", "mean_squared_true", "MSE", "MSE_score"],
            ["r2_score", "r2", "R_score", "R2"],
        ]
        correct_answers = ["mean_squared_error", "r2_score"]
        
        q6_1 = st.selectbox(questions[0], options[0], key="s6_q1", index=None)
        q6_2 = st.selectbox(questions[1], options[1], key="s6_q2", index=None)

    if 'step6_success' not in st.session_state:
        st.session_state.step6_success = False        
        
    if st.button("运行代码", key="run_step6"):
        current_answers = [q6_1, q6_2]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题情况
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(6, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(6, q, ans, correct_ans)

        if all(correct):
            y_test = st.session_state.y_test,
            y_pred_linear = st.session_state.y_pred_linear,
            y_pred_nn = st.session_state.y_pred_nn,
            linear_mse = st.session_state.linear_mse,
            nn_mse = st.session_state.nn_mse,
            linear_r2 = st.session_state.linear_r2
            nn_r2 = st.session_state.nn_r2
            
            st.success("模型评估完成！")
            
            # 显示评估指标对比
            st.subheader("模型评估指标对比：")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                ##### 线性回归 📈  
                ##### 均方误差（MSE）：{linear_mse[-1]:.4f} 🔻  
                ##### R²分数：{linear_r2:.4f} 🔺
                """)

                
            with col2:
                st.info(f"""
                ##### 神经网络 📈  
                ##### 均方误差（MSE）：{nn_mse[-1]:.4f} 🔻  
                ##### R²分数：{nn_r2:.4f} 🔺
                """)
            fig=plt.figure(figsize=(12, 5))                
            plt.subplot(1, 2, 1)
            # 线性回归预测 vs 实际值
            plt.scatter(st.session_state.y_test, y_pred_linear, alpha=0.5)
            plt.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                        [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--')
            plt.title('线性回归：预测值 vs 实际值')
            plt.xlabel('实际房价')
            plt.ylabel('预测房价')
            plt.xlim(0, 5)  
            plt.ylim(0, 12)  
            plt.grid(True)
            # 神经网络预测 vs 实际值
            plt.subplot(1, 2, 2)
            plt.scatter(st.session_state.y_test, y_pred_nn, alpha=0.5)
            plt.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                        [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--')
            plt.title('神经网络：预测值 vs 实际值')
            plt.xlabel('实际房价')
            plt.ylabel('预测房价')
            plt.xlim(0, 5)  
            plt.ylim(0, 12)
            plt.grid(True)            
            st.pyplot(fig)
            st.session_state.step6_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step6_success = False
    
    # 添加反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：R² 或 MSE）",
        key="step6_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.nn_step_records['reflection']['step_6'] = reflection

    # 下一步按钮
    if st.session_state.step6_success: 
        st.info("当数据关系单纯时，线性回归的简洁就是王道✨；当变量纠缠如乱麻，神经网络的深度才显神通🚀—— 没有绝对王者，只有适配场景的智者！")
        if st.button("进入下一步骤：总结与思考", key="to_step7"):
            complete_step(6)
            st.session_state.step = 7
            st.session_state.step5_success = False
            st.rerun()    

# 步骤7：总结与思考
def step7():
    st.header("总结与思考")
    st.subheader("目标：总结两种回归方法的特点，理解神经网络的优势与局限")
    
    # 检查前置条件
    if st.session_state.step < 6:
        st.warning("请先完成前面所有步骤再进行总结！")
        st.button("返回步骤6", on_click=lambda: setattr(st.session_state, 'step', 6))
        return

    st.info("""
    **任务说明**：
    1. 对比线性回归和神经网络在房价预测任务上的表现
    2. 分析两种模型的优缺点和适用场景
    3. 思考如何进一步改进模型性能
    """)
    linear_mse = st.session_state.linear_mse,
    nn_mse = st.session_state.nn_mse,
    linear_r2 = st.session_state.linear_r2
    nn_r2 = st.session_state.nn_r2
    
    # 显示步骤6的评估指标
    st.subheader("📊 模型评估结果回顾")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        ##### 线性回归 📈  
        ##### 均方误差（MSE）：{linear_mse[-1]:.4f} 🔻  
        ##### R²分数：{linear_r2:.4f} 🔺
        """)                
    with col2:
        st.info(f"""
        ##### 神经网络 📈  
        ##### 均方误差（MSE）：{nn_mse[-1]:.4f} 🔻  
        ##### R²分数：{nn_r2:.4f} 🔺
        """)
    
    # 知识理解测试
    st.subheader("📌 理解测试")
    questions = [
        "T1. 线性回归和神经网络的本质区别是什么？",
        "T2. 为什么神经网络通常比线性回归更适合处理复杂非线性关系？",
        "T3. R²分数的含义是什么？"
    ]
    options = [
        [
            "神经网络可以建模非线性关系，线性回归只能建模线性关系",
            "神经网络不需要标准化，线性回归需要标准化",
            "神经网络总是比线性回归更准确",
            "神经网络不需要训练，线性回归需要训练"
        ],
        [
            "神经网络通过激活函数和多层结构实现非线性映射",
            "神经网络参数更多，计算更复杂",
            "神经网络使用梯度下降优化，线性回归使用最小二乘法",
            "神经网络可以自动选择特征，线性回归不能"
        ],
        [
            "表示模型解释的目标变量变异比例，越接近1越好",
            "表示预测值与实际值的平均误差",
            "表示模型的计算复杂度",
            "表示特征之间的相关性"
        ]
    ]
    correct_answers = [
        "神经网络可以建模非线性关系，线性回归只能建模线性关系",
        "神经网络通过激活函数和多层结构实现非线性映射",
        "表示模型解释的目标变量变异比例，越接近1越好"
    ]
    
    # 生成测验选项
    q7_1 = st.radio(questions[0], options[0], key="q7_1", index=None)
    q7_2 = st.radio(questions[1], options[1], key="q7_2", index=None)
    q7_3 = st.radio(questions[2], options[2], key="q7_3", index=None)

    current_answers = [q7_1, q7_2, q7_3]

    # 4. 学习反思输入
    st.subheader("📌 分析与改进")
    reflection = st.text_input(
        "请结合评估指标，思考线性回归模型和神经网络模型的回归效果差异，并分析原因",
        key="step7_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.nn_step_records['reflection']['step_7'] = reflection
         
    # 提交与验证逻辑
    if st.button("提交理解测试与我的分析改进意见", key="submit_kmeans_summary"):
        # 验证测验答案
        quiz_correct = [a == b for a, b in zip(current_answers, correct_answers)]
        all_answered = all(ans is not None for ans in current_answers)
        
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, quiz_correct):
            record_answer(7, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(7, q, ans, correct_ans)      


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
            complete_step(7)
            st.balloons()
            st.success("🎉 恭喜你完成所有步骤！你已成功掌握神经网络回归的完整流程～")
            st.info("""
                本次实践总结：
                - 掌握了神经网络回归模型的完整构建流程（数据预处理→模型定义→训练→预测→评估）
                - 学会了使用均方误差（MSE）、R² 分数等指标评估回归模型性能
                - 理解了特征标准化对神经网络训练的重要性及实现方法
                - 对比了神经网络与线性回归的适用场景，明确了非线性建模的优势
                
                后续探索方向：
                - 尝试不同的神经网络结构（如增加深层数、调整神经元数量）优化性能
                - 探索正则化方法（如 L2 正则、Dropout）解决过拟合问题
                - 对比不同激活函数（如 sigmoid、tanh）对模型效果的影响
                - 结合特征重要性分析，优化输入特征提升神经网络预测精度
                """)
                
        # 生成报告按钮 - 核心修改点
        if st.button("2.生成神经网络分步编程学习报告", key="generate_report"):
            st.session_state.show_report = True  # 切换状态
            st.rerun()  # 刷新页面
        if st.session_state.show_report:
            # 显示报告内容
            report = generate_report_step(
                raw_records=st.session_state.nn_step_records,steps=8
            )
            st.subheader("📊 神经网络分步编程学习报告")
            st.caption(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(report)
            st.session_state.show_report = False
            
# 主程序
def main():
    st.title("📝 神经网络 vs 线性回归")
    st.title("（加州房价预测）")
    # 初始化会话状态
    init_session_state()
    
    # 侧边栏
    st.sidebar.title("步骤进度")
    steps = [
        "0. 项目说明",
        "1. 数据观察", "2. 数据集划分", "3. 特征标准化",
        "4. 线性回归模型", "5. 神经网络模型", "6. 模型评估", "7. 总结与思考"
    ]
    for i, step in enumerate(steps):
        if st.session_state.step > i:
            st.sidebar.markdown(f"✔️ **{step}**")
        elif st.session_state.step == i:
            st.sidebar.markdown(f"🌟 **{step}**")
        else:
            st.sidebar.markdown(f"⭕ {step}")
    
    # 核心：根据当前步骤显示对应主内容
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
    elif st.session_state.step == 7:
        step7()

if __name__ == "__main__":
    main()
