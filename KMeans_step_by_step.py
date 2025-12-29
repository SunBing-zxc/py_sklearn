import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import time
from learning_report import generate_report_step
from datetime import datetime

# 葡萄酒特征英文到中文的映射
feature_names_cn = [
    "酒精含量", "苹果酸含量", "灰分含量", "灰分碱度", 
    "镁含量", "总酚含量", "类黄酮含量", "非黄烷类酚类", 
    "原花青素", "颜色强度", "色调", "稀释葡萄酒的OD280/OD315", 
    "脯氨酸含量"
]
# 初始化会话状态
def init_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'kmeans_step_records' not in st.session_state:
        st.session_state.kmeans_step_records = {
            'step_records': {
                f'step_{i}': {'error_count': 0, 'error_details': []} for i in range(8)
            },
            'total_errors': 0,
            'reflection': {f'step_{i}': '' for i in range(8)}
        }
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'true_labels' not in st.session_state:
        st.session_state.true_labels = None
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'X_scaled' not in st.session_state:
        st.session_state.X_scaled = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'cluster_labels' not in st.session_state:
        st.session_state.cluster_labels = None
    if 'silhouette' not in st.session_state:
        st.session_state.silhouette = 0
    if 'calinski_harabasz' not in st.session_state:
        st.session_state.calinski_harabasz = 0
    if 'X_pca' not in st.session_state:
        st.session_state.X_pca = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None

# 记录答案
def record_answer(step_num, question, user_answer, correct_answer, is_correct):
    st.session_state.kmeans_step_records['step_records'][f'step_{step_num}'].setdefault('answers', []).append({
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
    st.session_state.kmeans_step_records['step_records'][f'step_{step_num}']['error_count'] += 1
    st.session_state.kmeans_step_records['step_records'][f'step_{step_num}']['error_details'].append(error_info)
    st.session_state.kmeans_step_records['total_errors'] += 1

# 标记步骤完成
def complete_step(step_num):
    st.session_state.kmeans_step_records['step_records'][f'step_{step_num}']['completed'] = True
    st.session_state.kmeans_step_records['step_records'][f'step_{step_num}']['completed_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

# 步骤0：项目说明
def step0():
    st.header("项目说明")
    st.subheader("葡萄酒聚类分析")
    
    st.info("""
    **数据集说明**：
    葡萄酒数据集本质上源于UCI 葡萄酒数据集，其划分的 3 类并非抽象标签，而是对应意大利同一地区
    3 种不同品种的葡萄酿造的葡萄酒。这三类葡萄酒的实际差异主要体现在化学成分、感官特性
    （口感 / 风味 / 色泽）和酿造定位上，可结合数据集的 13 个特征（如酒精含量、脯氨酸、类黄酮等）
    具体分析。

        类别1：“高端浓郁”，靠品种的高脯氨酸、高类黄酮，支撑复杂风味和陈年能力。
        类别2：“轻量易饮”，靠品种的低成分积累，主打清新、平价；
        类别3：“高酸果香”，靠品种的高色素和酸度，平衡口感与性价比。
    
    **项目目标**：  
    通过葡萄酒的化学成分特征（如酒精含量、苹果酸含量等），使用KMeans聚类算法对葡萄酒进行分组，
    理解无监督学习中聚类问题的完整流程。
    """)
    
    # 加载数据集
    wine = load_wine()
    st.session_state.raw_dataset = wine
    
    # 数据集展示（使用中文特征名）
    st.subheader("数据集介绍（显示前10条样本数据）")
    df = pd.DataFrame(
        data=wine.data,
        columns=feature_names_cn  # 使用中文特征名作为列名
    )
    df['原始类别'] = wine.target
    st.dataframe(df.head(10), use_container_width=True)
    
    # 知识小测验部分
    st.subheader("📌 知识小测验")
    questions = [
        "T1. 在葡萄酒聚类分析中，KMeans算法的核心作用是什么？",
        "T2. 若用葡萄酒数据集的原始类别（3种葡萄酒）评估KMeans聚类结果，发现聚类标签与原始类别不完全一致，可能的原因是什么？"
    ]
    options = [
        ["根据已知的葡萄酒类别标签（如高端/轻量/高酸）训练预测模型",
         "自动从葡萄酒的13种化学成分特征中发现相似样本的分组规律",
         "计算不同葡萄酒之间的化学成分差异显著性",
         "筛选对葡萄酒分类最关键的特征（如脯氨酸、类黄酮）"],
        
        ["KMeans只能处理2类聚类，无法识别3类数据",
         "聚类是无监督学习，仅根据特征相似度分组，可能与实际品种划分存在差异",
         "葡萄酒的化学成分特征无法区分不同品种",
         "原始类别标签存在错误标注"]
    ]
    correct_answers = ['自动从葡萄酒的13种化学成分特征中发现相似样本的分组规律',
                       '聚类是无监督学习，仅根据特征相似度分组，可能与实际品种划分存在差异']    
    q0_1 = st.radio(questions[0], options[0], key="q0_1", index=None)
    q0_2 = st.radio(questions[1], options[1], key="q0_2", index=None)
    current_answers = [q0_1, q0_2]
    
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
        st.session_state.kmeans_step_records['reflection']['step_0'] = reflection
    
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

# 步骤1：数据观察与理解
def step1():
    st.header("数据观察与理解")
    st.subheader("目标：加载数据集，用numpy观察基本信息")
    
    st.info("""
    **数据集说明**：  
    葡萄酒数据集包含178个样本，13个特征，原始数据分为3类（但聚类时不使用标签）。
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 加载数据并定义特征中文名称
from sklearn.datasets import load_wine
wine = load_wine()
X_raw = wine.data  # 特征数据
true_labels = wine.___Q1___ # 原始标签（聚类时不使用）
feature_names_en = wine.feature_names  # 英文特征名

print("数据形状：", X_raw.shape)  # 提示：使用.shape获取数据维度
print("前3行特征：", X_raw[:3])  # 提示：使用[:3]获取前3行

import numpy as np
# 显示每个特征的均值和方差
feature_means = np.mean(X_raw, ___Q2___=___Q3___)  # 计算列均值
feature_vars = np.___Q4___(X_raw, axis=0)  # 计算列方差

print("每个特征的均值和方差：")
for i in range(len(feature_names_en)):
    print(f"特征 {i+1} [{feature_names_en[i]}]:")
    print(f"  均值: {feature_means[i]:.4f}")
    print(f"  方差: {feature_vars[i]:.4f}")
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 获取原始类别标签",
            "Q2. 完善计算列均值参数",
            "Q3. 完善计算列均值参数",
            "Q4. 计算列方差的函数"
        ]
        options = [
            ["target", "label", "object", "data"],
            ["axis", "ax", "column", "row"],
            ["0", "1", "-1", "None"],
            ["var", "svar", "std", "sqrt"]
        ]
        correct_answers = ["target", "axis", "0", "var"]
        
        q1_ans = st.selectbox(questions[0], options[0], key="s1_q1", index=None)
        q2_ans = st.selectbox(questions[1], options[1], key="s1_q2", index=None)
        q3_ans = st.selectbox(questions[2], options[2], key="s1_q3", index=None)
        q4_ans = st.selectbox(questions[3], options[3], key="s1_q4", index=None)
    
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

            wine = load_wine()
            X_raw = wine.data
            true_labels = wine.target

            st.session_state.data = X_raw
            st.session_state.true_labels = true_labels
            st.session_state.feature_names = feature_names_cn

            with st.expander("查看输出"):
                st.write(f"数据形状：{X_raw.shape}")
                st.write("前3行特征：", X_raw[:3].tolist())
                st.write("前3个特征的均值：", [f"{v:.4f}" for v in np.mean(X_raw, axis=0)[:3]])

                # 特征均值和方差显示
            data = {
                "特征名称":feature_names_cn,
                "均值": np.mean(X_raw, axis=0),
                "方差": np.var(X_raw, axis=0)
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            st.session_state.step1_success = True

        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step1_success = False
    
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：特征均值计算）",
        key="step1_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.kmeans_step_records['reflection']['step_1'] = reflection
    
    if st.session_state.step1_success: 
        st.info("哇！✨ 数据观察任务完美完成，太厉害啦！为后续分析打下好基础，继续加油！💪")
        if st.button("进入下一步：特征数据准备", key="to_step2"):
            complete_step(1)
            st.session_state.step = 2
            st.session_state.step1_success = False
            st.rerun()


# 步骤2：特征数据准备
def step2():
    st.header("特征数据准备")
    st.subheader("目标：提取特征数据并查看原始标签分布")
    
    if st.session_state.data is None:
        st.warning("请先完成步骤1！")
        if st.button("返回步骤1", key="back_to_step1"):
            st.session_state.step = 1
            st.rerun()
        return
    
    st.info("""
    **任务说明**：  
    1. 特征（X）：使用所有13个化学成分特征（X_raw）  
    2. 原始标签（true_labels）：数据集自带的3类标签（0、1、2），仅用于后续对比分析
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 查看原始标签分布（了解数据本来的类别数量）
print("原始标签值：", np.___Q1___(true_labels))

# 统计每个类别的样本数量
print("各类别样本数：", np.___Q2___(true_labels))  

# 查看特征形状
print("X形状：", X_raw.shape)  # 应是(178, 13)
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 查看原始标签分布",
            "Q2. 统计每个类别的样本数量"
        ]
        options = [
            ["unique", "label", "features", "data"],
            ["bincount", "cnt", "count", "length"]
        ]
        correct_answers = ["unique", "bincount"]
        
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

            X = st.session_state.data
            true_labels = st.session_state.true_labels
                
            st.session_state.X = X
                
            st.success("数据准备结果：")
            st.write(f"X形状：{X.shape}")
            st.write(f"原始标签值：{np.unique(true_labels)}")
            st.write(f"各类别样本数：{np.bincount(true_labels)}")
                
            st.session_state.step2_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step2_success = False
    
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：特征数据定义）",
        key="step2_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.kmeans_step_records['reflection']['step_3'] = reflection
    
    if st.session_state.step2_success: 
        st.info("特征数据准备就绪啦🎉,随时准备迎接下一步的标准化挑战🚀")
        if st.button("进入下一步：数据预处理", key="to_step3"):
            complete_step(2)
            st.session_state.step = 3
            st.session_state.step1_success = False
            st.rerun()

# 步骤3：数据预处理
def step3():
    st.header("数据预处理")
    st.subheader("目标：标准化特征（KMeans对特征尺度敏感）")
    
    if st.session_state.X is None:
        st.warning("请先完成步骤2！")
        if st.button("返回步骤2", key="back_to_step2"):
            st.session_state.step = 2
            st.rerun()
        return
    
    st.info("""
    **任务说明**：  
    1. KMeans基于距离计算，需对特征进行标准化（均值为0，方差为1）  
    2. 使用StandardScaler完成标准化处理
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 特征标准化
from sklearn.preprocessing import ___Q1___
scaler = StandardScaler()

# 对特征数据进行标准化
X_scaled = scaler.___Q2___(X_raw)  # 提示：使用fit_transform

# 查看标准化后的均值和方差（应接近0和1）
print("标准化后各特征的均值（应接近0）：", np.mean(X_scaled, axis=0).round(4))
print("标准化后各特征的方差（应接近1）：", np.var(X_scaled, axis=0).round(4))
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 标准化类名",
            "Q2. 标准化方法"
        ]
        options = [
            ["StandardScaler", "MinMaxScaler", "Normalizer", "Standardizer"],
            ["fit_transform", "transform", "fit", "scale"]
        ]
        correct_answers = ["StandardScaler", "fit_transform"]
        
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

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(st.session_state.X)                
            st.session_state.X_scaled = X_scaled                
            st.success("预处理完成！")
            # 创建对比表格
            comparison_df = pd.DataFrame({
                "特征名称": feature_names_cn,
                "标准化前均值": np.mean(st.session_state.X, axis=0).round(4),
                "标准化前方差": np.var(st.session_state.X, axis=0).round(4),
                "标准化后均值": np.mean(X_scaled, axis=0).round(4),
                "标准化后方差": np.var(X_scaled, axis=0).round(4)
            })
            st.dataframe(comparison_df, use_container_width=True)
            
            st.session_state.step3_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step3_success = False
    
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：标准化作用）",
        key="step3_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.kmeans_step_records['reflection']['step_4'] = reflection
    
    if st.session_state.step3_success: 
        st.info("🎉 数据预处理完美收官！所有特征都穿上了 “标准制服”，均值乖乖站回 0 点，方差稳稳锁定 1 值🚀")
        if st.button("进入下一步：数据预处理", key="to_step4"):
            complete_step(3)
            st.session_state.step = 4
            st.session_state.step3_success = False
            st.rerun()

# 步骤4：构建KMeans模型
def step4():
    st.header("构建KMeans模型")
    st.subheader("目标：实例化KMeans聚类模型")
    
    st.info("""
    **任务说明**：  
    1. 从sklearn.cluster导入KMeans  
    2. 实例化模型，设置聚类数n_clusters=3（与原始数据类别数一致）
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 导入KMeans模型
from sklearn.cluster import ___Q1___

# 实例化模型（设置3个聚类，随机种子42保证结果可复现）
model = KMeans(n_clusters = ___Q2___, random_state = 42)

# 查看模型参数
print("模型参数：", model.get_params())
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. KMeans模型类名",
            "Q2. 聚类数量参数值"
        ]
        options = [
            ["KMeans", "KMeansCluster", "KCluster", "KMeansModel"],
            ["3", "2", "4", "5"]
        ]
        correct_answers = ["KMeans", "3"]
        
        q1_ans = st.selectbox(questions[0], options[0], key="s4_q1", index=None)
        q2_ans = st.selectbox(questions[1], options[1], key="s4_q2", index=None)
    
    if 'step4_success' not in st.session_state:
        st.session_state.step4_success = False
    
    if st.button("运行代码", key="run_step4"):
        current_answers = [q1_ans, q2_ans]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(4, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(4, q, ans, correct_ans)
        
        if all(correct):
            model = KMeans(n_clusters=3, random_state=42)
            st.session_state.model = model                
            st.success("模型构建成功！")
            st.write("模型参数：", model.get_params())                
            st.session_state.step4_success = True

        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step4_success = False
    
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：聚类数量选择）",
        key="step4_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.kmeans_step_records['reflection']['step_4'] = reflection
    
    if st.session_state.step4_success: 
        st.info("🚀 KMeans 模型组建完毕啦！聚类核心引擎已启动，下一站训练走起💨！")
        if st.button("进入下一步：模型训练与聚类", key="to_step5"):
            complete_step(4)
            st.session_state.step = 5
            st.session_state.step1_success = False
            st.rerun()

# 步骤5：模型训练与聚类
def step5():
    st.header("模型训练与聚类")
    st.subheader("目标：训练模型并获取聚类结果")
    
    if 'model' not in st.session_state:
        st.warning("请先完成步骤4！")
        if st.button("返回步骤4", key="back_to_step4"):
            st.session_state.step = 4
            st.rerun()
        return
    
    st.info("""
    **任务说明**：  
    1. 用标准化的特征数据训练KMeans模型  
    2. 获取每个样本的聚类标签（0、1、2）
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 训练模型并获取聚类标签
cluster_labels = model.___Q1___(X_scaled)  # 同时完成训练和预测

# 查看聚类结果分布
print("聚类标签值：", np.unique(cluster_labels))  # 应输出[0 1 2]
print("各聚类的样本数：", np.bincount(cluster_labels))  # 统计每个聚类的样本数量

# 对比原始标签与聚类标签的分布差异
print("原始标签分布：", np.bincount(true_labels))
print("聚类标签分布：", np.bincount(cluster_labels))
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 同时完成训练和预测的方法"
        ]
        options = [
            ["fit_predict", "fit_transform", "train_predict", "fit_predict_labels"]
        ]
        correct_answers = ["fit_predict"]
        
        q1_ans = st.selectbox(questions[0], options[0], key="s5_q1", index=None)
    
    if 'step5_success' not in st.session_state:
        st.session_state.step5_success = False
    
    if st.button("运行代码", key="run_step5"):
        current_answers = [q1_ans]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(5, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(5, q, ans, correct_ans)
        
        if all(correct):

            model = st.session_state.model
            X_scaled = st.session_state.X_scaled
            true_labels = st.session_state.true_labels
                
            cluster_labels = model.fit_predict(X_scaled)
            st.session_state.cluster_labels = cluster_labels
                
            st.success("聚类完成！")
            # 计算各类别数量
            original_counts = np.bincount(true_labels)
            cluster_counts = np.bincount(cluster_labels)

            # 创建对比数据框
            comparison_df = pd.DataFrame({
                "类别编号": [f"类别{i}" for i in range(len(original_counts))],
                "原始标签样本数": original_counts,
                "聚类标签样本数": cluster_counts,
                "数量差异": original_counts - cluster_counts  # 新增差异列，直观展示偏差
            })
            st.dataframe(comparison_df, use_container_width=True)                
            st.session_state.step5_success = True

        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step5_success = False
    
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：聚类标签含义）",
        key="step5_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.kmeans_step_records['reflection']['step_5'] = reflection
    
    if st.session_state.step5_success: 
        st.info("🚀 模型训练与聚类开始啦！给每个样本精准贴上聚类标签🏷️")
        if st.button("进入下一步：聚类结果评估与可视化", key="to_step6"):
            complete_step(5)
            st.session_state.step = 6
            st.session_state.step1_success = False
            st.rerun()

# 步骤6：聚类结果评估与可视化
def step6():
    st.header("聚类结果评估与可视化")
    st.subheader("目标：用评估指标和降维可视化分析聚类效果")
    
    if 'cluster_labels' not in st.session_state:
        st.warning("请先完成步骤5！")
        if st.button("返回步骤5", key="back_to_step5"):
            st.session_state.step = 5
            st.rerun()
        return
    
    st.info("""
    **任务说明**：  
    1. 计算轮廓系数（越接近1越好）和Calinski-Harabasz指数（越大越好）  
    2. 用PCA降维到2D，可视化聚类结果与原始标签的对比
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 导入评估指标和PCA
from sklearn.metrics import ___Q1___, ___Q2___
from sklearn.decomposition import ___Q3___
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 计算聚类评估指标
silhouette = silhouette_score(X_scaled, cluster_labels)  # 轮廓系数
calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)  # CH指数

print(f"轮廓系数（越接近1越好）：{silhouette:.4f}")
print(f"Calinski-Harabasz指数（越大越好）：{calinski_harabasz:.4f}")

# PCA降维用于可视化（降到2维）
pca = PCA(n_components=2)
X_pca = pca.___Q4___(X_scaled)  # 对标准化数据进行降维

# 绘制聚类结果与原始标签的对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 聚类结果可视化
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
ax1.set_title('KMeans聚类结果（PCA降维）', fontsize=14)
ax1.set_xlabel('PCA维度1')
ax1.set_ylabel('PCA维度2')

# 原始标签可视化
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='viridis', s=50, alpha=0.8)
ax2.set_title('原始标签分布（PCA降维）', fontsize=14)
ax2.set_xlabel('PCA维度1')
ax2.set_ylabel('PCA维度2')

plt.tight_layout()
plt.show()
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 轮廓系数函数",
            "Q2. Calinski-Harabasz指数函数",
            "Q3. 降维类名",
            "Q4. PCA降维方法"
        ]
        options = [
            ["silhouette_score", "silhouette", "silhouette_index", "cluster_score"],
            ["calinski_harabasz_score", "calinski_score", "harabasz_score", "ch_score"],
            ["PCA", "PCAAnalysis", "PrincipalComponent", "PCADecomposition"],
            ["fit_transform", "transform", "fit", "decompose"]
        ]
        correct_answers = ["silhouette_score", "calinski_harabasz_score", "PCA", "fit_transform"]
        
        q1_ans = st.selectbox(questions[0], options[0], key="s6_q1", index=None)
        q2_ans = st.selectbox(questions[1], options[1], key="s6_q2", index=None)
        q3_ans = st.selectbox(questions[2], options[2], key="s6_q3", index=None)
        q4_ans = st.selectbox(questions[3], options[3], key="s6_q4", index=None)

        st.info("""
        **PCA** 就是个**数据压缩小能手**😊！
        
        📌数据有 13 个特征（比如酒精含量、苹果酸等），像 13 条缠在一起的线，无法画图呈现。

        📌PCA 会挑出 2 条最关键的新线，把 13 个特征的复杂数据**投影**上去，变成简单的 2 个特征（就是代码里的 X_pca）。

        这样就能轻松用散点图看聚类结果啦～ ✨""")
    
    if 'step6_success' not in st.session_state:
        st.session_state.step6_success = False
    
    if st.button("运行代码", key="run_step6"):
        current_answers = [q1_ans, q2_ans, q3_ans, q4_ans]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(6, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(6, q, ans, correct_ans)
        
        if all(correct):

            X_scaled = st.session_state.X_scaled
            cluster_labels = st.session_state.cluster_labels
            true_labels = st.session_state.true_labels
                
            # 计算评估指标
            silhouette = silhouette_score(X_scaled, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
                
            # PCA降维
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
                
            # 保存结果
            st.session_state.silhouette = silhouette
            st.session_state.calinski_harabasz = calinski_harabasz
            st.session_state.X_pca = X_pca
                
            st.success("评估与可视化完成！")
            st.write(f"##### 💡 轮廓系数：{silhouette:.4f}")
            st.write(f"##### 💡 Calinski-Harabasz指数：{calinski_harabasz:.4f}")
                
            # 绘制可视化图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='viridis', s=50, alpha=0.8)
            ax1.set_title('原始标签分布（PCA降维）', fontsize=14)
            ax1.set_xlabel('PCA维度1')
            ax1.set_ylabel('PCA维度2')
                
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
            ax2.set_title('KMeans聚类结果（PCA降维）', fontsize=14)
            ax2.set_xlabel('PCA维度1')
            ax2.set_ylabel('PCA维度2')
                
            plt.tight_layout()
            st.pyplot(fig)
            st.info("""
            ✨ 给两个指标来个 “性格速写”～
            
            **轮廓系数（0.2849）**：像给聚类结果打 “紧凑度 + 分离度” 分！范围是 [-1,1]，越接近 1 说明 “团内亲如一家，团间互不打扰”。现在 0.28 刚过及格线，意思是：每个葡萄酒小团体内部还算抱团，但团体之间边界有点模糊，像挤在一个房间里的三伙人，虽然能看出是三伙，但距离太近啦～

            **Calinski-Harabasz 指数（70.94）**：更像 “聚类明显度” 打分！数值越大，说明团体之间差异越显著（像红葡萄、白葡萄一眼就能分清）。70.94 不算特别高，说明这三类葡萄酒的化学成分差异被聚类捕捉到了一些，但不算特别突出，有点像 “双胞胎穿了不同衣服”—— 能分，但得仔细看～

            总体来说，聚类结果 “能看出是三类”，但不算超清晰！可能是因为有些葡萄酒的化学成分太像啦，让 KMeans 有点 “脸盲”～ 😝""")
            st.session_state.step6_success = True
        else:
            st.error("代码中有错误，请检查填写的内容")
            for i, is_correct in enumerate(correct):
                if not is_correct:
                    st.warning(f"第{i+1}个填空存在错误")
            st.session_state.step6_success = False
    
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：评估指标含义）",
        key="step6_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.kmeans_step_records['reflection']['step_6'] = reflection
    
    if st.session_state.step6_success: 
        st.info("🎉 聚类结果评估与可视化环节顺利通过！这可是检验咱们聚类效果的 “放大镜” 时刻哦～ 🔍")
        if st.button("进入下一步：总结与思考", key="to_step7"):
            complete_step(6)
            st.session_state.step = 7
            st.session_state.step1_success = False
            st.rerun()

# 步骤7：总结与思考
def step7():
    st.header("总结与思考")
    st.subheader("目标：梳理KMeans聚类完整流程与学习收获")
    st.info("""
    **任务说明**：  
    1. 总结KMeans聚类的核心原理与应用场景  
    2. 回顾本次实践的关键发现与遇到的问题  
    3. 整理学习收获与未来可探索的方向  
    """)   
   
    # 1. 流程回顾
    st.subheader("📝 完整流程回顾")
    st.info("""
        1. 项目说明：明确聚类任务目标与数据背景
        2. 数据观察：理解特征分布与数据基本情况
        3. 数据预处理：标准化特征以适应距离计算
        4. 确定K值：通过肘部法等选择合适的聚类数量
        5. 模型训练：使用KMeans进行聚类并获取标签
        6. 结果评估：通过轮廓系数等指标分析聚类效果
        7. 可视化分析：用PCA降维直观展示聚类结果
        """)

    
    # 2. 核心结果展示
    st.subheader("📊 聚类核心结果摘要")
    if 'silhouette' in st.session_state and 'calinski_harabasz' in st.session_state:
        st.success(f"""
        1. **轮廓系数**：{st.session_state.silhouette:.4f}（越接近1越好）
        2. **Calinski-Harabasz指数**：{st.session_state.calinski_harabasz:.4f}（越大越好）
        3. **关键发现**：通过聚类结果与原始标签对比，验证了数据中潜在类别的合理性""")
    
    # 3. 知识理解测试
    st.subheader("📌 理解测试")

    questions = [
        "T1. KMeans聚类与逻辑回归的本质区别是什么？",
        "T2. 为什么KMeans需要对特征进行标准化处理？",
        "T3. 选择K值时，肘部法的原理是？"
    ]
    options = [
        [
            "KMeans是无监督学习（无标签），逻辑回归是监督学习（有标签）",
            "KMeans只能处理数值型数据，逻辑回归可以处理类别型数据",
            "KMeans不需要迭代优化，逻辑回归需要",
            "KMeans只能用于聚类，逻辑回归可以用于聚类和分类"
        ],
        [
            "KMeans基于距离计算，标准化可避免量纲影响",
            "标准化能提高KMeans的迭代速度",
            "标准化可以增加聚类的数量",
            "KMeans要求所有特征均值必须为0"
        ],
        [
            "找到误差开始缓慢下降的拐点作为最佳K值",
            "选择误差最小的K值",
            "选择误差最大的K值",
            "通过特征数量确定K值"
        ]
    ]
    correct_answers = [
        "KMeans是无监督学习（无标签），逻辑回归是监督学习（有标签）",
        "KMeans基于距离计算，标准化可避免量纲影响",
        "找到误差开始缓慢下降的拐点作为最佳K值"
    ]
    
    # 生成测验选项
    q7_1 = st.radio(questions[0], options[0], key="q7_1", index=None)
    q7_2 = st.radio(questions[1], options[1], key="q7_2", index=None)
    q7_3 = st.radio(questions[2], options[2], key="q7_3", index=None)
    current_answers = [q7_1, q7_2, q7_3]

    # 初始化状态变量
    if 'analysis_submitted' not in st.session_state:
        st.session_state.analysis_submitted = False
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False  # 新增报告显示状态

    # 4. 学习反思输入
    st.subheader("📌 分析与改进")
    reflection = st.text_input(
        "请结合聚类评估指标，对模型的聚类效果做出评价和分析，并给出改进意见",
        key="step7_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.kmeans_step_records['reflection']['step_7'] = reflection
         
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
            st.success("🎉 恭喜你完成所有步骤！你已成功掌握KMeans聚类的完整流程～")
            st.info("""
                本次实践总结：
                1. 掌握了KMeans聚类模型的完整构建流程
                2. 学会了使用轮廓系数、CH指数等指标评估聚类效果
                3. 理解了PCA降维在高维数据可视化中的应用
                            
                后续探索方向：
                - 尝试不同的聚类算法（如DBSCAN、层次聚类）对比效果
                - 探索更优的K值选择方法（如轮廓系数法）
                - 结合领域知识对聚类结果进行更深入的解读
                """)
                
        # 生成报告按钮 - 核心修改点
        if st.button("2.生成KMeans分步编程学习报告", key="generate_report"):
            st.session_state.show_report = True  # 切换状态
            st.rerun()  # 刷新页面
        if st.session_state.show_report:
            # 显示报告内容
            report = generate_report_step(
                raw_records=st.session_state.kmeans_step_records,steps=7
            )
            st.subheader("📊 KMeans聚类分步编程学习报告")
            st.caption(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(report)
            st.session_state.show_report = False
# 主程序
def main():
    st.title("📝 KMeans聚类分步编程训练")
    st.subheader("（葡萄酒数据集版）")
  
    init_session_state()
    
    # 侧边栏步骤进度
    st.sidebar.title("步骤进度")
    steps = [
        "0. 项目说明",
        "1. 数据观察", "2. 特征准备", "3. 数据预处理",
        "4. 模型构建", "5. 训练聚类", "6. 结果评估", "7. 总结与思考"
    ]
    for i, step in enumerate(steps):
        if st.session_state.step > i:
            st.sidebar.markdown(f"✔️ **{step}**")
        elif st.session_state.step == i:
            st.sidebar.markdown(f"🌟 **{step}**")
        else:
            st.sidebar.markdown(f"⭕ {step}")
    
    # 显示对应步骤内容
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
