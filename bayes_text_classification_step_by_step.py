# streamlit run bayes_text_classification_step_by_step.py
# 贝叶斯文本分类 - 完整流程

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import time
import os
from datetime import datetime
from learning_report import generate_report_step
import json


# ===================== Session State 管理 =====================
def init_session_state():
    """初始化所有Session State变量"""
    default_state = {
        'step': 0,
        # 答题记录
        'bys_step_records': {
            'step_records': {f'step_{i}': {'error_count': 0, 'error_details': [], 'answers': []} for i in range(8)},
            'total_errors': 0,
            'reflection': {f'step_{i}': '' for i in range(8)}
        },
        # 数据相关
        'X_train_text': None,
        'X_test_text': None,
        'y_train': None,
        'y_test': None,
        # 新增两个变量的初始化
        'analysis_submitted': False,  # 用于标记分析是否提交
        'show_report': False,  # 用于控制报告显示状态
        # 模型相关
        'X_train_tfidf': None,
        'X_test_tfidf': None,
        'model': None,
        'accuracy': None            
    }
    # 只初始化不存在的变量
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

# 记录答案
def record_answer(step_num, question, user_answer, correct_answer, is_correct):
    st.session_state.bys_step_records['step_records'][f'step_{step_num}'].setdefault('answers', []).append({
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
    st.session_state.bys_step_records['step_records'][f'step_{step_num}']['error_count'] += 1
    st.session_state.bys_step_records['step_records'][f'step_{step_num}']['error_details'].append(error_info)
    st.session_state.bys_step_records['total_errors'] += 1

# 标记步骤完成
def complete_step(step_num):
    st.session_state.bys_step_records['step_records'][f'step_{step_num}']['completed'] = True
    st.session_state.bys_step_records['step_records'][f'step_{step_num}']['completed_time'] = time.strftime("%Y-%m-%d %H:%M:%S")


# ===================== 数据加载=====================
# 加载本地20新闻组数据集（适配Streamlit Cloud）
def load_newsgroups_data():
    # 定义数据集路径（使用新的5个主题的JSON文件）
    data_path = os.path.join(os.path.dirname(__file__), "datasets", "20newsgroups_selected.json")
    
    # 读取JSON文件
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # 封装为sklearn数据集格式（增加中文类别名属性）
    class NewsgroupsData:
        def __init__(self, data, target, target_names, chinese_target_names):
            self.data = data
            self.target = target
            self.target_names = target_names  # 英文类别名
            self.chinese_target_names = chinese_target_names  # 新增中文类别名属性
    
    # 构造训练集和测试集（加载中文类别名）
    train_data = NewsgroupsData(
        data=dataset["train"]["data"],
        target=np.array(dataset["train"]["target"]),
        target_names=dataset["train"]["target_names"],
        chinese_target_names=dataset["train"]["chinese_target_names"]
    )
    test_data = NewsgroupsData(
        data=dataset["test"]["data"],
        target=np.array(dataset["test"]["target"]),
        target_names=dataset["test"]["target_names"],
        chinese_target_names=dataset["test"]["chinese_target_names"]
    )
    
    return train_data, test_data

def init_data():
    """初始化数据（将加载的数据存入Session State）"""
    if st.session_state['X_train_text'] is None:
        train_data, test_data = load_newsgroups_data()
        st.session_state['X_train_text'] = train_data.data
        st.session_state['X_test_text'] = test_data.data
        st.session_state['y_train'] = train_data.target
        st.session_state['y_test'] = test_data.target
        st.session_state['chinese_target_names'] = train_data.chinese_target_names
        
# 步骤0：项目说明与数据展示
def step0():
    st.header("项目说明")
    st.subheader("朴素贝叶斯文本分类")
    
    # 项目目标
    st.info("""
    **数据集说明**：
    我们将使用20 Newsgroups数据集的一个子集，包含5个新闻主题类别：
    - rec.sport.baseball（棒球运动）
    - rec.motorcycles（摩托车）
    - sci.space（太空科学）
    - comp.graphics（计算机图形学）
    - talk.politics.misc（政治讨论）
    
    **项目目标**：  
    通过朴素贝叶斯算法对新闻文本进行分类，理解文本分类的完整流程，
    包括文本数据预处理、特征提取、模型训练与评估。
    """)
    
    # 数据集预览（调用缓存函数加载数据）
    st.subheader("数据集预览")
    init_data()  # 确保数据加载并存入Session State
    
    # 安全获取样本数据
    if st.session_state['X_train_text'] is not None:
        # 取前2个样本展示（避免取第9/13个导致困惑）
        sample_texts = st.session_state['X_train_text'][:2]
        sample_targets = st.session_state['y_train'][:2]
        
        st.write("**样本文本示例**：")
        for i, (text, target_idx) in enumerate(zip(sample_texts, sample_targets)):
            cn_name = FEATURE_NAMES_CN[target_idx]
            st.info(f"**样本 {i+1}**（类别：{cn_name}）：{text[:300]}...")
   
    # 知识小测验部分
    st.subheader("📌 知识小测验")
    questions = [
        "T1. 针对本项目的 “新闻文本分类” 场景，选择朴素贝叶斯算法的核心优势不包括？",
        "T2. 完成本项目 “朴素贝叶斯文本分类” 的核心流程，以下步骤排序正确的是？\n① 文本预处理（去除冗余内容、分词等） \n② 模型评估（准确率、混淆矩阵等）\n③ 特征提取（将文本转为 TF-IDF / 词袋特征） \n④ 朴素贝叶斯模型训练"
    ]
    options = [
        ["对高维文本特征（如 TF-IDF 向量）计算效率高，训练速度快",
         "无需大量样本即可训练，适配新闻文本子集的小数据场景",
         "能自动学习文本中的语义关联，处理一词多义问题",
         "模型原理简单，易于解释分类结果的逻辑"],
        
        ["①→③→④→②",
         "①→④→③→②",
         "③→①→④→②",
         "③→④→①→②"]
    ]
    correct_answers = ['能自动学习文本中的语义关联，处理一词多义问题',
                       '①→③→④→②']    
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
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：语义）",
        key="step0_reflection",
        autocomplete="off",
    )    
    if reflection:
        st.session_state.bys_step_records['reflection']['step_0'] = reflection
    
    # 下一步按钮
    all_answered = all(ans is not None for ans in current_answers)
    if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
        st.info("太棒了！🎉 你已掌握基础概念，准备好深入分析吧！")
        if st.button("进入下一步：数据加载", key="next_step0"):
            complete_step(0)
            st.session_state.step = 1
            st.rerun()
    elif all_answered:
        st.warning("请先回答正确所有问题才能继续")
    else:
        st.info("请完成所有问题的回答")


# 步骤1：数据加载
def step1():
    st.header("数据加载")
    st.subheader("目标：加载20 Newsgroups数据集的训练集和测试集")
    
    st.info("""
    **任务说明**：  
    1. 使用fetch_20newsgroups加载指定类别的新闻数据
    2. 分别加载训练集(subset='train')和测试集(subset='test')
    3. 移除邮件头、签名和引用内容，减少噪声
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 导入数据集加载工具
from sklearn.datasets import fetch_20newsgroups

# 2. 选择5个目标新闻主题
target_categories = [
    'rec.sport.baseball',   # 棒球运动
    'rec.motorcycles',      # 摩托车
    'sci.space',            # 太空科学
    'comp.graphics',        # 计算机图形学
    'talk.politics.misc'    # 政治讨论
]

# 3. 加载训练集（用于模型学习）
newsgroups_train = fetch_20newsgroups(
    subset='train',          # 训练集
    categories=target_categories,
    remove=('headers', 'footers', 'quotes'),  # 移除噪声内容
    shuffle=True,            # 打乱数据
    random_state=42          # 固定随机种子，确保结果可复现
)

# 4. 加载测试集（用于模型评估）
newsgroups_test = fetch_20newsgroups(
    subset='test',           # 测试集
    categories=target_categories,
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=42
)

# 5. 查看数据集基本信息
print(f"训练集文本数：{len(newsgroups_train.data)}")
print(f"测试集文本数：{len(newsgroups_test.data)}")
print(f"新闻主题类别：{newsgroups_train.target_names}")
        """.strip()
        st.code(code_template, language="python")
    with right:
        st.info("""
##### 数据集加载代码步骤解释
1. **导入数据集加载工具**：
从 scikit-learn 库导入fetch_20newsgroups函数，用于直接加载 “20 个新闻组” 结构化文本数据。
2. **定义目标新闻类别列表**：
创建包含 5 个指定新闻类别的列表（棒球、摩托车、太空科学、计算机图形学、政治讨论），明确数据集范围，降低训练复杂度。
3. **加载训练集数据**：
加载指定类别的训练集（用于模型学习），同时做预处理：移除页眉、页脚等噪声内容，打乱数据顺序避免无效规律。返回的 Bunch 对象包含文本、类别标签等关键信息。
4. **加载测试集数据**：
加载指定类别的测试集（用于模型评估），参数与训练集一致以保证预处理规则统一。训练集与测试集严格分离，确保评估结果客观有效。
5. **查看数据集基本信息**：
打印训练集 / 测试集文本数量、加载的类别名称，验证数据加载正确性，了解数据集规模。


        """)

    # 会话状态保存运行成功的标志
    if 'step1_success' not in st.session_state:
        st.session_state.step1_success = False
        
    if st.button("运行代码", key="run_step1"):
        # 确保数据已初始化
        init_data()
        # 展示结果（从Session State读取，避免重复加载）
        st.success("代码运行成功！")
        st.info(f"""
        1. 训练集文本数：{len(st.session_state['X_train_text'])}
        2. 测试集文本数：{len(st.session_state['X_test_text'])}
        3. 新闻主题类别：{FEATURE_NAMES_CN}
        """)
        st.session_state.step1_success = True

    current_answers = []
    correct_answers = []
    if st.session_state.step1_success:        
        st.subheader("📌 知识小测验")
        # 定义题目、选项、正确答案
        questions = [
            "T1. 在本项目加载20 Newsgroups数据时，remove=('headers', 'footers', 'quotes') 参数的主要作用是？",
            "T2. 代码中 subset='train' 和 subset='test' 分别加载训练集和测试集，关于两者的作用描述❌错误的是？",
            "T3. 在加载数据时设置 random_state=42，以下说法正确的是？"
        ]
        options = [
            [
                "删除文本中的所有标点符号和数字，只保留纯文字内容",
                "移除新闻文本的页眉、页脚和引用内容，减少无关噪声",
                "过滤掉长度小于指定阈值的短文本样本",
                "将文本统一转换为小写格式，避免大小写干扰"
            ],
            [
                "训练集用于让模型学习文本特征与类别之间的对应关系",
                "测试集用于评估模型在未见过的新数据上的分类能力",
                "训练集和测试集的预处理规则（如remove参数）需保持一致",
                "为了提升模型准确率，可将测试集数据混入训练集一起训练"
            ],
            [
                "42是固定值，修改为其他数字会导致代码报错",
                "固定随机种子，确保每次运行代码数据打乱的结果一致，实验可复现",
                "该参数会控制加载的样本数量，42代表只加载42条文本",
                "该参数仅对训练集生效，对测试集无任何影响"
            ]
        ]
        correct_answers = [
            "移除新闻文本的页眉、页脚和引用内容，减少无关噪声",
            "为了提升模型准确率，可将测试集数据混入训练集一起训练",
            "固定随机种子，确保每次运行代码数据打乱的结果一致，实验可复现"
        ]
            
        # 生成单选按钮（key区分不同题目）
        q1_1 = st.radio(questions[0], options[0], key="q1_1", index=None)
        q1_2 = st.radio(questions[1], options[1], key="q1_2", index=None)
        q1_3 = st.radio(questions[2], options[2], key="q1_3", index=None)
        current_answers = [q1_1, q1_2, q1_3]
            
        # 显示每个问题的即时反馈并记录答题情况
        for i, (q, ans, correct) in enumerate(zip(questions, current_answers, correct_answers)):
            if ans is not None:
                record_answer(1, q, ans, correct, ans == correct)
                if ans == correct:
                    st.success(f"{i+1}. 回答正确 ✅")
                else:
                    st.error(f"{i+1}. 回答错误 ❌，正确答案是：{correct}")
                    record_error(1, q, ans, correct)

    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：语义）",
        key="step1_reflection",
        autocomplete="off",
    )    
    if reflection:
        st.session_state.bys_step_records['reflection']['step_1'] = reflection
    
    # 下一步按钮
    if st.session_state.step1_success:
        all_answered = all(ans is not None for ans in current_answers)
        if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
            st.info("太棒了！🎉 数据集获取成功！")
            if st.button("进入下一步：数据观察与理解", key="to_step1"):
                complete_step(1)
                st.session_state.step = 2
                st.session_state.step1_success = False
                st.rerun()
        elif all_answered:
            st.warning("请先回答正确所有问题才能继续")
        else:
            st.info("请完成所有问题的回答")

# 步骤2：数据观察与理解
def step2():
    st.header("数据观察与理解")
    st.subheader("目标：探索文本数据特征和类别分布")
    
    if st.session_state.X_train_text is None:
        st.warning("请先完成步骤1！")
        st.button("返回步骤1", on_click=lambda: setattr(st.session_state, 'step', 1))
        return
    
    st.info("""
    **任务说明**：  
    1. 提取文本特征和对应标签
    2. 分析训练集和测试集的类别分布
    3. 查看样本文本内容，了解数据特点
    """)
    
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
import matplotlib.pyplot as plt
from collections import Counter 👈
import numpy as np

# 提取特征与标签
X_train_text = newsgroups_train.data  # 训练集文本
X_test_text = newsgroups_test.data    # 测试集文本
y_train = newsgroups_train.target     # 训练集标签
y_test = newsgroups_test.target       # 测试集标签
class_names = newsgroups_train.target_names  # 类别名称

# 统计各类别样本数量
train_class_count = Counter(y_train) 👈
test_class_count = Counter(y_test) 👈

print("训练集类别分布：")
for idx, count in train_class_count.items():
    print(f"{class_names[idx]}: {count}个样本")

print("测试集类别分布：")
for idx, count in test_class_count.items():
    print(f"{class_names[idx]}: {count}个样本")

# 绘制类别分布柱状图
plt.figure(figsize=(12, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题

# 训练集分布
plt.subplot(1, 2, 1)
plt.bar([class_names[idx] for idx in train_class_count.keys()],
        train_class_count.values(), color='skyblue')
plt.title('训练集新闻主题分布')
plt.ylabel('样本数量')

# 测试集分布
plt.subplot(1, 2, 2)
plt.bar([class_names[idx] for idx in test_class_count.keys()],
        test_class_count.values(), color='lightgreen')
plt.title('测试集新闻主题分布')
plt.ylabel('样本数量')

plt.tight_layout()
plt.show()
    """.strip()
    
        st.code(code_template, language="python")
    with right:
        st.info("""
**from collections import Counter** 用于从 Python 标准库的 collections 模块中导入 Counter 类，它是一种专门用于计数可哈希对象的工具，尤其适合统计元素出现的频率。以下是其核心用法介绍📝：
                 """)
        st.write("""
1. **📌基本功能**
Counter 本质上是字典（dict）的子类，它将元素作为键，元素出现的次数作为值，能快速实现元素计数。
2. **📌常用用法--初始化与计数**：
通过传入可迭代对象（如列表、元组、字符串等）创建 Counter 对象，自动统计元素出现次数：
                """)
        
        st.info("""
        - from collections import Counter 
        - **# 统计列表元素** 
        - nums = [1, 2, 2, 3, 3, 3, 4] 
        - count = Counter(nums)
        - print(count)
        """)
        st.write("输出：Counter({3: 3, 2: 2, 1: 1, 4: 1})")
        st.info("""
        - **# 统计字符串字符**
        - text = "hello world"
        - char_count = Counter(text)
        - print(char_count)
        """)
        st.write("输出：Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})")

        
    # 会话状态保存运行成功的标志
    if 'step2_success' not in st.session_state:
        st.session_state.step2_success = False
  
    if st.button("运行代码", key="run_step2"):      
        st.success("数据观察完成！")
        st.session_state.step2_success = True 
        # 显示类别分布图表
        st.subheader("类别分布：")
        train_class_count = Counter(st.session_state['y_train'])  # 使用标签列表计算分布
        test_class_count = Counter(st.session_state['y_test'])
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
        # 训练集分布
        ax1.bar(train_class_count.keys(), train_class_count.values(), color='skyblue')
        ax1.set_title('训练集新闻主题分布')
        ax1.set_xticks(list(train_class_count.keys()))
        ax1.set_xticklabels(
            [FEATURE_NAMES_CN[i] for i in train_class_count.keys()],
            rotation=45, 
            ha='right'
        )
        ax1.set_ylabel('样本数量')
            
        # 测试集分布
        ax2.bar(test_class_count.keys(), test_class_count.values(), color='lightgreen')
        ax2.set_title('测试集新闻主题分布')
        ax2.set_xticks(list(test_class_count.keys()))
        ax2.set_xticklabels(
            [FEATURE_NAMES_CN[i] for i in test_class_count.keys()],
            rotation=45, 
            ha='right'
        )
        ax2.set_ylabel('样本数量')            
        plt.tight_layout()
        st.pyplot(fig)
        
    if st.session_state.step2_success:        
        st.subheader("📌 知识小测验")
        # 定义题目、选项、正确答案
        questions = [
            "T1. 本步骤中使用Counter(y_train)统计类别分布，关于Counter的作用描述正确的是？",
            "T2. 在绘制类别分布柱状图时，设置plt.rcParams['font.sans-serif'] = ['SimHei']的目的是？",
            "T3. 分析训练集和测试集的类别分布，主要是为了检查什么问题？"
        ]
        options = [
            [
                "对文本内容进行分词并统计关键词出现频率",
                "计数可迭代对象中元素出现的次数（如不同类别标签的样本数）",
                "将文本数据转换为数值特征矩阵",
                "计算不同类别之间的相似度"
            ],
            [
                "调整图表的尺寸大小，使其更适合展示",
                "设置中文显示字体，避免中文乱码问题",
                "将坐标轴刻度转换为整数格式",
                "改变柱状图的颜色和样式"
            ],
            [
                "检查数据是否存在类别不平衡问题（部分类别样本过多/过少）",
                "直接模型在测试集上的准确率",
                "计算文本的平均长度",
                "查看不同类别文本的关键词差异"
            ]
        ]
        correct_answers = [
            "计数可迭代对象中元素出现的次数（如不同类别标签的样本数）",
            "设置中文显示字体，避免中文乱码问题",
            "检查数据是否存在类别不平衡问题（部分类别样本过多/过少）"
        ]
            
        # 生成单选按钮（key区分不同题目）
        q2_1 = st.radio(questions[0], options[0], key="q2_1", index=None)
        q2_2 = st.radio(questions[1], options[1], key="q2_2", index=None)
        q2_3 = st.radio(questions[2], options[2], key="q2_3", index=None)
        current_answers = [q2_1, q2_2, q2_3]
            
        # 显示每个问题的即时反馈并记录答题情况
        for i, (q, ans, correct) in enumerate(zip(questions, current_answers, correct_answers)):
            if ans is not None:
                record_answer(2, q, ans, correct, ans == correct)
                if ans == correct:
                    st.success(f"{i+1}. 回答正确 ✅")
                else:
                    st.error(f"{i+1}. 回答错误 ❌，正确答案是：{correct}")
                    record_error(2, q, ans, correct)

    # 反思输入（保持原逻辑）
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：哈希对象）",
        key="step2_reflection",
        autocomplete="off",
    )    
    if reflection:
        st.session_state.bys_step_records['reflection']['step_2'] = reflection

    # 下一步按钮逻辑（补充）
    if st.session_state.step2_success:
        all_answered = all(ans is not None for ans in current_answers)
        if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
            st.info("太棒了！🎉 数据观察与理解环节大功告成啦！📊 我们不仅清晰地看到了文本数据的类别分布，还通过图表直观地掌握了数据特征，为后续的分析打下了超棒的基础呢！")
            if st.button("进入下一步：文本特征提取", key="to_step2"):
                complete_step(2)
                st.session_state.step = 3
                st.session_state.step2_success = False
                st.rerun()
        elif all_answered:
            st.warning("请先回答正确所有问题才能继续")
        else:
            st.info("请完成所有问题的回答")



# 步骤3：文本特征提取
def step3():
    st.header("文本特征提取")
    st.subheader("目标：使用TF-IDF将文本转换为数值特征")
    
    if st.session_state.X_train_text is None:
        st.warning("请先完成步骤2！")
        st.button("返回步骤2", on_click=lambda: setattr(st.session_state, 'step', 2))
        return
    
    st.info("""
    **任务说明**：  
    1. 使用TF-IDF方法将文本转换为数值特征
    2. 训练集使用fit_transform，测试集使用transform
    3. 移除停用词并限制最大特征数量，优化特征质量
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 导入TF-IDF特征提取工具
from sklearn.feature_extraction.text import TfidfVectorizer

# 2. 初始化TF-IDF转换器
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',  # 移除英语停用词（如"the"、"and"等无实际语义的词）
    max_features=5000,     # 仅保留5000个最常见词，控制特征维度
    min_df=5               # 忽略在少于5篇文本中出现的词
)

# 3. 对训练集文本进行"拟合+转换"
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
# 4. 对测试集文本仅"转换"（使用训练集的词表规则）
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# 5. 查看TF-IDF特征结构
print(f"训练集TF-IDF矩阵形状：{X_train_tfidf.shape}")  # (样本数, 特征数)
print(f"TF-IDF词表大小：{len(tfidf_vectorizer.vocabulary_)}")
print(f"前10个关键词示例：{list(tfidf_vectorizer.vocabulary_.keys())[:10]}")
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.info("""
##### TF-IDF特征提取核心原理
TF-IDF（词频-逆文档频率）是文本特征提取的经典方法，核心思想是：
- **词频(TF)**：词语在当前文本中出现的频率（值越高越重要）
- **逆文档频率(IDF)**：词语在所有文本中出现的频率倒数（值越高说明该词越稀有，区分度越强）

最终通过两者乘积，让"重要且稀有"的词获得更高权重。
        """)


    # 会话状态保存运行成功的标志
    if 'step3_success' not in st.session_state:
        st.session_state.step3_success = False
        
    if st.button("运行代码", key="run_step3"):
        # 执行特征提取
        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            min_df=5
        )
        X_train_tfidf = tfidf_vectorizer.fit_transform(st.session_state.X_train_text)
        X_test_tfidf = tfidf_vectorizer.transform(st.session_state.X_test_text)
            
        # 保存结果到会话状态
        st.session_state.X_train_tfidf = X_train_tfidf
        st.session_state.X_test_tfidf = X_test_tfidf
        st.session_state.tfidf_vectorizer = tfidf_vectorizer
            
        st.success("特征提取完成！")
        st.info(f"""
        1. 训练集TF-IDF矩阵形状：{X_train_tfidf.shape}
        2. TF-IDF词表大小：{len(tfidf_vectorizer.vocabulary_)}
        3. 前10个关键词示例：{list(tfidf_vectorizer.vocabulary_.keys())[:10]}
        """)
        st.session_state.step3_success = True

    current_answers = []
    correct_answers = []
    if st.session_state.step3_success:        
        st.subheader("📌 知识小测验")
        # 定义题目、选项、正确答案
        questions = [
            "T1. 关于TF-IDF中TF（词频）和IDF（逆文档频率）的描述，正确的是？",
            "T2. 为什么对测试集文本使用transform()而不是fit_transform()？",
            "T3. 参数max_features=5000的作用是？"
        ]
        options = [
            [
                "TF值越高说明词语越稀有，IDF值越高说明词语在当前文本中越重要",
                "TF值越高说明词语在当前文本中出现越频繁，IDF值越高说明词语在所有文本中出现越稀少",
                "TF和IDF都是值越高越好，两者乘积越大代表词语重要性越低",
                "TF只计算词语在单篇文本中的频率，IDF只计算词语在训练集中的总出现次数"
            ],
            [
                "transform()运行速度更快，适合大规模测试集",
                "避免测试集的词表污染训练集学到的规律，确保特征空间一致",
                "测试集数据量通常较小，不需要fit操作",
                "transform()能自动处理缺失值，而fit_transform()不能"
            ],
            [
                "只保留在至少5000篇文本中出现过的词",
                "将文本统一截断或补齐到5000个字符长度",
                "限制词表最大规模为5000个词，防止特征维度过高",
                "要求每个文本至少包含5000个不同的词语"
            ]
        ]
        correct_answers = [
            "TF值越高说明词语在当前文本中出现越频繁，IDF值越高说明词语在所有文本中出现越稀少",
            "避免测试集的词表污染训练集学到的规律，确保特征空间一致",
            "限制词表最大规模为5000个词，防止特征维度过高"
        ]
            
        # 生成单选按钮（key区分不同题目）
        q3_1 = st.radio(questions[0], options[0], key="q3_1", index=None)
        q3_2 = st.radio(questions[1], options[1], key="q3_2", index=None)
        q3_3 = st.radio(questions[2], options[2], key="q3_3", index=None)
        current_answers = [q3_1, q3_2, q3_3]
            
        # 显示每个问题的即时反馈并记录答题情况
        for i, (q, ans, correct) in enumerate(zip(questions, current_answers, correct_answers)):
            if ans is not None:
                record_answer(3, q, ans, correct, ans == correct)
                if ans == correct:
                    st.success(f"{i+1}. 回答正确 ✅")
                else:
                    st.error(f"{i+1}. 回答错误 ❌，正确答案是：{correct}")
                    record_error(3, q, ans, correct)

    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：TF-IDF原理）",
        key="step3_reflection",
        autocomplete="off",
    )    
    if reflection:
        st.session_state.bys_step_records['reflection']['step_3'] = reflection
    
    # 下一步按钮
    if st.session_state.step3_success:
        all_answered = all(ans is not None for ans in current_answers)
        if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
            st.info("哇塞！🎉 文本特征特征提取顺利通关！🥳 文字现在都变成了闪闪发光的数字特征✨")
            if st.button("进入下一步：构建朴素贝叶斯模型", key="to_step3"):
                complete_step(3)
                st.session_state.step = 4
                st.session_state.step3_success = False
                st.rerun()
        elif all_answered:
            st.warning("请先回答正确所有问题才能继续")
        else:
            st.info("请完成所有问题的回答")


# 步骤4：构建贝叶斯模型
def step4():
    st.header("构建贝叶斯模型")
    st.subheader("目标：实例化多项式朴素贝叶斯分类模型")
    
    if st.session_state.X_train_tfidf is None:
        st.warning("请先完成步骤3！")
        st.button("返回步骤3", on_click=lambda: setattr(st.session_state, 'step', 3))
        return
    
    st.info("""
    **任务说明**：  
    1. 导入MultinomialNB模型
    2. 实例化多项式朴素贝叶斯模型
    3. 了解模型参数含义，尤其是平滑系数alpha的作用
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 导入多项式朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB

# 2. 初始化模型（alpha为平滑系数，防止概率为0）
# alpha参数说明：
# - alpha=1.0：完全拉普拉斯平滑
# - alpha→0+：接近无平滑（可能出现零概率）
# - alpha增大：平滑效果增强，模型泛化能力提升但可能欠拟合

model = MultinomialNB(
    alpha=1.0,           # 拉普拉斯平滑系数，避免出现零概率
    fit_prior=True,      # 是否学习先验概率，默认True
    class_prior=None     # 自定义类的先验概率，默认None表示从数据中学习
)

# 3. 查看模型参数
print("模型参数：", model.get_params())
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.info("""
##### 多项式朴素贝叶斯原理
MultinomialNB是适用于离散特征（如词频计数）的朴素贝叶斯变种，核心特点：
- **平滑机制**：通过alpha参数实现拉普拉斯平滑，解决"零概率"问题（当某个词在训练集中未出现时）
- **先验概率**：默认从训练数据中学习各类别的先验概率（样本占比）
- **文本适配性**：特别适合处理文本分类任务中的词频/TF-IDF特征
        """)


    # 会话状态保存运行成功的标志
    if 'step4_success' not in st.session_state:
        st.session_state.step4_success = False
        
    if st.button("运行代码", key="run_step4"):
        # 执行模型构建
        model = MultinomialNB(
            alpha=1.0,
            fit_prior=True,
            class_prior=None
        )
        
        # 保存结果到会话状态
        st.session_state.model = model            
        st.success("模型构建完成！")
        st.info(f"""
        模型参数：{model.get_params()}
        """)
        st.session_state.step4_success = True
        
    current_answers = []
    correct_answers = []
    if st.session_state.step4_success:        
        st.subheader("📌 知识小测验")
        # 定义题目、选项、正确答案
        questions = [
            "T1. 多项式朴素贝叶斯中，alpha参数的主要作用是？",
            "T2. 为什么朴素贝叶斯算法特别适合处理文本分类任务？",
            "T3. 当fit_prior=False时，模型会如何处理先验概率？"
        ]
        options = [
            [
                "控制模型训练的迭代次数，防止过拟合",
                "实现拉普拉斯平滑，避免因某个词未出现导致的零概率问题",
                "设置特征的最大数量，减少计算复杂度",
                "调整学习率，加快模型收敛速度"
            ],
            [
                "能自动理解文本语义，处理同义词和多义词",
                "对高维稀疏特征（如文本TF-IDF）计算高效，且需要样本量小",
                "不需要特征提取步骤，可以直接处理原始文本",
                "在所有文本分类任务中准确率都高于其他算法"
            ],
            [
                "使用均匀分布作为先验概率（各类别概率相等）",
                "忽略先验概率，只使用似然概率进行预测",
                "会报错，因为必须从数据中学习先验概率",
                "自动设置先验概率与训练集中类别比例一致"
            ]
        ]
        correct_answers = [
            "实现拉普拉斯平滑，避免因某个词未出现导致的零概率问题",
            "对高维稀疏特征（如文本TF-IDF）计算高效，且需要样本量小",
            "使用均匀分布作为先验概率（各类别概率相等）"
        ]
            
        # 生成单选按钮（key区分不同题目）
        q4_1 = st.radio(questions[0], options[0], key="q4_1", index=None)
        q4_2 = st.radio(questions[1], options[1], key="q4_2", index=None)
        q4_3 = st.radio(questions[2], options[2], key="q4_3", index=None)
        current_answers = [q4_1, q4_2, q4_3]
            
        # 显示每个问题的即时反馈并记录答题情况
        for i, (q, ans, correct) in enumerate(zip(questions, current_answers, correct_answers)):
            if ans is not None:
                record_answer(4, q, ans, correct, ans == correct)
                if ans == correct:
                    st.success(f"{i+1}. 回答正确 ✅")
                else:
                    st.error(f"{i+1}. 回答错误 ❌，正确答案是：{correct}")
                    record_error(4, q, ans, correct)

    # 反思输入
    reflection = st.text_input(
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：拉普拉斯平滑）",
        key="step4_reflection",
        autocomplete="off",
    )    
    if reflection:
        st.session_state.bys_step_records['reflection']['step_4'] = reflection
    
    # 下一步按钮
    if st.session_state.step4_success:
        all_answered = all(ans is not None for ans in current_answers)
        if all_answered and all(a == b for a, b in zip(current_answers, correct_answers)):
            st.info("太棒了！🎉 朴素贝叶斯模型构建完成！")
            if st.button("进入下一步：模型训练", key="to_step4"):
                complete_step(4)
                st.session_state.step = 5
                st.session_state.step4_success = False
                st.rerun()
        elif all_answered:
            st.warning("请先回答正确所有问题才能继续")
        else:
            st.info("请完成所有问题的回答")


# 步骤5：模型训练
def step5():
    st.header("模型训练")
    st.subheader("目标：用训练集数据训练朴素贝叶斯模型")
    
    if st.session_state.model is None:
        st.warning("请先完成步骤4！")
        st.button("返回步骤4", on_click=lambda: setattr(st.session_state, 'step', 4))
        return
    
    st.info("""
    **任务说明**：  
    1. 使用训练集的TF-IDF特征和标签训练模型
    2. 分析模型学到的主题-关键词关联
    3. 理解朴素贝叶斯模型的概率学习机制
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 用训练集的TF-IDF特征与标签训练模型
model.fit(___Q1___, ___Q2___)  # 填入训练特征和训练标签

# 2. 获取词表特征名称（用于解释模型）
feature_names = tfidf_vectorizer.get_feature_names_out()

# 3. 查看模型学到的"主题-关键词"关联
print("各主题的核心关键词（概率最高的前5个）：")
for class_idx, class_name in enumerate(class_names): 
    # 提取该主题下概率最高的5个词的索引（feature_log_prob_存储对数概率）
    top_word_idx = model.feature_log_prob_[class_idx].argsort()[___Q3___]  # 补充切片参数
    # 映射为词名
    top_words = [feature_names[idx] for idx in top_word_idx]
    print(f"{class_name}：{top_words}")
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 模型训练需要传入的训练数据特征",
            "Q2. 模型训练需要传入的训练数据标签",
            "Q3. 获取概率最高的5个词的索引"
        ]
        options = [
            ["X_test_tfidf", "X_train_tfidf", "X_train_text", "tfidf_vectorizer"],
            ["y_train", "y_test", "model", "tfidf_vectorizer"],
            ["-5:", ":5", "5:", "-5:-1"]
        ]
        correct_answers = ["X_train_tfidf", "y_train", "-5:"]
        
        q5_1 = st.selectbox(questions[0], options[0], key="fill_1", index=None)
        q5_2 = st.selectbox(questions[1], options[1], key="fill_2", index=None)
        q5_3 = st.selectbox(questions[2], options[2], key="fill_3", index=None)

    # 会话状态保存运行成功的标志
    if 'step5_success' not in st.session_state:
        st.session_state.step5_success = False
        
    if st.button("运行代码", key="run_step5"):
        current_answers = [q5_1, q5_2,q5_3]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(5, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(5, q, ans, correct_ans)
        
        if all(correct):
            # 执行模型训练
            st.session_state.model.fit(
                st.session_state.X_train_tfidf, 
                st.session_state.y_train
            )
                
            # 获取特征名称
            feature_names = st.session_state.tfidf_vectorizer.get_feature_names_out()
            st.success("模型训练完成！")
            
            # 定义5种不同的颜色（可根据需要调整）
            colors = [
                '#FF6B6B',  # 红色系
                '#4ECDC4',  # 青绿色系
                '#45B7D1',  # 蓝色系
                '#FFA07A',  # 浅橙色
                '#98D8C8'   # 薄荷绿
            ]
            for class_name in FEATURE_NAMES_CN:  # 使用中文类别名
                class_idx = FEATURE_NAMES_CN.index(class_name)
                top_word_idx = st.session_state.model.feature_log_prob_[class_idx].argsort()[-5:]  # 补充切片参数
                top_words = [feature_names[idx] for idx in top_word_idx]
                    
                # 可视化关键词重要性
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(top_words,
                        st.session_state.model.feature_log_prob_[class_idx][top_word_idx],
                        color=colors[class_idx])
                ax.set_title(f'{class_name} 核心关键词',fontsize=16)
                ax.set_xlabel('对数概率（值越高越重要）',fontsize=16)
                ax.tick_params(axis='y', labelsize=16)  # y轴刻度标签字体大小
                
                # 根据根据索引判断布局位置
                if class_idx < 2:  # 前2个放第一行
                    if class_idx == 0:
                        cols1 = st.columns(2)  # 只创建一次第一行列布局
                    cols1[class_idx].pyplot(fig)
                elif 2 <= class_idx < 4:  # 中间2个放第二行
                    if class_idx == 2:
                        cols2 = st.columns(2)  # 只创建一次第二行列布局
                    cols2[class_idx - 2].pyplot(fig)
                else:  # 最后1个放第三行（居中）
                    cols3 = st.columns(2)  # 居中布局
                    cols3[0].pyplot(fig)
                    cols3[1].info("""
1. **横坐标为什么为负**❓ 
横坐标展示的是**特征对数概率**，数值为负是因为概率的取值范围是 0 < p ≤ 1，对 0~1 之间的数取自然对数（ln），结果必然是负数。
                            """)
                    cols3[1].info("""
2. **如何判断关键词的 “关键性大小”**❓
判断关键词对该类别的重要程度，核心看横坐标的数值 “离 0 越近（越小的负数），关键性越强”。
👉直观判断：在条形图中，条形越长（越向右延伸），关键词越关键（因为越长代表数值越接近 0，对数概率越大）。
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
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：对数概率）",
        key="step5_reflection",
        autocomplete="off",
    )    
    if reflection:
        st.session_state.bys_step_records['reflection']['step_5'] = reflection
    
    # 下一步按钮
    if st.session_state.step5_success:
        st.info("太棒了！🎉 模型训练完成，我们已经掌握了文本特征和类别的隐藏规律！")
        if st.button("进入下一步：模型评估与可视化", key="to_step5"):
            complete_step(5)
            st.session_state.step = 6
            st.session_state.step5_success = False
            st.rerun()



# 步骤6：模型评估与结果分析
def step6():
    st.header("模型评估与结果分析")
    st.subheader("目标：评估模型性能并分析分类错误原因")
    
    st.info("""
    **任务说明**：  
    1. 计算模型在测试集上的准确率
    2. 生成详细分类报告（精确率、召回率、F1值）
    3. 分析错误分类样本，总结模型不足
    """)
    
    left, mid, right = st.columns([13, 0.2, 6])
    
    with left:
        code_template = """
# 1. 导入评估指标工具
from sklearn.metrics import ___Q1___, ___Q2___

# 用训练好的模型预测测试集文本类别
y_pred = model.predict(X_test_tfidf)

# 2. 计算准确率（所有预测正确的样本占比）
accuracy = accuracy_score(___Q3___, ___Q4___)
print(f"模型准确率：{accuracy:.4f}")

# 3. 生成详细分类报告
report = classification_report(
    y_test, 
    y_pred,
    target_names=class_names
)
print("分类详细报告：")
print(report)

# 4. 分析错误分类样本
error_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
print(f"错误分类样本数：{len(error_indices)}")
        """.strip()
        st.code(code_template, language="python")
    
    with right:
        st.write("请选择正确的代码片段填空:")
        questions = [
            "Q1. 用于计算准确率的函数",
            "Q2. 用于生成分类报告的函数",
            "Q3. 计算准确率时需要的真实标签",
            "Q4. 计算准确率时需要的预测标签"
        ]
        options = [
            ["accuracy_score", "precision_score", "recall_score", "f1_score"],
            ["confusion_matrix", "classification_report", "roc_auc_score", "mean_squared_error"],
            ["y_train", "y_test", "y_pred", "X_test"],
            ["y_pred", "y_train", "X_pred", "y_true"]
        ]
        correct_answers = [
            "accuracy_score", 
            "classification_report", 
            "y_test", 
            "y_pred"
        ]
        
        q6_1 = st.selectbox(questions[0], options[0], key="fill_q1", index=None)
        q6_2 = st.selectbox(questions[1], options[1], key="fill_q2", index=None)
        q6_3 = st.selectbox(questions[2], options[2], key="fill_q3", index=None)
        q6_4 = st.selectbox(questions[3], options[3], key="fill_q4", index=None)


    # 会话状态保存运行成功的标志
    if 'step6_success' not in st.session_state:
        st.session_state.step6_success = False
        
    if st.button("运行代码", key="run_step6"):
        current_answers = [q6_1, q6_2,q6_3,q6_4]
        correct = [a == b for a, b in zip(current_answers, correct_answers)]
        
        # 记录答题详情和错误信息
        for q, ans, correct_ans, is_cor in zip(questions, current_answers, correct_answers, correct):
            record_answer(6, q, ans, correct_ans, is_cor)
            if not is_cor:
                record_error(6, q, ans, correct_ans)
        
        if all(correct):
            # 用训练好的模型预测测试集文本类别
            y_pred = st.session_state.model.predict(st.session_state.X_test_tfidf)  # 填入测试集特征

            accuracy = accuracy_score(st.session_state.y_test, y_pred)
            st.session_state.accuracy = accuracy
            
            st.success("模型评估完成！")
            st.subheader(f"模型准确率：{accuracy:.4f}")            
            # 生成分类报告（返回字符串格式）
            report_str = classification_report(
                st.session_state.y_test,
                y_pred,
                target_names=FEATURE_NAMES_CN,
                output_dict=False  # 先获取字符串格式用于解析
            )

            # 将字符串报告转换为DataFrame
            lines = report_str.split('\n')
            report_data = []
            for line in lines[2:-3]:  # 提取类别行（排除标题和汇总行）
                row = line.strip().split()
                if len(row) == 5:  # 类别行包含：类别名、precision、recall、f1-score、support
                    report_data.append({
                        '类别': row[0],
                        '精确率（Precision）': float(row[1]),
                        '召回率（Recall）': float(row[2]),
                        'F1值': float(row[3]),
                        '样本数（Support）': int(row[4])
                    })

            # 提取汇总行（加权平均）
            avg_line = lines[-2].strip().split()
            report_data.append({
                '类别': avg_line[0] + ' ' + avg_line[1],  # "weighted avg"
                '精确率（Precision）': float(avg_line[2]),
                '召回率（Recall）': float(avg_line[3]),
                'F1值': float(avg_line[4]),
                '样本数（Support）': int(avg_line[5])
            })

            # 转换为DataFrame并显示
            report_df = pd.DataFrame(report_data)
            st.subheader("分类报告")
            st.dataframe(report_df.style.format({
                '精确率（Precision）': '{:.4f}',
                '召回率（Recall）': '{:.4f}',
                'F1值': '{:.4f}'
            }), use_container_width=True)

            # 分析错误样本
            error_indices = [i for i, (true, pred) in enumerate(zip(st.session_state.y_test, y_pred)) if true != pred]
            st.info(f"##### 错误分类样本总数：{len(error_indices)}")
            st.success(""" 
##### **分类报告解读**:
1. 模型对**计算机图形学**、**棒球运动**识别能力极强，对**政治讨论**识别能力最弱（样本少 + 特征区分度低）；
2. **摩托车**类是 “漏判少、误判多”，**太空科学 / 政治讨论**是 “漏判多、误判相对少”，优化方向不同；
3. 整体 85% 的 F1 值可接受，优先优化**政治讨论**（补样本 + 特征） 和**摩托车**（降误判） 能显著提升整体性能。
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
        "【反思】在本步骤中，你有什么不太理解的内容？（例如：分类报告）",
        key="step6_reflection",
        autocomplete="off",
    )    
    if reflection:
        st.session_state.bys_step_records['reflection']['step_6'] = reflection
    
    # 下一步按钮
    if st.session_state.step5_success:
        st.info("模型评估与分析环节圆满收官咯！🥳 揪出了那些错误样本🤔，把模型的小短板都看得明明白白～")
        if st.button("进入下一步：总结与思考", key="to_step6"):
            complete_step(6)
            st.session_state.step = 7
            st.session_state.step5_success = False
            st.rerun()    
       

# 步骤7：反思与总结
def step7():
    st.header("反思与总结")
    st.subheader("目标：梳理朴素贝叶斯文本分类完整流程与学习收获")
    st.info("""
    **任务说明**：  
    1. 总结朴素贝叶斯模型的核心原理与文本分类应用场景  
    2. 回顾本次实践的关键发现与遇到的问题  
    3. 整理学习收获与未来可探索的方向  
    """)   
   
    # 1. 流程回顾
    st.subheader("📝 完整流程回顾")
    st.info("""
        1. 项目说明：明确新闻文本分类的任务目标与数据集情况
        2. 数据加载：获取20 Newsgroups数据集的训练集和测试集
        3. 数据观察：分析文本数据的类别分布与基本特征
        4. 特征提取：使用TF-IDF将文本转换为数值特征
        5. 模型构建：实例化多项式朴素贝叶斯分类模型
        6. 模型训练与评估：训练模型并使用准确率等指标分析表现 
        """)

    
    # 2. 核心结果展示
    st.subheader("📊 模型核心结果摘要")
    st.subheader(f"模型准确率：{st.session_state.accuracy:.4f}") 
    st.info("关键发现：朴素贝叶斯模型在文本分类任务中表现高效，能快速处理高维TF-IDF特征")
    
    # 3. 知识理解测试
    st.subheader("📌 理解测试")
    questions = [
        "T1. 朴素贝叶斯算法中'朴素'一词的含义是什么？",
        "T2. 为什么TF-IDF比单纯的词袋模型（词频计数）更适合文本特征提取？",
        "T3. 当朴素贝叶斯模型在测试集上表现不佳时，可能的原因是什么？"
    ]
    options = [
        [
            "假设特征之间相互独立，简化了计算复杂度",
            "模型结构简单，训练速度快",
            "只能处理小规模数据集",
            "预测精度较低，是简单的基础模型"
        ],
        [
            "能自动进行文本分词和去停用词处理",
            "通过逆文档频率调整权重，突出稀有但重要的词",
            "生成的特征维度更低，计算更高效",
            "不需要对训练集和测试集使用相同的转换规则"
        ],
        [
            "训练数据量不足或类别分布不均衡",
            "文本特征提取效果差，未捕捉关键信息",
            "特征之间存在较强相关性，违反独立性假设",
            "以上都是"
        ]
    ]
    correct_answers = [
        "假设特征之间相互独立，简化了计算复杂度",
        "通过逆文档频率调整权重，突出稀有但重要的词",
        "以上都是"
    ]
    
    # 生成测验选项
    q7_1 = st.radio(questions[0], options[0], key="q7_1", index=None)
    q7_2 = st.radio(questions[1], options[1], key="q7_2", index=None)
    q7_3 = st.radio(questions[2], options[2], key="q7_3", index=None)
    current_answers = [q7_1, q7_2, q7_3]
    
    # 4. 学习反思输入
    st.subheader("📌 分析与改进")
    reflection = st.text_input(
        "请总结本次朴素贝叶斯文本分类实践的主要收获、遇到的问题及解决方法",
        key="step7_reflection",
        autocomplete="off",
    )
    if reflection:
        st.session_state.bys_step_records['reflection']['step_7'] = reflection
     
    # 提交与验证逻辑
    if st.button("提交理解测试与我的分析改进意见", key="submit_summary"):
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
            st.success("🎉 恭喜完成朴素贝叶斯文本分类全流程实践！")
            st.info("""
                本次实践总结：
                1. 掌握了朴素贝叶斯文本分类的完整构建流程
                2. 学会了使用TF-IDF进行文本特征提取的方法
                3. 理解了朴素贝叶斯模型在文本分类中的优势与局限
                            
                后续探索方向：
                - 尝试调整TF-IDF参数（如max_features、stop_words）优化特征
                - 对比不同朴素贝叶斯变种（如伯努利贝叶斯、高斯贝叶斯）的表现
                - 结合文本预处理（如词干提取、lemmatization）提高分类效果
            """)
                
        # 生成报告按钮
        if st.button("2.生成朴素贝叶斯分步编程学习报告", key="generate_report"):
            st.session_state.show_report = True  # 切换状态
            st.rerun()  # 刷新页面
        if st.session_state.show_report:
            # 显示报告内容
            report = generate_report_step(
                raw_records=st.session_state.bys_step_records, steps=8
            )
            st.subheader("📊 朴素贝叶斯文本分类分步编程学习报告")
            st.caption(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(report)
            st.session_state.show_report = False

# 主程序
def main():
    st.title("📝 朴素贝叶斯文本分类分步编程训练")
    init_session_state()   
    # 侧边栏步骤进度
    st.sidebar.title("步骤进度")
    steps = [
        "0. 项目说明",
        "1. 数据加载", "2. 数据观察", "3. 特征提取",
        "4. 模型构建", "5. 模型训练", "6. 结果评估", "7. 总结与思考"
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







