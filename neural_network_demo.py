# streamlit run neural_network_demo.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification, make_regression, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import time
from api_deepseek import client, ask_ai_assistant
from matplotlib.colors import ListedColormap
import neural_network_step_by_step
from datetime import datetime
from learning_report import generate_evaluation
# 设置页面
st.set_page_config(page_title="神经网络交互式学习平台", layout="wide")
st.title("🧠 神经网络交互式学习平台")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化会话状态（在主程序入口处）
def init_session_state():
    if 'nn_records' not in st.session_state:
        st.session_state.nn_records = {
            "nn_basics_section": [],  # 神经网络基本概念
            "multi_layer_nn_section": [],  # 多层神经网络与反向传播
            "activation_functions_section": [],  # 激活函数作业与选择
            "nn_parameter_tuning_section": [],  # 神经网络参数调优
            "nn_applications_section": [], #实际案例
            "module_sequence": [],  # 模块访问顺序
            "module_timestamps": {},  # 模块停留时间
            "ANN_quiz": {},  # 测验记录
            "ai_interactions": []  # AI交互记录
        }
        
def display_chat_interface(context=""):
    """显示聊天界面"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 AI助教已就绪")
    
    # 预设问题快捷按钮
    st.sidebar.markdown("**快捷问题:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        btn1 = st.button("什么是神经网络?")
        btn2 = st.button("激活函数的作用?")
    
    with col2:
        btn3 = st.button("反向传播算法?")
        btn4 = st.button("过拟合如何解决?")
    
    # 处理快捷问题
    question = ""
    if btn1:
        question = "什么是神经网络?"
    elif btn2:
        question = "激活函数的作用是什么?"
    elif btn3:
        question = "什么是反向传播算法?"
    elif btn4:
        question = "如何解决神经网络的过拟合问题?"
    
    # 提问输入框
    user_input = st.sidebar.text_input("输入你的问题:", key="question_input")
    if user_input:
        question = user_input
    
    # 处理提问
    if question:
        # 记录AI交互
        if "ai_interactions" not in st.session_state.nn_records:
            st.session_state.nn_records["ai_interactions"] = []

        st.session_state.nn_records["ai_interactions"].append({
            "question": question,
            "timestamp": datetime.now().timestamp()
        })
        
        # 显示当前问题
        st.sidebar.markdown(f"**你:** {question}")
        
        # 获取回答
        with st.spinner("助教思考中..."):
            time.sleep(1)  # 模拟思考时间
            answer = ask_ai_assistant(question, context)
        
        # 显示当前回答
        st.sidebar.markdown(f"**助教:** {answer}")
        st.sidebar.markdown("---")

# 数据生成函数
def generate_data(data_type, n_samples=300, noise=0.1):
    """生成不同类型的数据用于神经网络演示"""
    np.random.seed(42)
    
    if data_type == "二分类问题":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_informative=2,
            n_redundant=0, n_clusters_per_class=1, random_state=42
        )
        problem_type = "classification"
    
    elif data_type == "多分类问题":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_informative=2,
            n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=42
        )
        problem_type = "classification"
    
    elif data_type == "非线性分类":
        # 生成环形数据
        X = np.random.randn(n_samples, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
        y = y.astype(int)
        problem_type = "classification"
    
    elif data_type == "回归问题":
        X, y = make_regression(
            n_samples=n_samples, n_features=1, noise=noise*10, random_state=42
        )
        # 使关系非线性化
        y = y + 30 * np.sin(X).ravel()
        problem_type = "regression"
    
    return X, y, problem_type

# 绘制数据分布
def plot_data(X, y=None, title="数据分布", problem_type="classification"):
    """绘制数据集的散点图或线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if problem_type == "classification":
        if y is not None:
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
            ax.legend(*scatter.legend_elements(), title="类别")
        else:
            ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.7, s=50)
        ax.set_xlabel('特征 1')
        ax.set_ylabel('特征 2')
    
    elif problem_type == "regression":
        ax.scatter(X, y, alpha=0.7, s=50)
        ax.set_xlabel('特征')
        ax.set_ylabel('目标值')
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# 可视化神经网络决策边界
def plot_decision_boundary(X, y, model, title="神经网络决策边界", problem_type="classification"):
    """绘制神经网络的决策边界"""
    if problem_type != "classification" or X.shape[1] != 2:
        return None
    
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, alpha=0.7, s=50, edgecolor="k")
    
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# 可视化神经网络训练过程
def plot_training_curve(history, title="神经网络训练曲线"):
    """绘制神经网络的训练曲线（损失和准确率）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制损失曲线
    ax1.plot(history['loss'], label='训练损失')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='验证损失')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制准确率曲线（如果是分类问题）
    if 'accuracy' in history:
        ax2.plot(history['accuracy'], label='训练准确率')
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='验证准确率')
        ax2.set_title('准确率曲线')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
    else:
        ax2.set_visible(False)
    
    plt.tight_layout()
    return fig

# 激活函数可视化
def plot_activation_functions():
    """绘制常用激活函数"""
    x = np.linspace(-5, 5, 100)
    
    # 定义激活函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def relu(x):
        return np.maximum(0, x)
    
    def leaky_relu(x, alpha=0.1):
        return np.where(x >= 0, x, alpha * x)
    
    # 绘制激活函数
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('常用激活函数', fontsize=16)
    
    axes[0, 0].plot(x, sigmoid(x))
    axes[0, 0].set_title('Sigmoid')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    axes[0, 1].plot(x, tanh(x))
    axes[0, 1].set_title('Tanh')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    axes[1, 0].plot(x, relu(x))
    axes[1, 0].set_title('ReLU')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1, 1].plot(x, leaky_relu(x))
    axes[1, 1].set_title('Leaky ReLU')
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # 绘制完所有激活函数曲线后，遍历所有子图添加参考线
    for ax in axes.flatten():
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)  # x=0垂直线
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)  # y=0水平线
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# 神经网络基础概念模块
def nn_basics_section():
    st.header("🔍 神经网络基础概念")
    
    # 使用列布局将文本和示意图并排显示
    col_text, col_image = st.columns([3, 2])  # 文本占3份，图片占2份
    
    with col_text:
        st.markdown("""
        **神经网络的基本组成:**
        神经网络由相互连接的人工神经元组成，主要包括：
        
        - **输入层**: 接收原始数据
        - **隐藏层**: 进行特征学习和转换
        - **输出层**: 产生最终预测结果
        - **权重和偏置**: 网络的参数，通过训练学习得到
        - **激活函数**: 引入非线性，使网络能够学习复杂模式
        
        **神经网络的工作原理:**
        1. 前向传播: 输入数据通过网络层层传递，计算输出
        2. 计算损失: 比较预测输出与真实值的差异
        3. 反向传播: 计算损失对各参数的梯度
        4. 参数更新: 使用优化算法（如梯度下降）更新权重和偏置
        """)
    
    with col_image:
        # 显示神经网络结构示意图，设置较小的宽度
        st.markdown("**神经网络结构示意图**")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/800px-Colored_neural_network.svg.png",
                 caption="三层神经网络结构（输入层、隐藏层、输出层）",
                 width=250)  # 调整宽度使图片变小
    
    # 感知器演示
    st.subheader("感知器演示（最简单的神经网络）")

    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        感知器是最简单的神经网络，包含一个神经元：
        1. 接收多个输入并加权求和
        2. 应用激活函数产生输出
        3. 通过学习调整权重以正确分类数据

        感知器只能解决线性可分问题，无法解决异或（XOR）等非线性问题。
        """)

      
    with col2:
        # 显示感知器公式
        st.markdown("""**感知器学习规则:**""")       
        st.latex(r"y = \text{sign}(w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n)")
        st.markdown("""**当预测错误时，更新权重：**""") 
        st.latex(r"w_i = w_i + \eta \cdot (y_{true} - y_{pred}) \cdot x_i")

    # 记录数据生成操作
    st.session_state.nn_records["nn_basics_section"].append({
        "timestamp": datetime.now().timestamp()
    })

    return "神经网络基础概念模块: 介绍了神经网络组成和感知器原理"

# 多层神经网络模块
def multi_layer_nn_section():
    st.header("🏗️ 多层神经网络与反向传播")
   

    st.markdown("""
    **多层神经网络的优势:**
    多层神经网络（深度学习）通过叠加多个隐藏层，可以学习更复杂的非线性关系，能够解决感知器无法解决的非线性问题（如异或问题）
    """)
    
    import os
    # 显示图片
    st.subheader("多层神经网络结构")
    image_path = f"{os.path.dirname(os.path.abspath(__file__))}/ANN.jpg"
    st.image(image_path,
                caption="多层神经网络结构",
                width=500)
    
    # 反向传播算法部分，使用上下排版而非分栏
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **反向传播算法:**
        反向传播是训练多层神经网络的核心算法：
        1. 前向传播计算预测值和损失
        2. 反向计算损失对每个参数的梯度
        3. 使用梯度下降更新参数以最小化损失
        """)
    with col2:
        st.markdown("""
        **反向传播的核心思想:**
        - 利用链式法则计算损失对每个权重的梯度
        - 从输出层向输入层反向传播误差
        - 通过梯度下降更新权重，使损失最小化
        
        对于一个简单的两层网络，输出层权重更新公式：""")
        st.latex(r"\Delta w_{jk} = \eta \cdot \delta_k \cdot a_j")
        st.markdown("其中 $\delta_k$ 是输出层误差项，$a_j$ 是隐藏层激活值。")
            
    # 异或问题演示
    st.subheader("异或问题的解决")
    st.markdown("""
    异或（XOR）是一个经典的非线性问题，单层感知器无法解决，但多层神经网络可以解决：
    """)
    
    # 生成异或数据
    X, y, problem_type = generate_data("非线性分类", n_samples=100)
    col1, col2 = st.columns([2,3])
    with col1: 
    # 神经网络参数设置
        hidden_units = st.slider("隐藏层神经元数", 2, 20, 4)
        activation = st.selectbox("激活函数", ["relu", "tanh", "logistic"])
        max_iter = st.slider("最大迭代次数", 100, 800, 200)
        learning_rate = st.slider("学习率", 0.001, 0.3, 0.01)    
    with col2:    
        # 显示异或数据
        fig_xor = plot_data(X, y, "异或问题数据分布")
        st.pyplot(fig_xor)

    # 记录数据生成操作
    st.session_state.nn_records["multi_layer_nn_section"].append({
        "hidden_units":hidden_units,
        "activation":activation,
        "max_iter":max_iter,
        "learning_rate":learning_rate,
        "timestamp": datetime.now().timestamp()
    })
        
    if st.button("训练神经网络解决异或问题"):
        # 训练神经网络
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation=activation,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=42,
            solver='sgd',
            verbose=False)
            
        model.fit(X, y)
            
        # 显示训练结果
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        st.success(f"训练完成！准确率: {accuracy:.4f}")
        col1, col2, col3 = st.columns([1,4,1])
        with col2:             
            # 绘制决策边界
            fig_db = plot_decision_boundary(X, y, model, f"神经网络决策边界 (准确率: {accuracy:.4f})")
            if fig_db: st.pyplot(fig_db)
                
        # 绘制训练曲线
        history = {
            'loss': model.loss_curve_,
            'accuracy': [accuracy_score(y, model.predict(X))] * len(model.loss_curve_)
            }
        fig_train = plot_training_curve(history, "神经网络训练曲线")
        st.pyplot(fig_train)
        
    
    return "多层神经网络模块: 演示了多层网络解决异或问题"

# 激活函数模块
def activation_functions_section():
    st.header("⚡ 激活函数的作用与选择")
    
    st.info("""
    **激活函数的重要性:**
    激活函数为神经网络引入非线性，使模型能够学习复杂的模式和关系：
    - 没有激活函数，无论多少层的神经网络都只能表示线性关系
    - 激活函数决定神经元是否被"激活"（输出信号强度）
    - 不同的激活函数适用于不同的场景和网络结构
    """)
    
    # 显示激活函数图像
    fig_activation = plot_activation_functions()
    st.pyplot(fig_activation)
    
    # 激活函数对比表格
    st.subheader("常用激活函数对比")
    activation_data = {
        "激活函数": ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU"],
        "值域": ["(0, 1)", "(-1, 1)", "[0, ∞)", "(-∞, ∞)"],
        "优点": [
            "输出在0-1之间，可表示概率",
            "均值为0，训练更稳定",
            "计算简单，缓解梯度消失",
            "解决ReLU的死亡神经元问题"
        ],
        "缺点": [
            "梯度消失，计算成本高",
            "仍存在梯度消失问题",
            "存在死亡神经元问题",
            "增加了一个超参数"
        ],
        "适用场景": [
            "二分类输出层",
            "隐藏层",
            "隐藏层（最常用）",
            "隐藏层（替代ReLU）"
        ]
    }
    activation_df = pd.DataFrame(activation_data)
    st.dataframe(activation_df)
    
    # 激活函数对训练的影响演示
    st.subheader("激活函数对神经网络训练的影响")
    
    # 生成非线性分类数据
    X, y, problem_type = generate_data("非线性分类", n_samples=150)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 选择激活函数进行比较
    activation1 = st.selectbox("激活函数 1", ["relu", "tanh", "logistic"], key="a1")
    activation2 = st.selectbox("激活函数 2", ["tanh", "relu", "logistic"], key="a2")

    # 记录数据生成操作
    st.session_state.nn_records["activation_functions_section"].append({
        "activation1":activation1,
        "activation2":activation2,
        "timestamp": datetime.now().timestamp()
    })
    
    if st.button("比较激活函数效果"):
        # 训练两个不同激活函数的网络
        models = []
        histories = []
        
        for activation in [activation1, activation2]:
            model = MLPClassifier(
                hidden_layer_sizes=(10,),
                activation=activation,
                max_iter=1000,
                random_state=42,
                solver='adam'
            )
            model.fit(X_train, y_train)
            models.append(model)
            
            # 计算训练和测试准确率
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            
            histories.append({
                'loss': model.loss_curve_,
                'accuracy': [train_acc] * len(model.loss_curve_),
                'val_accuracy': [test_acc] * len(model.loss_curve_)
            })
        
        # 绘制训练曲线对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 损失曲线
        ax1.plot(histories[0]['loss'], label=f'{activation1} 损失')
        ax1.plot(histories[1]['loss'], label=f'{activation2} 损失')
        ax1.set_title('损失曲线对比',fontsize=18)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('损失值')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(histories[0]['accuracy'], label=f'{activation1} 训练准确率')
        ax2.plot(histories[0]['val_accuracy'], label=f'{activation1} 测试准确率')
        ax2.plot(histories[1]['accuracy'], label=f'{activation2} 训练准确率')
        ax2.plot(histories[1]['val_accuracy'], label=f'{activation2} 测试准确率')
        ax2.set_title('准确率曲线对比',fontsize=18)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 显示决策边界对比
        fig_db, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 绘制决策边界的辅助函数
        def plot_db(model, ax, title):
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
            cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
            
            ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, alpha=0.7, edgecolor="k")
            ax.set_title(title,fontsize=18)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plot_db(models[0], ax1, f'{activation1} 决策边界 (准确率: {histories[0]["val_accuracy"][0]:.4f})')
        plot_db(models[1], ax2, f'{activation2} 决策边界 (准确率: {histories[1]["val_accuracy"][0]:.4f})')
        
        plt.tight_layout()
        st.pyplot(fig_db)
        
   
    return "激活函数模块: 比较了不同激活函数的效果"

# 神经网络参数调优模块
def nn_parameter_tuning_section():
    st.header("🎛️ 神经网络参数调优")
    
    st.markdown("""
    **神经网络的关键参数:**
    神经网络的性能很大程度上取决于参数设置，主要包括：🎯 """)
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
    1. ⚡ **网络结构**:
       - 隐藏层数量
       - 每层神经元数量""")
        st.info("""
    2. ⚡ **训练参数**:
       - 学习率
       - 迭代次数
       - 批大小
       - 正则化参数""")
    with col2:
        st.info("""
    3. ⚡ **优化器选择**:
       - SGD（随机梯度下降）
       - Adam
       - RMSprop
       - Adagrad
    """)
    
    # 加载手写数字数据集
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.subheader("🏃 参数调优对模型性能的影响")
    st.markdown("使用手写数字识别任务演示参数调优的影响：")
    
    # 显示示例数字
    st.subheader("示例数据")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r)
        ax.set_title(f'标签: {digits.target[i]}')
        ax.axis('off')
    st.pyplot(fig)
    
    # 参数设置
    col1, col2 = st.columns(2)
    with col1:
        hidden_layers = st.slider("隐藏层数量", 1, 3, 1)
        neurons_per_layer = st.slider("每层神经元数量", 10, 100, 30, step=10)
        
        # 根据隐藏层数量构建网络结构
        hidden_layer_sizes = tuple([neurons_per_layer] * hidden_layers)
        
        learning_rate = st.slider("学习率", 0.01, 0.1, 0.01)
        regularization = st.slider("L2正则化系数", 0.01, 0.1, 0.01)
        solver = st.selectbox("优化器", ["sgd", "adam", "lbfgs", "rmsprop"])
        
         
    with col2:
        st.subheader("参数调优建议")
        st.markdown("""
        **网络结构:**
        - 隐藏层数量: 简单问题1-2层，复杂问题3-5层
        - 神经元数量: 输入特征多则适当增加，避免过多导致过拟合
        
        **学习率:**
        - 过小: 收敛慢，需要更多迭代
        - 过大: 可能跳过最优解，训练不稳定
        - 通常在0.001-0.1之间
        
        **正则化:**
        - 用于防止过拟合
        - 系数过大会导致欠拟合
        - 通常在0.0001-0.1之间
        
        **优化器选择:**
        - SGD: 基础优化器，可配合动量
        - Adam: 自适应学习率，大多数情况下表现良好
        - RMSprop: 在递归神经网络中表现较好
        """)

    # 记录数据生成操作
    st.session_state.nn_records["nn_parameter_tuning_section"].append({
        "hidden_layer_sizes":hidden_layer_sizes,
        "neurons_per_layer":neurons_per_layer,
        "regularization":regularization,
        "learning_rate":learning_rate,
        "solver":solver,
        "timestamp": datetime.now().timestamp()
    })
        
    if st.button("训练神经网络"):
        # 训练神经网络
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver=solver,
            learning_rate_init=learning_rate,
            alpha=regularization,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
            
        with st.spinner("模型训练中..."):
            model.fit(X_train_scaled, y_train)
            
        # 评估模型
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
            
        st.success(f"训练完成！训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
            
        # 绘制训练曲线
        history = {
            'loss': model.loss_curve_,
            'val_loss': model.validation_scores_,
            'accuracy': [train_acc] * len(model.loss_curve_),
            'val_accuracy': [test_acc] * len(model.loss_curve_)
        }
        fig_train = plot_training_curve(history, "神经网络训练曲线")
        st.pyplot(fig_train)
            
        # 显示错误分类的例子
        y_pred = model.predict(X_test_scaled)
        misclassified = X_test[y_pred != y_test]
        true_labels = y_test[y_pred != y_test]
        pred_labels = y_pred[y_pred != y_test]
            
        if len(misclassified) > 0:
            st.subheader("错误分类的示例")
            fig_mis, axes = plt.subplots(1, min(5, len(misclassified)), figsize=(10, 2))
            for i, ax in enumerate(axes):
                ax.imshow(misclassified[i].reshape(8, 8), cmap=plt.cm.gray_r)
                ax.set_title(f'真实: {true_labels[i]}, 预测: {pred_labels[i]}')
                ax.axis('off')
            st.pyplot(fig_mis)    
    return "神经网络参数调优模块: 演示了参数对模型性能的影响"

# 概念测验模块
def quiz_section():
    st.header("🎯 神经网络概念测验")
    st.write("请完成以下5道单选题，全部答完后可提交查看结果")
    
    # 定义神经网络测验题目、选项、正确答案及解析
    quiz_data = [
        {
            "question": "1. 神经网络中激活函数的核心作用是什么？",
            "options": [
                "A. 加速数据归一化过程",
                "B. 引入非线性，让网络能拟合复杂模式",
                "C. 减少模型的参数数量",
                "D. 提升数据的维度"
            ],
            "correct": "B",
            "explanation": "激活函数的核心是引入非线性变换，若没有激活函数，多层神经网络退化为线性模型，无法拟合复杂的非线性关系。"
        },
        {
            "question": "2. 反向传播（Backpropagation）的主要目的是什么？",
            "options": [
                "A. 初始化神经网络的权重参数",
                "B. 计算损失函数对各参数的梯度，用于更新权重",
                "C. 对输入数据进行特征提取",
                "D. 划分训练集和验证集"
            ],
            "correct": "B",
            "explanation": "反向传播通过链式法则从输出层向输入层反向计算损失函数对每个参数的梯度，是梯度下降优化模型的核心步骤。"
        },
        {
            "question": "3. 以下哪种方法不能缓解神经网络的过拟合问题？",
            "options": [
                "A. 早停法（Early Stopping）",
                "B. 增加训练数据量",
                "C. 减少网络的隐藏层神经元数量",
                "D. 增大学习率"
            ],
            "correct": "D",
            "explanation": "增大学习率可能导致模型不收敛或震荡，无法缓解过拟合；早停法、增加数据、简化网络结构都是常见的过拟合缓解策略。"
        },
        {
            "question": "4. 全连接神经网络（FCN）中，相邻两层神经元的连接方式是？",
            "options": [
                "A. 每个神经元只连接下一层的一个神经元",
                "B. 每个神经元连接下一层的所有神经元",
                "C. 神经元随机连接下一层的部分神经元",
                "D. 同层神经元互相连接"
            ],
            "correct": "B",
            "explanation": "全连接层的核心特征是：上一层的每个神经元与下一层的所有神经元都有权重连接，因此参数数量通常较多。"
        },
        {
            "question": "5. 梯度下降算法中，学习率（Learning Rate）过大会导致什么问题？",
            "options": [
                "A. 模型训练速度过慢",
                "B. 模型可能越过最优解，无法收敛",
                "C. 模型泛化能力增强",
                "D. 梯度计算错误"
            ],
            "correct": "B",
            "explanation": "学习率过大时，参数更新步长太大，可能在最优解附近震荡甚至越过最优解，导致模型无法收敛；学习率过小则训练速度慢。"
        }
    ]
    
    # 初始化会话状态存储用户答案
    if "nn_user_answers" not in st.session_state:
        st.session_state.nn_user_answers = [None] * len(quiz_data)
    
    # 初始化测验记录（若未存在）
    if "nn_records" not in st.session_state:
        st.session_state.nn_records = {}
    
    # 显示所有题目和选项（初始无选中状态）
    for i, item in enumerate(quiz_data):
        st.markdown(f"**{item['question']}**")
        # 设置默认值为None实现初始无选中状态，通过会话状态保存答案
        answer = st.radio(
            "选择答案:",
            item["options"],
            key=f"nn_quiz_{i}",
            index=None,  # 关键：初始无选中项
            label_visibility="collapsed"
        )
        
        # 更新会话状态中的答案（提取选项字母A/B/C/D）
        if answer is not None:
            st.session_state.nn_user_answers[i] = answer[0]
        
    # 检查是否所有题目都已作答
    all_answered = all(ans is not None for ans in st.session_state.nn_user_answers)
    
    # 提交按钮：只有全部答完才可用
    submit_btn = st.button(
        "提交答案", 
        key="submit_nn_quiz",
        disabled=not all_answered  # 未答完时禁用
    )
    
    # 未答完时显示提示
    if not all_answered:
        st.info("请完成所有5道题目后再提交")
    
    # 处理提交
    if submit_btn and all_answered:
        # 计算得分和错误题目（每题20分）
        score = 0
        results = []
        incorrect_questions = []
        for i, item in enumerate(quiz_data):
            is_correct = st.session_state.nn_user_answers[i] == item["correct"]
            if is_correct:
                score += 20  # 每题20分，总分100
            else:
                incorrect_questions.append({
                    "topic": item["question"], 
                    "user_answer": st.session_state.nn_user_answers[i]
                })

            results.append({
                "question": item["question"],
                "user_answer": st.session_state.nn_user_answers[i],
                "correct_answer": item["correct"],
                "is_correct": is_correct,
                "explanation": item["explanation"]
            })
            
        # 记录测验结果到会话状态
        st.session_state.nn_records["ANN_quiz"] = {
            "score": score,
            "incorrect_questions": incorrect_questions,
            "timestamp": datetime.now().timestamp()
        }
       
        # 显示得分
        st.success(f"📊 测验完成！你的得分是：{score}分")
        st.write("### 答案解析：")
        
        # 展开式显示每题结果
        for res in results:
            # 使用emoji标记正确/错误状态
            status_text = "✅ 正确" if res["is_correct"] else "❌ 错误"
            
            with st.expander(f"{res['question']} {status_text}"):
                if res["is_correct"]:
                    st.success(f"你的答案：{res['user_answer']}（正确）")
                else:
                    st.error(f"你的答案：{res['user_answer']}（错误）")
                    st.info(f"正确答案：{res['correct_answer']}")
                st.write(f"解析：{res['explanation']}")

        # 准备AI分析的输入
        incorrect_topics = [
            res["question"] for res in results if not res["is_correct"]
        ]
        
        analysis_prompt = f"""
        以下是学生在神经网络测验中的答题情况：
        - 总得分：{score}分
        - 错误题目：{len(incorrect_topics)}道
        - 错误知识点：{'; '.join(incorrect_topics) if incorrect_topics else '无'}
        
        请分析该学生的知识掌握情况，指出未掌握的核心概念，并给出具体的学习建议和指导方向，帮助学生针对性提升。
        答案必须控制在450字以内
        """
        
        # 调用AI分析（带加载状态）
        with st.spinner("AI正在分析你的答题情况..."):
            ai_analysis = ask_ai_assistant(analysis_prompt, "神经网络测验分析")
        
        # 显示AI分析结果
        st.write("### 🤖 AI学习诊断：")
        st.info(ai_analysis)       
  
    return "神经网络概念测验模块：完成5题单选题测试"

# 神经网络应用案例模块
def nn_applications_section():
    st.header("🌍 神经网络实际应用案例")
    
    example = st.selectbox(
        "选择应用案例:",
        ["图像识别", "回归预测", "上传自己的数据"]
    )
    
    if example == "上传自己的数据":
        uploaded_file = st.file_uploader("上传CSV文件", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("数据预览:", data.head())
            
            # 检查目标列
            target_col = st.selectbox("选择目标列", data.columns)
            if target_col:
                X = data.drop(target_col, axis=1)
                y = data[target_col]
                
                # 检查是否为分类问题
                is_classification = len(y.unique()) < 10 or str(y.dtype) == 'object'
                
                # 处理分类特征
                X = pd.get_dummies(X)
                
                # 标准化数据
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 训练神经网络
                st.subheader("神经网络参数")
                layers = st.slider("隐藏层数量", 1, 3, 1)
                neurons = st.slider("每层神经元数量", 10, 100, 50)
                
                if st.button("训练模型"):
                    if is_classification:
                        model = MLPClassifier(
                            hidden_layer_sizes=tuple([neurons]*layers),
                            max_iter=500,
                            random_state=42
                        )
                    else:
                        model = MLPRegressor(
                            hidden_layer_sizes=tuple([neurons]*layers),
                            max_iter=500,
                            random_state=42
                        )
                    
                    model.fit(X_scaled, y)
                    
                    # 评估模型
                    if is_classification:
                        y_pred = model.predict(X_scaled)
                        accuracy = accuracy_score(y, y_pred)
                        st.success(f"分类准确率: {accuracy:.4f}")
                    else:
                        y_pred = model.predict(X_scaled)
                        mse = mean_squared_error(y, y_pred)
                        st.success(f"均方误差: {mse:.4f}")
                    
                    # 绘制训练曲线
                    history = {'loss': model.loss_curve_}
                    if is_classification:
                        history['accuracy'] = [accuracy_score(y, model.predict(X_scaled))] * len(model.loss_curve_)
                    fig = plot_training_curve(history)
                    st.pyplot(fig)
    
    elif example == "图像识别":
        st.markdown("""
        **图像识别应用:**
        神经网络在图像识别领域取得了巨大成功，从简单的数字识别到复杂的物体检测：
        - 卷积神经网络(CNN)是图像识别的首选模型
        - 能够自动学习图像的边缘、纹理等特征
        - 应用包括人脸识别、医学影像分析、自动驾驶等
        """)
        
        # 使用手写数字数据集演示
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # 训练一个简单的神经网络
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        model.fit(X, y)
        
        # 展示一些预测结果
        st.subheader("数字识别示例")
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(digits.images[i], cmap=plt.cm.gray_r)
            pred = model.predict([digits.data[i]])[0]
            ax.set_title(f'预测: {pred}, 真实: {digits.target[i]}')
            ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("""
        实际应用中的图像识别系统通常使用更深的卷积神经网络：
        - LeNet-5: 早期的数字识别网络
        - AlexNet: 深度学习革命的里程碑
        - ResNet: 使用残差连接解决深层网络训练问题
        - YOLO: 实时目标检测系统
        """)
    
    elif example == "回归预测":
        st.markdown("""
        **回归预测应用:**
        神经网络可以用于预测连续值输出：
        - 房价预测
        - 股票价格预测
        - 销售额预测
        - 温度预测等
        """)
        
        # 生成非线性回归数据
        X, y, problem_type = generate_data("回归问题", n_samples=200, noise=0.2)
        X = X.reshape(-1, 1)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 训练神经网络
        model = MLPRegressor(
            hidden_layer_sizes=(50, 30),
            activation='relu',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 预测
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_range)
        
        # 绘制结果
        cols=st.columns([1,4,1])
        with cols[1]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_train, y_train, alpha=0.5, label='训练数据')
            ax.scatter(X_test, y_test, alpha=0.5, c='red', label='测试数据')
            ax.plot(X_range, y_pred, 'g-', linewidth=2, label='神经网络预测')
            ax.set_title('神经网络回归预测')
            ax.set_xlabel('特征')
            ax.set_ylabel('目标值')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # 评估模型
        train_mse = mean_squared_error(y_train, model.predict(X_train))
        test_mse = mean_squared_error(y_test, model.predict(X_test))
        st.write(f"训练集均方误差: {train_mse:.4f}")
        st.write(f"测试集均方误差: {test_mse:.4f}")
        
        st.session_state.nn_records["nn_applications_section"] = {
            "timestamp": datetime.now().timestamp()
        }    
    return f"神经网络应用模块: 展示了{example}应用案例"

# 主程序
def main():

    # 初始化会话状态
    init_session_state()
    
    # 初始化会话状态
    if 'section' not in st.session_state:
        st.session_state.section = "神经网络基础概念"
        
    # 记录模块访问顺序
    current_section = st.session_state.section
    st.session_state.nn_records["module_sequence"].append(current_section)
    if current_section not in st.session_state.nn_records["module_timestamps"]:
        st.session_state.nn_records["module_timestamps"][current_section] = {
            "enter_time": time.time()
        }
        
    st.sidebar.title("导航菜单")
    section = st.sidebar.radio("选择学习模块", [
        "神经网络基础概念",
        "多层神经网络与反向传播",
        "激活函数的作用与选择",
        "神经网络参数调优",
        "概念测验",  
        "神经网络实际应用案例",
        "编程实例（加州房价数据集）"
    ])
  
    # 更新会话状态
    st.session_state.section = section
    
    
    context = ""
    if section == "神经网络基础概念":
        context = nn_basics_section()
    elif section == "多层神经网络与反向传播":
        context = multi_layer_nn_section()
    elif section == "激活函数的作用与选择":
        context = activation_functions_section()
    elif section == "神经网络参数调优":
        context = nn_parameter_tuning_section()
    elif section == "概念测验":  
        context = quiz_section()
    elif section == "神经网络实际应用案例":
        context = nn_applications_section()
    elif section == "编程实例（加州房价数据集）":
        # 初始化step变量（如果不存在）
        if 'step' not in st.session_state:
            st.session_state.step = 0
        neural_network_step_by_step.main()
        context = "编程实例模块: 加州房价数据集神经网络分步练习"
    
    # 显示聊天界面
    display_chat_interface(context)
    
    # 记录模块退出时间
    if current_section in st.session_state.nn_records["module_timestamps"]:
        st.session_state.nn_records["module_timestamps"][current_section]["exit_time"] = datetime.now().timestamp()
    
    if section != "编程实例（加州房价数据集）":
        # 侧边栏添加学习报告按钮（调用独立模块）
        st.sidebar.markdown("---")
        if st.sidebar.button("神经网络模块学习报告"):
            report = generate_evaluation(
                module_type="ANN",
                raw_records=st.session_state.nn_records
            )
            st.write("### 神经网络模块学习情况报告")
            st.info(report)
            
    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **神经网络交互式学习平台**
    
    设计用于机器学习教学，帮助学生理解:
    - 神经网络的基本原理与结构
    - 多层神经网络与反向传播算法
    - 激活函数的作用与选择
    - 神经网络参数调优方法
    - 神经网络的实际应用场景
    """)


if __name__ == "__main__":
    main()
