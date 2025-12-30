#  C:\Users\孙冰\Desktop\AI助教
#  streamlit run text_analysis_demo.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, ConfusionMatrixDisplay)
import os
import jieba
import re
import time
import native_bys
import json
import bayes_text_classification_step_by_step
from api_deepseek import ask_ai_assistant
from datetime import datetime
from learning_report import generate_evaluation
# 页面设置
st.set_page_config(page_title="文本分析与分类学习平台", layout="wide")
st.title("📄 文本分析与分类交互式学习平台")

# 初始化会话状态（在主程序入口处）
def init_session_state():
    if "text_analysis_records" not in st.session_state:
        st.session_state.text_analysis_records = {
            "text_introduction_section": [],  # 文本分析基础
            "text_preprocessing_section": [],  # 文本预处理
            "text_analysis_section": [],  # 文本分类专项
            "sentiment_analysis_section": [],  # 情感分析专项
            "native_bys_section":[], #朴素贝叶斯
            "module_sequence": [],  # 模块访问顺序
            "module_timestamps": {},  # 模块停留时间
            "text_analysis_quiz": {},  # 测验记录
            "ai_interactions": []  # AI交互记录
        }

def display_chat_interface(context=""):
    """显示贝叶斯文本分类相关的聊天界面"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 AI助教已就绪")
    
    # 预设贝叶斯文本分类相关的快捷问题
    st.sidebar.markdown("**快捷问题:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        btn1 = st.button("什么是贝叶斯文本分类?")
        btn2 = st.button("贝叶斯分类的核心原理?")
    
    with col2:
        btn3 = st.button("TF-IDF的作用是什么?")
        btn4 = st.button("贝叶斯分类的优缺点?")
    
    # 处理快捷问题
    question = ""
    if btn1:
        question = "什么是贝叶斯文本分类?它适用于哪些场景?"
    elif btn2:
        question = "贝叶斯文本分类的核心原理是什么?基于哪些数学公式?"
    elif btn3:
        question = "在文本分类中，TF-IDF特征提取的作用是什么?如何计算?"
    elif btn4:
        question = "贝叶斯文本分类有哪些优点和缺点?与其他分类算法相比有何特点?"
    
    # 提问输入框
    user_input = st.sidebar.text_input("输入你的问题(关于贝叶斯文本分类):", key="question_input")
    if user_input:
        question = user_input
    
    # 处理提问
    if question:

        # 记录AI交互
        if "ai_interactions" not in st.session_state.text_analysis_records:
            st.session_state.text_analysis_records["ai_interactions"] = []

        st.session_state.text_analysis_records["ai_interactions"].append({
            "question": question,
            "timestamp": datetime.now().timestamp()
        })
        
        # 显示当前问题
        st.sidebar.markdown(f"**你:** {question}")
        
        # 获取回答
        with st.spinner("助教思考中..."):
            answer = ask_ai_assistant(question, context)
        
        # 显示当前回答
        st.sidebar.markdown(f"**助教:** {answer}")
        st.sidebar.markdown("---")

# 文本预处理函数
def preprocess_text(text, is_chinese=False):
    """文本预处理：清洗、分词"""
    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\u4e00-\u9fa5]', ' ', text)
    # 转为小写
    text = text.lower() if not is_chinese else text
    # 分词
    if is_chinese:
        words = jieba.cut(text)
        return " ".join(words)
    else:
        return text

# 加载示例数据
@st.cache_data
def load_sample_data(dataset_name):
    """加载不同类型的文本数据集"""
    if dataset_name == "新闻主题分类":
        data_path = os.path.join(
            os.path.dirname(__file__), 
            "datasets", 
            "20newsgroups_selected.json"  # 👈 修改为新的JSON文件名
        )
        
        # 2. 读取本地 JSON 文件
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件未找到：{data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        texts = [preprocess_text(text) for text in dataset["train"]["data"]]
        labels = dataset["train"]["target"]
        label_names = dataset["train"]["target_names"]  # 英文类别名
        # 新增：获取中文类别名（从JSON中读取）
        chinese_label_names = dataset["train"]["chinese_target_names"]
        
        # 4. 保持返回值和原代码一致
        return texts, labels, label_names, "英文"

    
    elif dataset_name == "中文情感分析":
        # 优化后的正面样本（15条独立样本，含模糊表达和复杂语境）
        positive_samples = [
            "这手机续航比预期好，重度用一天还剩30%电，性价比可以",
            "虽然发货慢了两天，但包装很用心，产品没瑕疵，满意",
            "客服态度超赞，耐心解答了我一堆问题，必须好评",
            "味道不算惊艳，但家常味很足，吃着舒服，会回购",
            "外观设计简约大气，手感比图片看着好，值得入手",
            "第一次用这个牌子，没想到这么好用，超出预期",
            "价格小贵，但材质和做工明显比便宜货好，一分钱一分货",
            "功能不算多，但每一个都实用，没有花里胡哨的东西",
            "物流一般，但送货上门很方便，省了不少事",
            "安装有点麻烦，但说明书很详细，慢慢弄也能搞定",
            "颜色比想象中浅，但很耐看，越用越喜欢",
            "声音不算大，但清晰度高，日常用足够了",
            "尺码偏小一点，但版型很好，换大一码刚好合适",
            "刚用时有轻微异味，通风两天就没了，不影响使用",
            "操作界面有点复杂，但熟悉后效率很高，离不开了",
            "这个产品非常好，我很满意", "质量很棒，推荐购买", "体验超出预期，值得拥有",
            "服务态度很好，下次还会再来", "性价比高，非常划算", "物流很快，包装完好",
            "效果显著，确实有效", "外观设计很漂亮，很喜欢", "使用简单方便，操作流畅",
            "味道很好，家人都喜欢"
        ]

        # 优化后的负面样本（15条独立样本，含模糊表达和复杂语境）
        negative_samples = [
            "手机发热严重，玩10分钟游戏就烫手，不敢长时间用",
            "客服只会说套话，问题根本没解决，体验很差",
            "味道太咸了，料包全放根本没法吃，踩雷了",
            "外观看着廉价，塑料感强，和图片差距大",
            "用了不到一周就卡顿，后台清了也没用，不推荐",
            "价格虚高，同配置的其他牌子便宜一半，不值这个价",
            "功能鸡肋，很多用不上的设计，徒增复杂度",
            "物流超慢，显示三天到，结果等了一周才收到",
            "安装说明一团糟，看半天看不懂，最后找人帮忙才装上",
            "颜色发错了，退换还要自己承担运费，很不合理",
            "声音忽大忽小，调节也不灵敏，影响使用体验",
            "尺码严重不准，标注XL实际像M码，退换太麻烦",
            "异味特别重，放了一周还有味，不敢给孩子用",
            "操作反人类，很多基础功能藏得很深，老人根本不会用",
            "宣传说防水，结果溅了点水就坏了，质量堪忧",
            "质量太差了，完全不值这个价", "服务态度恶劣，非常失望", "一点用都没有，浪费钱",
            "物流太慢，包装破损", "体验很差，不会再买了", "味道很难闻，无法接受",
            "外观粗糙，有瑕疵", "操作复杂，一点都不方便", "效果很差，不如宣传的好",
            "性价比低，不推荐购买"
        ]

        texts0 = positive_samples  + negative_samples 
        labels = [1] * len(positive_samples)  + [0] * len(negative_samples)   # 1:正面, 0:负面
        label_names = ["负面", "正面"]
        # 中文预处理
        texts = [preprocess_text(text, is_chinese=True) for text in texts0]
        return texts, labels, label_names, "中文", texts0
    
    else:  # 自定义文本
        return [], [], [], "中文", []

# 特征提取演示
def demo_feature_extraction(texts, lang):
    """演示词袋模型和TF-IDF"""
    st.subheader("📁 文本向量化：从文字到数字")
    
    # 选择向量化方法
    vec_method = st.radio("选择向量化方法",
                          ["词袋模型 (CountVectorizer)", "TF-IDF (TfidfVectorizer)"],
                          horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
    # 设置参数
        max_features = st.slider("最大特征数（选出现频率最高的 n 个词作为特征）", 10, 500, 100)

    with col2:
        ngram_range = st.slider("N-gram范围（使用连续的 n 个词组合，如'机器学习'作为一个特征）", 1, 3, 1)

    st.info("""
- **词袋模型**📦 ：把文本转换成数字特征的方法。**核心**作用是统计词的出现次数，不考虑词的**顺序，语法**，即只看有什么词、出现多少次。
- **TF-IDF**🔍 ：是一种加权统计方法。**核心**作用是：衡量一个词在某篇文档中的**重要性**，既考虑词在当前文档的出现频率，也兼顾词在整个语料库中的稀缺性。
        """)
    # 初始化向量器
    if vec_method.startswith("词袋"):
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(ngram_range, ngram_range),
            stop_words="english" if lang == "英文" else None
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(ngram_range, ngram_range),
            stop_words="english" if lang == "英文" else None
        )
    
    # 拟合并转换
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # 展示结果
    st.write(f"向量化后形状: {X.shape} (样本数 × 特征数)")
    
    # 显示前5个样本的特征
    if len(texts) >= 5:
        st.subheader("样本特征示例")
        df = pd.DataFrame(
            X[:5].toarray(), 
            columns=feature_names,
            index=[f"样本 {i+1}" for i in range(5)]
        )
        st.dataframe(df.style.highlight_max(axis=1))
    
    # 解释向量化原理
    if vec_method.startswith("词袋"):
        st.info(f""" 
    - 向量化后形状：**{X.shape}** ， **{X.shape[0]}** 是样本数，**{X.shape[1]}** 是 最大特征数（词汇表的大小） 
    - 每一行对应 1 个样本，每一列对应 1 个特征，单元格里的数字是这个特征在该样本中出现的次数
    - ✅ 黄色单元格里的 1 ：表示这个特征在该样本中出现了 1 次
    - ✅ 单元格里的 0 ：表示这个特征在该样本中没有出现
        """)
    else:
        st.info(f""" 
    - 向量化后形状：**{X.shape}** ， **{X.shape[0]}** 是样本数，**{X.shape[1]}** 是 最大特征数（词汇表的大小） 
    - 每一行对应 1 个样本，每一列对应 1 个特征，单元格里的数字是这个特征在该样本中出现的次数
    - ✅ **TF（词频）**：某词在单篇文档中出现的**次数 / 该文档的总词数**，反映词对当前文档的“贡献度”   
    - ✅ **IDF（逆文档频率）**：**log (总文档数 / 包含该词的文档数)** ，反映词的 “稀缺性”（越稀有，IDF 越高，重要性越强）
    - ✅ **TF-IDF = TF × IDF**：最终得分越高，说明这个词是当前文档的 “核心特征词”
    """)
    
    return X, vectorizer, lang

# 文本预测功能
def text_prediction_demo(model, vectorizer, label_names, lang):
    """演示文本预测"""
    # 输入文本
    user_text = st.text_input("输入文本进行预测:", "这个产品很好，我非常满意" )
    
    if st.button("文本分类预测"):
        # 预处理
        processed_text = preprocess_text(user_text, is_chinese=(lang == "中文"))
        # 向量化
        text_vec = vectorizer.transform([processed_text])
        # 预测
        pred = model.predict(text_vec)[0]
        pred_proba = model.predict_proba(text_vec)[0].max()
        
        st.success(f"预测结果:  {st.session_state.en_label_names[pred]} / {st.session_state.cn_label_names[pred]}  (置信度: {pred_proba:.2f})")

        st.subheader("关键特征分析")
        st.info("""只有选择**逻辑回归**才能显示关键特征分析图，因为只有系数的模型才能分析 “特征重要性”
- ✅ **逻辑回归**：有coef_属性 → coef_里存的是 “每个词（特征）对 4 个类别（计算机图形学 / 曲棍球等）的权重值”，比如 “图形” 这个词对 “计算机图形学” 类别的系数为正且数值大，说明这个词能显著预测该类别。
- ❌ **朴素贝叶斯**：没有coef_属性 → 朴素贝叶斯是基于概率的模型，不计算特征系数，因此无法通过coef_分析特征重要性。
            """)        
        # 显示重要特征
        if hasattr(model, 'coef_'):
            # 获取特征重要性
            coefs = model.coef_[0]
            feature_names = vectorizer.get_feature_names_out()
            
            # 排序并显示
            top_n = min(10, len(feature_names))
            indices = np.argsort(np.abs(coefs))[-top_n:]
            top_features = [feature_names[i] for i in indices]
            top_coefs = [coefs[i] for i in indices]
            cols=st.columns([1,5,1])
            with cols[1]:            
                # 可视化
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=top_coefs, y=top_features, ax=ax)
                ax.set_title("对预测影响最大的特征")
                st.pyplot(fig)
            st.info("""
- 👉 **特征**就是词袋 / TF-IDF 生成的词汇表中的所有词（比如 “graphics”、“hockey”、“space” 等），数量等于你设置的 “最大特征数”。
- 👉 最终可视化的是 “权重绝对值 Top10 的特征”，而非所有特征。""")



# 各模块实现
def text_introduction_section():
    """文本分析基础介绍"""
    st.subheader("📚 文本分析基础")    
    st.markdown("### **什么是文本分析:📊**")
   
    st.info("""
##### 文本分析是从非结构化文本数据中提取有价值信息的过程，主要包括：
1. **文本分类**：将文本划分到预定义类别。例如新闻平台自动将稿件归类为“时政”、“娱乐”、“体育”等栏目
2. **情感分析**：判断文本情感倾向，如分析小红书美妆评论（正面/负面）、监测新车评测的质疑情绪
3. **主题提取**：识别核心主题，如职场论坛的“加班/薪资/晋升”、政策反馈的“实施细则/受益范围”
4. **命名实体识别**：识别人名、地名等，如在病历中提取“张三/冠心病”、新闻中提取“工商银行/上海”
    """)

    st.markdown("### **文本数据的特点:📊**")   
    st.info("""
##### 文本分析是从非结构化文本数据中提取有价值信息的过程，主要包括：
1. **非结构化**：无固定格式，如微信聊天、商品评论、手写病历等，无统一结构与规范字段
2. **高维度**：词汇表庞大，如商品评论含数万词汇，每个词汇均可视为一个数据维度
3. **稀疏性**：多数词汇在多数文本中不出现，如手机评论少美妆词汇，美妆评论少手机词汇
4. **语义复杂性**：存在一词多义、歧义，如“苹果”可指水果或品牌，“有点意思”可褒可贬，需结合语境判断。
    """)
    # 记录数据生成操作
    st.session_state.text_analysis_records["text_introduction_section"].append({
        "timestamp": datetime.now().timestamp()
    })

def text_preprocessing_section():
    """文本预处理模块"""
    st.subheader("✂️ 文本预处理")    
    st.markdown("""
    **预处理的目的:**
    清洗文本数据，去除噪声，标准化格式，为后续向量化做准备
    """)

    st.info("""
    **基本步骤:**
    1. 去除特殊字符和无关符号
    2. 大小写转换（英文）
    3. 分词（将句子拆分为词语）
    4. 去除停用词（如"的"、"是"、"the"等无实际意义的词）
    5. 词形还原/词干提取（英文）
    """)
    
    # 演示预处理效果
    st.subheader("预处理效果演示")
    lang = st.radio("选择语言", ["中文", "英文"])
    
    if lang == "中文":
        raw_text = st.text_input("输入中文文本:", "大家好！今天天气真不错，我们去公园玩吧！")
        processed_text = preprocess_text(raw_text, is_chinese=True)
        st.write("**原始文本**:", raw_text)
        st.write("**预处理后**:", processed_text)
        st.write("**分词结果**:", " / ".join(jieba.cut(raw_text)))
    else:
        raw_text = st.text_input("输入英文文本:", "Hello! Today is a beautiful day, let's go to the park!")
        processed_text = preprocess_text(raw_text)
        st.write("**原始文本**:", raw_text)
        st.write("**预处理后**:", processed_text)
    
    st.info("""
    **中文vs英文处理差异:**
    - 中文需要专门的分词工具（如 **jieba** ），英文可直接按空格分割
    - 英文有词形变化（复数、时态等），需要词干提取或词形还原
    - 中英文停用词表不同
    """)
    st.markdown("---")
    st.subheader("🔥 中文分词之挑战不可能！")
    # 定义难分词的测试句子列表
    test_sentences = [
        "南京市长江大桥",          
        "欢迎新老师生前来就餐",    
        "我想过过过儿过过的生活",        
        "下雨天留客天留我不留",   
        "乒乓球拍卖完了",
        "做核酸的队长死了",                      
    ]
    st.write("##### 🔤 选择/输入测试句子")
    # 选择预设句子
    selected_sentence = st.selectbox(
        "选择需要分词的句子",
        test_sentences,
        index=0,
        help="选择预设的易歧义句子测试分词效果"
    )
    # 确定最终要分词的句子
    target_sentence = selected_sentence

    # 分词处理（普通分词 + 精确模式 + 全模式）
    st.write("##### 📌 分词结果对比")
    st.caption("注：结巴分词已内置中文常用词库，对生僻词/人名可自定义添加词库")
    col1, col2, col3 = st.columns([1.05,1.3,1.2])

    # 1. 普通分词（默认精确模式）
    default_cut = jieba.lcut(target_sentence)
    with col1:
        st.markdown("**默认精确模式**--最常用，精准切分")
        st.write(" / ".join(default_cut))

    # 2. 全模式（找出所有可能的分词结果）
    full_cut = jieba.lcut(target_sentence, cut_all=True)
    with col2:
        st.markdown("**全模式**--穷尽所有可能，有冗余")
        st.write(" / ".join(full_cut))

    # 3. 搜索引擎模式（精确模式基础上，对长词再次切分）
    search_cut = jieba.lcut_for_search(target_sentence)
    with col3:
        st.markdown("**搜索引擎模式**--适合搜索场景")
        st.write(" / ".join(search_cut))
        
    # 记录数据生成操作
    st.session_state.text_analysis_records["text_preprocessing_section"].append({
        "lang":lang,
        "timestamp": datetime.now().timestamp()
    })
    
def text_analysis_section():
    """文本分类专项（适配5个类别）"""
    st.subheader("文本分类流程演示")
    st.write("### 1. 📊 文本分类数据展示")
    
    texts, labels, label_names, _ = load_sample_data("新闻主题分类")
    # 手动定义中文类别名（和JSON中一致）
    cn_label_names = ["计算机图形学", "摩托车", "棒球运动", "太空科学", "政治讨论"]
    
    st.write(f"💡 **数据集信息: {len(texts)}个样本，{len(label_names)}个类别**")
    
    if texts:
        category_data = {
            "类别编号": list(range(len(label_names))),  # 动态适配5个类别（0-4）
            "类别名称": label_names,  # 替换原st.session_state.en_label_names（避免依赖外部状态）
            "中文释义": cn_label_names  # 替换原st.session_state.cn_label_names
        }
        category_df = pd.DataFrame(category_data)
            
        st.dataframe(
            category_df,
            column_config={
                "类别编号": st.column_config.NumberColumn("🔢 类别编号", width="small"),
                "类别名称": st.column_config.TextColumn("📁 英文名称", width="medium"),
                "中文释义": st.column_config.TextColumn("🇨🇳 中文释义", width="medium")
            },
            hide_index=True,
            use_container_width=True
        )            
       
        # 显示样本
        st.write("**📁 样本示例**")
        sample_options = [
            f"样本{idx} - {label_names[labels[idx]]}" 
            for idx in range(min(10, len(texts)))  # 最多展示前10条样本
        ]
        # 创建下拉列表，默认选中第0条
        selected_sample = st.selectbox(
            "选择要查看的样本",
            options=sample_options,
            index=0,
            help="选择不同样本查看文本和对应标签"
        )
        # 解析选中的样本索引
        sample_idx = sample_options.index(selected_sample)
        # 展示样本内容
        st.write(f"文本: {texts[sample_idx]}")
        # ========== 修改点3：样本标签显示适配5个类别 ==========
        st.write(f"标签: {label_names[labels[sample_idx]]} | {cn_label_names[labels[sample_idx]]}")
        
        # 特征提取演示
        st.write("### 2. 📊 情感分析文本特征提取") 
        X, vectorizer, lang = demo_feature_extraction(texts, "英文")
        
        st.write("### 3. 📊 模型训练与评估")
        # ========== 修改点4：更新任务说明中的类别描述（4类→5类） ==========
        st.info("""
        ##### 👉 任务说明
        基于包含 500 个样本、覆盖 “计算机图形学”、“摩托车”、“棒球运动”、“太空科学”、“政治讨论” 等 5 类主题的新闻数据集，用**朴素贝叶斯**或**逻辑回归**模型完成文本分类任务。
        """)    
        
        # 划分训练集和测试集
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels,
            test_size=test_size,
            random_state=42, 
            stratify=labels  # 分层抽样，保证5个类别在训练/测试集中分布一致
        )

        # 选择模型
        model_name = st.selectbox("选择分类模型", ["朴素贝叶斯 (MultinomialNB)", "逻辑回归 (LogisticRegression)"])
        
        # 初始化模型
        if model_name.startswith("朴素"):
            model = MultinomialNB()
        else:
            model = LogisticRegression(max_iter=1000)
            
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 评估指标
        st.write("### 4. 📊 模型评估结果")
        acc = accuracy_score(y_test, y_pred)
        st.metric("准确率 (Accuracy)", f"{acc:.4f}")
        
        # 分类详细报告
        st.write("##### 📋 文本分类详细报告")
        # 解析classification_report为DataFrame
        report_dict = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
        # 剔除无关行（如accuracy），保留类别级指标
        report_df = pd.DataFrame(report_dict).T.drop(["accuracy", "macro avg", "weighted avg"])
        # 保留4位小数，优化显示
        report_df = report_df.round(4)
        
        # ========== 修改点5：分类报告中插入5个中文类别名 ==========
        report_df.insert(0, "类别名(CN)", cn_label_names)
        # 重置索引并将原索引（英文类别名）转为列
        report_df = report_df.reset_index().rename(columns={"index": "类别名(EN)"})
        # 重命名指标列为中文
        report_df.rename(columns={
            "precision": "精确率",
            "recall": "召回率",
            "f1-score": "F1分数",
            "support": "样本数"
        }, inplace=True)
        st.dataframe(report_df, use_container_width=True)
        
        # 混淆矩阵（中文标签，适配5个类别）
        st.write("##### 🔍 混淆矩阵")
        cols=st.columns([1,5,1])
        with cols[1]:
            # ========== 修改点6：混淆矩阵适配5个类别，调整图表大小避免拥挤 ==========
            fig, ax = plt.subplots(figsize=(10, 8))  # 增大图表尺寸（原8,6→10,8）
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cn_label_names)
            disp.plot(ax=ax, cmap="Blues", text_kw={"size": 16})  # 调大字体
            plt.title("混淆矩阵（中文标签）", fontsize=16)
            plt.xticks(rotation=15)  # 标签旋转避免重叠
            ax.set_xlabel('预测值', fontsize=14)
            ax.set_ylabel('真实值', fontsize=14)
            st.pyplot(fig)
            
        st.write("### 5. 📊 文本分类预测")   
        # 预设例句（对应摩托车、棒球、太空类）
        example_texts = {
            "摩托车类例句": "The motorcycle engine has a powerful 1000cc motor",
            "棒球类例句": "The baseball player hit a home run in the game",
            "太空类例句": "The rocket launched into space to explore Mars"
        }
        # 下拉选择例句
        selected_example = st.selectbox(
            "选择预设例句",
            options=list(example_texts.keys()),
            index=0
        )
        if st.button("文本分类预测", type="primary"):
            # 预处理
            processed_text = preprocess_text(selected_example, is_chinese=(lang == "中文"))
            # 向量化
            text_vec = vectorizer.transform([processed_text])
            
            # 预测
            pred_idx = model.predict(text_vec)[0]  # 预测的类别索引
            pred_proba = model.predict_proba(text_vec)[0].max()  # 最高置信度
            
            # ========== 核心修复2：使用数据集的真实类别名，而非session_state ==========
            pred_en = label_names[pred_idx]  # 数据集返回的英文类别
            pred_cn = cn_label_names[pred_idx]  # 对应的中文名称
            
            # 显示正确的预测结果
            st.success(f"预测结果:  {pred_en} / {pred_cn}  (置信度: {pred_proba:.2f})")
    
            # ========== 核心修复2：正确的特征重要性分析 ==========
            st.subheader("关键特征分析")
            if hasattr(model, 'coef_'):  # 仅逻辑回归有coef_属性
                st.info("✅ 逻辑回归模型 - 显示对当前预测类别影响最大的特征")
                
                # 修复：取当前预测类别的系数，而非第一个类别的系数
                coefs = model.coef_[pred_idx]
                feature_names = vectorizer.get_feature_names_out()
                
                # 取绝对值前10的特征（影响最大）
                top_n = min(10, len(feature_names))
                # 按系数绝对值排序，取top_n
                indices = np.argsort(np.abs(coefs))[-top_n:]
                top_features = [feature_names[i] for i in indices]
                top_coefs = [coefs[i] for i in indices]
                
                # 可视化特征重要性
                cols = st.columns([1, 5, 1])
                with cols[1]:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x=top_coefs, y=top_features, ax=ax, palette="coolwarm")
                    ax.set_title(f"对「{pred_cn}」类别影响最大的特征", fontsize=12)
                    ax.set_xlabel("特征系数（正负表示促进/抑制）", fontsize=10)
                    st.pyplot(fig)
                
                st.info("""
                📌 特征系数解读：
                - 正数：该词越频繁，越倾向于预测为当前类别；
                - 负数：该词越频繁，越不倾向于预测为当前类别；
                - 绝对值越大，特征对分类的影响越强。
                """)
            else:
                # 朴素贝叶斯无coef_属性，友好提示
                st.warning("""
                ❌ 朴素贝叶斯模型无法显示特征重要性：
                - 朴素贝叶斯是基于概率的模型，无特征系数（coef_）属性；
                - 如需分析特征重要性，请选择「逻辑回归 (LogisticRegression)」模型。
                """)
            

    # 记录数据生成操作
    st.session_state.text_analysis_records["text_analysis_section"].append({
        "selected_sample":selected_sample,
        "test_size":test_size,
        "model_name":model_name,
        "timestamp": datetime.now().timestamp()
    })

def sentiment_analysis_section():
    """情感分析专项模块"""
    st.subheader("😊 情感分析基础")
    
    st.markdown("""
    **什么是情感分析?**
    情感分析是文本分类的一种特殊形式，专注于识别文本中的主观情感倾向，主要包括：
    - 极性分析：正面、负面、中性
    - 情感强度分析：情感的强烈程度
    - 情感类型分析：喜悦、愤怒、悲伤等具体情感
    
    **应用场景:**
    - 产品评价分析
    - 社交媒体情感监测
    - 舆情分析
    - 客户反馈处理
    """)
    
    # 简单情感分析演示
    st.subheader("情感分析流程演示")
    texts, labels, label_names, _ , texts0= load_sample_data("中文情感分析")

    # 创建数据列表
    data = []
    for text in texts0[:5]:
        data.append({"文本内容": text, "类别": "正面"})        
    for text in texts0[25:30]:
        data.append({"文本内容": text, "类别": "负面"})
    # 转换为DataFrame
    df = pd.DataFrame(data)
    st.write("### 1. 📊 情感分析原始样本数据展示")
    st.dataframe(df, use_container_width=True)

    data = []
    for text in texts[:5]:
        data.append({"文本内容": text, "类别": "正面"})        
    for text in texts[25:30]:
        data.append({"文本内容": text, "类别": "负面"})
    # 转换为DataFrame
    df = pd.DataFrame(data)
    st.write("### 2. 📊 情感分析预处理后样本数据展示")
    st.dataframe(df, use_container_width=True)

    st.write("### 3. 📊 情感分析文本特征提取")    
    X, vectorizer, lang = demo_feature_extraction(texts, "中文")    
    st.write("### 4. 📊 模型训练与评估")
    
    st.info("""
    ##### 👉 任务说明
    基于包含 50 个样本、覆盖**正面** / **负面**两类情感倾向的评论文本数据集，用朴素贝叶斯或逻辑回归模型完成文本情感分析任务。    """)    
    # 划分训练集和测试集
    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=test_size,
        random_state=42, stratify=labels
    )

    # 选择模型
    model_name = st.selectbox("选择分类模型", ["朴素贝叶斯 (MultinomialNB)", "逻辑回归 (LogisticRegression)"])
    
    # 初始化模型
    if model_name.startswith("朴素"):
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000)
        
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估指标
    st.write("### 5. 📊 模型评估结果")
    acc = accuracy_score(y_test, y_pred)
    st.metric("准确率 (Accuracy)", f"{acc:.4f}")
    
    # 分类详细报告
    st.write("##### 📋 情感分类详细报告")
    report_dict = classification_report(
        y_test, y_pred, 
        target_names=label_names, 
        output_dict=True
    )

    # 剔除无关行，保留正负两类
    report_df = pd.DataFrame(report_dict).T.drop(["accuracy", "macro avg", "weighted avg"])
    report_df = report_df.round(4)

    # 重置索引，英文类别名列
    report_df = report_df.reset_index().rename(columns={"index": "情感类别"})
    # 重命名指标列为中文
    report_df.rename(columns={
        "precision": "精确率",
        "recall": "召回率",
        "f1-score": "F1分数",
        "support": "样本数"  
    }, inplace=True)
    st.dataframe(report_df, use_container_width=True)

    
    # 混淆矩阵（中文标签，核心修改）
    st.write("##### 🔍 混淆矩阵")
    cols=st.columns([1,3,1])
    with cols[1]:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(ax=ax, cmap="Blues",text_kw={"size": 30})
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        plt.title("混淆矩阵", fontsize=14)
        ax.set_xlabel('预测值',fontsize=12)
        ax.set_ylabel('真实值',fontsize=12)
        st.pyplot(fig)
    
    # 实时预测
    st.write("### 6. 📊 文本情感预测")
    user_comment = st.selectbox("选择商品评论",
                                ["手机发热严重，玩10分钟游戏就烫手，不敢长时间用",
                                 "第一次用这个牌子，没想到这么好用，超出预期",
                                 "虽然发货慢了两天，但包装很用心，产品没瑕疵，满意",
                                 "功能鸡肋，很多用不上的设计，徒增复杂度"])
    if st.button("分析文本情感"):
        processed = preprocess_text(user_comment, is_chinese=True)
        vec = vectorizer.transform([processed])
        pred = model.predict(vec)[0]
        st.success(f"情感预测: {label_names[pred]}")

    # 记录数据生成操作
    st.session_state.text_analysis_records["sentiment_analysis_section"].append({
        "test_size":test_size,
        "model_name":model_name,
        "user_comment":user_comment,
        "timestamp": datetime.now().timestamp()
    })

def native_bys_section():
    # 页面标题
    st.header('🛡️ 朴素贝叶斯算法应用-诈骗短信识别')
    st.subheader('📚 朴素贝叶斯算法及基本概念')
    st.info("""
    朴素贝叶斯（Naive Bayes）是一种基于贝叶斯定理和特征条件独立性假设的分类算法，在文本分类、垃圾邮件过滤等领域有广泛应用。

    ##### 💡 核心概念
    1. **贝叶斯定理**：描述了后验概率与先验概率、似然概率的关系，公式为：
       $P(C|X) = \\frac{P(X|C) \\cdot P(C)}{P(X)}$
       其中：
       - $P(C|X)$：后验概率（已知特征X时，类别C的概率）
       - $P(C)$：先验概率（类别C的固有概率）
       - $P(X|C)$：似然概率（类别C中出现特征X的概率）
       - $P(X)$：证据因子（特征X的边际概率）

    2. **特征条件独立性假设**：假设所有特征之间相互独立，简化计算复杂度，这也是"朴素"（Naive）一词的由来。

    3. **文本分类应用**：
       - 将文本拆分为词语作为特征
       - 计算不同类别（如正常/诈骗短信）中词语出现的概率
       - 通过后验概率比较判断文本类别
    """)
    st.markdown("---")
    
    st.subheader('📝 应用场景说明')
    st.markdown("""
    随着移动互联网的发展，诈骗短信已成为影响用户财产安全的重要威胁。常见诈骗手段包括：
    - **冒充公检法**：以账户异常、涉嫌违法等理由要求转账
    - **中奖诈骗**：声称中奖需先缴纳手续费
    - **金融诈骗**：低息贷款、信用卡提额等诱饵
    - **冒充熟人**：伪装成领导、亲友要求转账
    - **钓鱼链接**：通过短信链接窃取个人信息
    
    本系统采用朴素贝叶斯算法，通过分析短信内容特征，自动识别诈骗短信，保护用户财产安全。
    """)
    st.markdown("---")
    
    # 准备数据
    train_data, normal_sms, fraud_sms = native_bys.prepare_training_data()
    
    # 使用sklearn的train_test_split划分训练集和测试集（8:2比例）
    train_set, test_set = train_test_split(
        train_data,
        test_size=0.2,  # 测试集占比20%
        random_state=42,  # 固定随机种子，保证结果可复现
        stratify=[d["label"] for d in train_data]  # 按标签分层抽样，保持类别比例
    )
    
    # 训练模型（仅使用训练集）
    with st.spinner("正在训练模型..."):
        model_params = native_bys.train_model(train_set)
    
    # 展示数据统计（包含数据集划分信息）
    native_bys.show_data_statistics(normal_sms, fraud_sms, len(train_set), len(test_set))
    
    # 展示关键词分析
    native_bys.show_keyword_analysis(model_params)
    
    # 展示模型评估（使用测试集）
    native_bys.evaluate_model(model_params, test_set)
    
    # 主交互界面
    user_guess=native_bys.main_interface(model_params)
    
    # 记录数据生成操作
    st.session_state.text_analysis_records["native_bys_section"].append({
        "user_guess":user_guess,
        "timestamp": datetime.now().timestamp()
    })

def quiz_section():
    st.header("🎯 文本分析概念测验")
    st.write("请完成以下5道单选题，全部答完后可提交查看结果")
    
    # 定义测验题目、选项、正确答案及解析（聚焦文本分类和情感分析）
    quiz_data = [
        {
            "question": "1. 文本分类中，词袋模型（Bag of Words）的核心思想是什么？",
            "options": [
                "A. 保留文本中词语的顺序和语法结构",
                "B. 将文本表示为词汇出现频率的向量，忽略词序",
                "C. 只能处理英文文本，无法处理中文分词",
                "D. 自动提取文本的情感倾向"
            ],
            "correct": "B",
            "explanation": "词袋模型将文本视为词汇的集合，通过统计每个词的出现频率构建特征向量，不考虑词语的顺序和语法关系，是文本分类中最基础的特征提取方法。"
        },
        {
            "question": "2. 情感分析与普通文本分类的主要区别在于？",
            "options": [
                "A. 情感分析只能处理中文，普通文本分类处理英文",
                "B. 情感分析专注于识别主观情感倾向（如正负向），普通分类侧重客观类别划分",
                "C. 情感分析不需要预处理，普通文本分类需要分词",
                "D. 情感分析只能用朴素贝叶斯算法"
            ],
            "correct": "B",
            "explanation": "情感分析是文本分类的特殊形式，核心任务是识别文本中的主观情感（如正面、负面、中性），而普通文本分类更关注客观类别的划分（如新闻主题、邮件类型等）。"
        },
        {
            "question": "3. TF-IDF特征提取中，IDF（逆文档频率）的作用是？",
            "options": [
                "A. 惩罚在多数文档中频繁出现的常见词（如“的”“是”）",
                "B. 增加高频词的权重，突出其重要性",
                "C. 确保每个文本的特征向量长度相同",
                "D. 自动去除文本中的特殊符号和数字"
            ],
            "correct": "A",
            "explanation": "IDF通过计算“log(总文档数/包含该词的文档数)”，降低在多数文档中都出现的常见词（如停用词）的权重，同时提升在少数文档中出现的稀有词的权重，更能反映词的区分度。"
        },
        {
            "question": "4. 以下哪种情况可能导致文本分类模型的测试准确率远低于训练准确率？",
            "options": [
                "A. 训练数据量过大",
                "B. 模型出现过拟合，过度学习训练数据中的噪声",
                "C. 使用了TF-IDF而非词袋模型",
                "D. 测试集与训练集分布一致"
            ],
            "correct": "B",
            "explanation": "过拟合是文本分类中常见的问题，表现为模型在训练数据上表现极好，但在未见过的测试数据上表现差，因为模型过度学习了训练数据中的细节（包括噪声），而没有抓住通用规律。"
        },
        {
            "question": "5. 中文文本预处理中，分词的主要目的是？",
            "options": [
                "A. 将英文单词转换为中文翻译",
                "B. 去除文本中的标点符号和特殊字符",
                "C. 将连续的中文句子拆分为有意义的词语单元，便于后续特征提取",
                "D. 直接计算文本的情感得分"
            ],
            "correct": "C",
            "explanation": "中文文本没有像英文那样的空格分隔，分词是将连续的字符序列拆分为有意义的词语（如将“南京市长江大桥”拆分为“南京市 / 长江大桥”），是中文文本特征提取的必要前置步骤。"
        }
    ]
    
    # 初始化会话状态存储用户答案
    if "text_analysis_user_answers" not in st.session_state:
        st.session_state.text_analysis_user_answers = [None] * len(quiz_data)
    
    # 显示所有题目和选项（初始无选中状态）
    for i, item in enumerate(quiz_data):
        st.markdown(f"**{item['question']}**")
        # 设置默认值为None实现初始无选中状态，通过会话状态保存答案
        answer = st.radio(
            "选择答案:",
            item["options"],
            key=f"text_quiz_{i}",
            index=None,  # 关键：初始无选中项
            label_visibility="collapsed"
        )
        
        # 更新会话状态中的答案（提取选项字母A/B/C）
        if answer is not None:
            st.session_state.text_analysis_user_answers[i] = answer[0]
    
    # 检查是否所有题目都已作答
    all_answered = all(ans is not None for ans in st.session_state.text_analysis_user_answers)
    
    # 提交按钮：只有全部答完才可用
    submit_btn = st.button(
        "提交答案", 
        key="submit_text_quiz",
        disabled=not all_answered  # 未答完时禁用
    )
    
    # 未答完时显示提示
    if not all_answered:
        st.info("请完成所有5道题目后再提交")
    
    # 处理提交
    if submit_btn and all_answered:
        # 计算得分和错误题目
        score = 0
        results = []
        incorrect_questions = []
        for i, item in enumerate(quiz_data):
            is_correct = st.session_state.text_analysis_user_answers[i] == item["correct"]
            if is_correct:
                score += 20  # 每题20分
            else:
                incorrect_questions.append({
                    "topic": item["question"], 
                    "user_answer": st.session_state.text_analysis_user_answers[i]
                })

            results.append({
                "question": item["question"],
                "user_answer": st.session_state.text_analysis_user_answers[i],
                "correct_answer": item["correct"],
                "is_correct": is_correct,
                "explanation": item["explanation"]
            })
        
        # 确保结果记录的会话状态存在
        if "text_analysis_records" not in st.session_state:
            st.session_state.text_analysis_records = {}
        
        # 记录测验结果（添加时间戳）
        st.session_state.text_analysis_records["text_analysis_quiz"] = {
            "score": score,
            "incorrect_questions": incorrect_questions,
            "timestamp": datetime.now().timestamp()
        }
       
        # 显示得分
        st.success(f"📊 测验完成！你的得分是：{score}分")
        st.write("### 答案解析：")
        
        # 显示每题结果
        for res in results:
            # 使用emoji和文字标记正确/错误状态
            if res["is_correct"]:
                status_text = "✅ 正确"
            else:
                status_text = "❌ 错误"
            
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
        以下是学生在文本分析测验中的答题情况：
        - 总得分：{score}分
        - 错误题目：{len(incorrect_topics)}道
        - 错误知识点：{'; '.join(incorrect_topics) if incorrect_topics else '无'}
        
        请分析该学生的知识掌握情况，指出未掌握的核心概念，并给出具体的学习建议和指导方向，帮助学生针对性提升。
        答案必须控制在450字以内
        """
        
        # 调用AI分析
        with st.spinner("AI正在分析你的答题情况..."):
            ai_analysis = ask_ai_assistant(analysis_prompt, "文本分析测验分析")
        
        # 显示AI分析结果
        st.write("### 🤖 AI学习诊断：")
        st.info(ai_analysis)       
  
    return "概念测验模块：完成5题单选题测试"


# 主程序
def main():
    init_session_state()
    
    # 初始化会话状态
    if 'section' not in st.session_state:
        st.session_state.section = "文本分析基础"

    if "label_mapping" not in st.session_state:
        st.session_state.label_mapping = {
            "comp.graphics": "计算机图形学",
            "rec.sport.hockey": "休闲体育-曲棍球",
            "sci.space": "科学-航天/太空",
            "talk.politics.misc": "讨论-政治杂项"
        }
        # 拆分出英文/中文类别名列表（存入Session，方便直接调用）
        st.session_state.en_label_names = list(st.session_state.label_mapping.keys())
        st.session_state.cn_label_names = list(st.session_state.label_mapping.values())

    # 记录模块访问顺序
    current_section = st.session_state.section
    st.session_state.text_analysis_records["module_sequence"].append(current_section)
    if current_section not in st.session_state.text_analysis_records["module_timestamps"]:
        st.session_state.text_analysis_records["module_timestamps"][current_section] = {
            "enter_time": time.time()
        } 
        
    # 侧边栏导航
    st.sidebar.title("导航菜单")
    section = st.sidebar.radio("选择学习模块", [
        "文本分析基础",
        "文本预处理",
        "文本分类专项",
        "情感分析专项",
        "朴素贝叶斯算法",
        "概念测验",
        "编程实例（新闻文本数据集）"
    ])
    
    # 显示对应模块编程实例模块: 贝叶斯文本分类分步编程训练"
    st.session_state.section = section    
    context = ""
    if section == "文本分析基础":
        text_introduction_section()
    elif section == "文本预处理":
        text_preprocessing_section()
    elif section == "文本分类专项":
        text_analysis_section()
    elif section == "情感分析专项":
        sentiment_analysis_section()
    elif section == "朴素贝叶斯算法":
        native_bys_section()
    elif section == "概念测验":
        quiz_section()
    elif section == "编程实例（新闻文本数据集）":
        # 初始化step变量（如果不存在）
        if 'step' not in st.session_state:
            st.session_state.step = 0
        bayes_text_classification_step_by_step.main()
        context = "编程实例模块: 朴素贝叶斯文本分类分步编程训练"

    # 显示聊天界面
    display_chat_interface(context)
    
    # 记录模块退出时间
    if current_section in st.session_state.text_analysis_records["module_timestamps"]:
        st.session_state.text_analysis_records["module_timestamps"][current_section]["exit_time"] = datetime.now().timestamp()
    
    if section != "编程实例（新闻文本数据集）":
        # 侧边栏添加学习报告按钮（调用独立模块）
        st.sidebar.markdown("---")
        if st.sidebar.button("文本分析模块学习报告"):
            report = generate_evaluation(
                module_type="text_analysis",
                raw_records=st.session_state.text_analysis_records
            )
            st.write("### 文本分析学习情况报告")
            st.info(report)
            
    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    本平台帮助学习文本分析基础知识：
    - 文本预处理方法
    - 特征提取技术（词袋模型、TF-IDF）
    - 文本分类算法
    - 情感分析基础
    """)

if __name__ == "__main__":
    main()









