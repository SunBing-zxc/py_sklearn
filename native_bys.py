import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import jieba
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.metrics import accuracy_score, confusion_matrix

# 准备训练数据（扩大样本量）
def prepare_training_data():
    # 正常短信样本
    normal_sms = [
        "您的话费余额不足10元，请及时充值，避免停机影响使用",
        "【快递通知】您的包裹已到达小区驿站，取件码123456，有效期2天",
        "本周六下午3点同学聚会，地点在学校门口餐厅，收到请回复",
        "电费提醒：您家7月电费150.50元，截止日期8月10日，可通过APP缴纳",
        "天气预报：明天多云转小雨，气温24-30℃，记得带伞",
        "【银行】您的储蓄卡账户于08:30存入工资5000元，余额12500元",
        "家长会通知：本周五下午4点在教室召开，请勿迟到",
        "您订购的图书已发货，快递单号SF1234567890",
        "提醒：明天是您的生日，祝您生日快乐！",
        "【外卖】您点的餐品已由骑手李师傅接单，预计30分钟送达",
        "您的信用卡账单已出，本期应还金额2350元，还款日9月5日",
        "小区通知：明天上午9点将停水检修，预计3小时",
        "张老师，我是学生家长李涛，孩子校服尺寸选错了，麻烦加我微信 138xxxx5678 发下正确尺码表，着急下周统一调换",
        "您的会员积分即将到期，可兑换礼品或抵扣现金",
        "【交通违章】您的车辆于XX路有一次违章停车记录，可网上处理",
        "公司通知：下周一上午10点召开全体员工大会，请准时参加",
        "【银行】您尾号 3456 的储蓄卡于 15:23 支出 2000 元（代缴物业费），如有疑问请拨打 955XX",
        "请点击链接https://work.weixin.qq.com/s/xxx 填写本周部门团建报名信息，截止今天 18 点 ",
    ]
    
    # 诈骗短信样本
    fraud_sms = [
        "恭喜您获得一台笔记本电脑，填写收货地址即可免费领取，限今日",        
        "恭喜您中了二等奖5万元！请提供银行卡号和身份证号领取（兑奖码：68XX）",
        "我是您领导王总，明天到我办公室一趟，有笔紧急款项需要你帮忙周转",
        "您的快递丢失，点击理赔链接填写信息即可获赔200元，24小时内有效",
        "免费领取500元手机话费！回复1即可办理，仅限今日，先到先得",
        "【法院通知】您有一张传票未领取，请立即联系010-12345678核实",
        "低息贷款，无抵押，秒批到账，最高50万，点击链接快速办理",
        "您的孩子在学校突发疾病，正在医院抢救，急需缴纳手术费，速转5万元到账户XXX",
        "兼职刷单，日入300-500元，无需押金，扫码加客服了解详情",
        "【移动客服】您本月消费达标，可免费领取 20G 流量包，点击链接https://10086-verify.cn 验证领取（1 小时内有效）",
        "【系统提示】您的微信账号存在安全风险，点击链接完成实名认证",
        "我是你朋友，我在外地出差遇急事，急需用钱，先转2万元到这个账户",
        "您被选为幸运用户，可免费领取一台智能手机，只需支付29元运费",
        "检测到您的社保账户未年审，逾期将停用，点击链接办理",
        "您在京东购买的运动鞋质检时发现轻微瑕疵，可补偿 50 元无门槛券，回复【同意】领取，客服将同步券码",
        "高额信用卡快速办理，无需征信，额度5-50万，联系电话138xxxx8888",
        "【紧急通知】您的账户存在异常，点击链接https://xxx验证身份，否则将冻结账户",
        "您的快递因地址模糊无法派送，联系派件员 135xxxx9012 或点击https://kd100.com/xxx 补充信息 ",
    ]
    
    # 整理成带标签的数据集
    train_data = []
    for sms in normal_sms:
        train_data.append({"text": sms, "label": 0})  # 0表示正常
    for sms in fraud_sms:
        train_data.append({"text": sms, "label": 1})  # 1表示诈骗
    
    return train_data, normal_sms, fraud_sms

# 数据预处理函数
def preprocess_text(text):
    """中文文本预处理：去除特殊字符、分词"""
    # 去除特殊字符和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # 分词
    words = jieba.cut(text)
    return " ".join(words)

# 训练模型（使用训练集）
def train_model(train_set):
    # 1. 统计先验概率 P(C)
    total_sms = len(train_set)
    p0 = sum(1 for d in train_set if d["label"] == 0) / total_sms  # 正常短信先验概率
    p1 = sum(1 for d in train_set if d["label"] == 1) / total_sms  # 诈骗短信先验概率
    
    # 2. 统计每个类别下的词频（加拉普拉斯平滑）
    word_count_0 = defaultdict(int)  # 正常短信词频
    word_count_1 = defaultdict(int)  # 诈骗短信词频
    total_words_0 = 0  # 正常短信总词数
    total_words_1 = 0  # 诈骗短信总词数
    
    # 分词+统计
    for data in train_set:
        processed_text = preprocess_text(data["text"])
        words = processed_text.split()
        if data["label"] == 0:
            for word in words:
                word_count_0[word] += 1
                total_words_0 += 1
        else:
            for word in words:
                word_count_1[word] += 1
                total_words_1 += 1
    
    # 所有唯一词汇（用于平滑）
    all_words = set(list(word_count_0.keys()) + list(word_count_1.keys()))
    vocab_size = len(all_words)
    alpha = 1  # 拉普拉斯平滑系数
    
    return {
        "p0": p0,
        "p1": p1,
        "word_count_0": word_count_0,
        "word_count_1": word_count_1,
        "total_words_0": total_words_0,
        "total_words_1": total_words_1,
        "vocab_size": vocab_size,
        "alpha": alpha
    }

# 预测函数
def predict_sms(model_params, text):
    """预测短信是否为诈骗"""
    processed_text = preprocess_text(text)
    words = processed_text.split()
    
    p0 = model_params["p0"]
    p1 = model_params["p1"]
    word_count_0 = model_params["word_count_0"]
    word_count_1 = model_params["word_count_1"]
    total_words_0 = model_params["total_words_0"]
    total_words_1 = model_params["total_words_1"]
    vocab_size = model_params["vocab_size"]
    alpha = model_params["alpha"]
    
    # 计算 P(X|C0)：正常短信下出现这些词的概率
    p_x_c0 = 1.0
    for word in words:
        count = word_count_0.get(word, 0)
        p_word_c0 = (count + alpha) / (total_words_0 + alpha * vocab_size)
        p_x_c0 *= p_word_c0
    
    # 计算 P(X|C1)：诈骗短信下出现这些词的概率
    p_x_c1 = 1.0
    for word in words:
        count = word_count_1.get(word, 0)
        p_word_c1 = (count + alpha) / (total_words_1 + alpha * vocab_size)
        p_x_c1 *= p_word_c1
    
    # 后验概率（忽略P(X)，直接比分子）
    p_c0_x = p_x_c0 * p0
    p_c1_x = p_x_c1 * p1
    
    # 归一化
    total = p_c0_x + p_c1_x
    p_c0_x_norm = p_c0_x / total if total != 0 else 0
    p_c1_x_norm = p_c1_x / total if total != 0 else 0
    
    # 预测类别
    pred_label = 0 if p_c0_x > p_c1_x else 1
    pred_label_name = "正常短信" if pred_label == 0 else "诈骗短信"
    
    return {
        "pred_label": pred_label,
        "pred_label_name": pred_label_name,
        "p_normal": p_c0_x_norm,
        "p_fraud": p_c1_x_norm,
        "p_x_c0": p_x_c0,
        "p_x_c1": p_x_c1,
        "processed_text": processed_text
    }

# 展示训练数据统计
def show_data_statistics(normal_sms, fraud_sms, train_size, test_size):
    st.subheader('📊 训练数据统计')
    
    col1, col2 = st.columns([2,3])
    with col1:
        st.markdown(f"### 数据集划分")
        st.write(f"- **总样本量**：{len(normal_sms) + len(fraud_sms)}条")
        st.write(f"- **正常短信**：{len(normal_sms)}条    **诈骗短信**：{len(fraud_sms)}条")
        st.write(f"- **训练集**：{train_size}条（80%）    **测试集**：{test_size}条（20%）")
   
    with col2:
        st.markdown("### 样本示例")
        st.info(f"""✔️正常短信示例：{normal_sms[0]}
            """)
        st.success(f"""❌诈骗短信示例：{fraud_sms[0]}
            """)


# 展示关键词分析
def show_keyword_analysis(model_params):
    st.subheader('🔑 关键词分析')
    
    # 获取高频词
    normal_top_words = sorted(
        model_params["word_count_0"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    fraud_top_words = sorted(
        model_params["word_count_1"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 正常短信高频词")
        normal_df = pd.DataFrame(normal_top_words, columns=["词语", "出现次数"])
        st.dataframe(normal_df, use_container_width=True)

    
    with col2:
        st.markdown("### 诈骗短信高频词")
        fraud_df = pd.DataFrame(fraud_top_words, columns=["词语", "出现次数"])
        st.dataframe(fraud_df, use_container_width=True)


# 主交互界面
def main_interface(model_params):
    st.markdown('#### 🔍 短信检测工具')
    
    # 预设短信选项
    preset_sms = [
"【超市通知】您上周购买的日用品已参与满减活动，退款 25 元将在 3 个工作日内退回原支付账户，请注意查收",
"您有一份未领取的周年庆礼品，内含价值 500 元购物卡，点击链接https://gift888.com填写地址即可免费领取，24 小时内有效",
"孩子学校组织周末研学活动，费用 180 元 / 人，需在周五前通过学校公众号缴费，详情已发至班级群"
    ]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_sms = st.selectbox("选择预设短信", preset_sms)
    with col2:
        user_guess = st.radio("你的判断", ["正常短信", "诈骗短信"], key="user_guess",horizontal=True)

       
    if st.button("开始检测"):
        # 执行预测
        result = predict_sms(model_params, selected_sms)                
        st.subheader("检测结果")
           
        # 展示结果对比
        col1, col2, col3,col4 = st.columns([1,1,0.1,1])
        with col1:
            st.info(f"你的判断：{user_guess}")
        with col2:
            if result["pred_label"] == 1:
                st.error(f"算法判断：{result['pred_label_name']}")
            else:
                st.success(f"算法判断：{result['pred_label_name']}")
        with col4:
            # 结果对比
            if user_guess == result['pred_label_name']:
                st.warning("✅ 恭喜！你的判断与算法一致～")
            else:
                st.warning("❌ 你的判断与算法不一致")
            
        # 展示概率
        prob_data = {
            "类型": ["正常短信", "诈骗短信"],
            "概率": [result["p_normal"], result["p_fraud"]]
        }
        cols=st.columns([2,1])
        with cols[0]:
            # 以表格形式显示数据
            st.subheader("概率分布表")
            st.dataframe(
                prob_data,
                use_container_width=True
            )
    return user_guess

# 模型评估（使用测试集）
def evaluate_model(model_params, test_set):
    st.subheader('📈 模型评估（基于测试集）')
    
    # 测试模型在测试集上的表现
    y_true = [d["label"] for d in test_set]
    y_pred = [predict_sms(model_params, d["text"])["pred_label"] for d in test_set]
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    st.write(f"### 模型准确率：{accuracy:.2%}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 以表格形式显示混淆矩阵
    cols=st.columns([2,1])
    with cols[0]:
        st.markdown("#### 🔍 混淆矩阵")
        cm_df = pd.DataFrame(
            cm,
            columns=["预测为正常短信", "预测为诈骗短信"],
            index=["实际为正常短信", "实际为诈骗短信"]
        )
        st.dataframe(cm_df, use_container_width=True)
    
