# cover.py
from openai import OpenAI
#import os
'''

# 从环境变量中获取密钥：os.getenv("变量名")，变量名建议大写，比如DEEPSEEK_API_KEY
#DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# 判空：如果没获取到环境变量，提示错误（避免运行时崩溃）
if not DEEPSEEK_API_KEY:
    raise ValueError("请先设置 DEEPSEEK_API_KEY 环境变量！")
'''
# 创建DeepSeek客户端并导出
client = OpenAI(
    api_key='sk-a49001abe22b430fa441ef79316391b4',
    base_url="https://api.deepseek.com"
)

# AI助教功能函数
def ask_ai_assistant(question, context=""):
    try:
        system_prompt = f"""
        你正在一个交互式学习平台中帮助学生。
        你回答他们的问题，或者对他们的学习情况做出评价
        当前上下文: {context}        
        请用中文回答学生的问题，保持专业友好，且饱含鼓励肯定的语气。
        适当使用emoj，使回答更生动。
        你的回答字数一定在500字以内
        解释概念时请尽量使用比喻和直观的例子。
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"出错了: {str(e)}。请检查您的API密钥是否正确或网络连接。"
