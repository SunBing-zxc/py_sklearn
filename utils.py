# utils.py
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_chinese_font():
    # 字体加载逻辑（和之前一致，仅需写一次）
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansSC-Regular.ttf")
    if not os.path.exists(font_path):
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        return False
    try:
        font_prop = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        return False
