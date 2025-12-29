import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ====================== 设置中文字体 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 生成线性可分的二分类数据集
np.random.seed(42)  # 固定随机种子，保证结果可复现
n_samples = 100  # 样本数量
X = np.random.randn(n_samples, 2)  # 特征：二维数据
# 生成标签：基于线性边界 y = 0.5x + 1，上方为1，下方为-1
y = np.where(0.5 * X[:, 0] + 1 < X[:, 1], 1, -1)

# 2. 定义感知器类
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iter=10):
        self.lr = learning_rate  # 学习率
        self.n_iter = n_iter  # 迭代次数
        self.weights = None  # 权重（包含偏置，权重的第一个元素是偏置b）
        self.errors_ = []  # 记录每次迭代的错误分类数

    def fit(self, X, y):
        # 初始化权重：偏置b + 两个特征的权重w1、w2
        self.weights = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # 感知器更新规则：Δw = lr * (y - y_pred) * x
                update = self.lr * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update  # 偏置b的更新（xi的常数项为1）
                errors += int(update != 0.0)  # 统计错误分类的样本数
            self.errors_.append(errors)
            if errors == 0:  # 没有错误分类，提前收敛
                break
        return self

    def net_input(self, X):
        # 计算净输入：z = b + w1*x1 + w2*x2
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        # 激活函数：阶跃函数，返回1或-1
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# 3. 初始化绘图（左右两个子图）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 缩小画布尺寸
fig.suptitle('感知器模型训练过程可视化', fontsize=16)

# 左侧子图：决策边界可视化
ax1.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax1.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax1.set_xlabel('特征1 (x1)')
ax1.set_ylabel('特征2 (x2)')
ax1.set_title('决策边界动态变化')


# 绘制数据集：正样本（1）用蓝色圆圈，负样本（-1）用红色叉号
scatter_pos = ax1.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='正样本 (1)')
scatter_neg = ax1.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='x', label='负样本 (-1)')


# 初始化分类线（感知器的超平面）
x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 2)
line, = ax1.plot(x_line, np.zeros_like(x_line), color='green', linewidth=2, label='分类超平面')
ax1.legend(loc='lower left')

# 添加权重显示文本框（位于左下角）
weight_text = ax1.text(
    0.02, 0.98,  # 左下角相对坐标
    '',
    transform=ax1.transAxes,
    fontsize=14,
    verticalalignment='top',  # 垂直顶部对齐
    horizontalalignment='left',  # 水平左对齐
    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
)

# 右侧子图：损失下降过程
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('错误分类数（损失）')
ax2.set_title('错误分类数变化过程')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 15)
ax2.grid(True, linestyle='--', alpha=0.7)
loss_line, = ax2.plot([], [], 'b-', linewidth=2)  # 移除标记点减少绘制量
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# 初始化感知器
perceptron = Perceptron(learning_rate=0.1, n_iter=15)
perceptron.weights = np.zeros(1 + X.shape[1])
perceptron.errors_ = []
current_iter = 0
errors = 0
sample_idx = 0  # 遍历样本的索引

# 4. 优化的更新函数
def update(frame):
    global current_iter, errors, sample_idx, perceptron

    # 每N个样本才更新一次画面（减少绘制频率）
    update_frequency = 2  # 每隔2个样本更新一次画面
    if sample_idx % update_frequency != 0 and sample_idx != len(X)-1:
        xi = X[sample_idx]
        target = y[sample_idx]
        y_pred = perceptron.predict(xi)
        update = perceptron.lr * (target - y_pred)
        perceptron.weights[1:] += update * xi
        perceptron.weights[0] += update
        errors += int(update != 0.0)
        sample_idx += 1
        return line, weight_text, loss_line  # 不更新画面直接返回

    # 执行正常更新逻辑
    if sample_idx < len(X):
        xi = X[sample_idx]
        target = y[sample_idx]
        y_pred = perceptron.predict(xi)
        update = perceptron.lr * (target - y_pred)
        perceptron.weights[1:] += update * xi
        perceptron.weights[0] += update
        errors += int(update != 0.0)
        sample_idx += 1
    else:
        perceptron.errors_.append(errors)
        current_iter += 1
        print(current_iter)
        sample_idx = 0
        errors = 0
        if (perceptron.errors_[-1] == 0) or (current_iter >= perceptron.n_iter):
            ani.event_source.stop()
            weight_text.set_text(f'收敛权重：\nb = {perceptron.weights[0]:.4f}\nw₁ = {perceptron.weights[1]:.4f}\nw₂ = {perceptron.weights[2]:.4f}')

    # 更新分类超平面
    w = perceptron.weights
    if w[2] != 0:
        y_line = (-w[0] - w[1] * x_line) / w[2]
        line.set_ydata(y_line)
    
    ax1.set_title('决策边界动态变化')

    # 更新权重文本
    weight_text.set_text(f'正在进行第 {current_iter+1} 次迭代\n当前权重：\nb = {w[0]:.4f}\nw1 = {w[1]:.4f}\nw2 = {w[2]:.4f}')
    
    # 更新损失图
    if perceptron.errors_:
        loss_x = list(range(len(perceptron.errors_)))
        loss_y = perceptron.errors_
        loss_line.set_xdata(loss_x)
        loss_line.set_ydata(loss_y)
        ax2.relim()
        ax2.autoscale_view()

    return line, weight_text, loss_line

# 5. 创建动画（优化参数）
ani = FuncAnimation(
    fig, update, 
    frames=range(perceptron.n_iter * len(X)), 
    interval=30,  # 适当增大间隔，减少绘制压力
    blit=True,  # 开启用blit加速渲染
    repeat=False
)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
