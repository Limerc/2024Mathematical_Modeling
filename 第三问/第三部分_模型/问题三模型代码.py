import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# 粒子群算法的参数
w = 0.25  # 惯性权重
c1 = 2  # 个体学习因子   #初始值为1.5
c2 = 2  # 社会学习因子  #初始值为1.5
num_particles = 30  # 粒子数量
max_iter = 10  # 最大迭代次数
num_simulations = 100  # 蒙特卡罗模拟次数

# 问题参数（以你的符号定义）
num_years = 7  # 从2024到2030年
num_crops = 41  # 作物种类数
num_plots = 54  # 地块数量
num_seasons = 2  # 季节数量

# 读取附件2.xlsx中的数据
file_path = 'C:/Users/26962/PycharmProjects/Math/data2/附件2.xlsx'

# 读取销售价格（p）、种植成本（c）、亩产量（q）等数据
data_stats = pd.read_excel(file_path, sheet_name='2023年统计的相关数据')
#读取附件1.xlsx中的数据
file_path2 = 'C:/Users/26962/PycharmProjects/Math/data2/附件1.xlsx'
# 将数据转换为字符串，以防止非字符串类型导致错误
data_stats['销售单价/(元/斤)'] = data_stats['销售单价/(元/斤)'].astype(str)
data_stats2 = pd.read_excel(file_path2,sheet_name='乡村的现有耕地')
A_j = data_stats2['地块面积/亩']
plot_types = data_stats2['地块类型'].map({
    '平旷地': 1, '梯田': 2, '山坡地': 3, '水浇地': 4, '普通大棚': 5, '智慧大棚': 6
}).values  # 假设地块类型在 "地块类型" 列中

plot_types[34:50] = 5   #读取出现问题，这里手工操作

T_1 = 1
T_2 = 2
T_3 = 3
T_4 = 4
T_5 = 5  # 普通大棚
T_6 = 6  # 智慧大棚

rice_id = 16-1   #水稻的编号，水稻有特别的约束
cabbage_id = 35-1
white_radish_id = 36-1   #con
red_radish_id = 37-1

# 构建作物编号从 0 到 40 的作物类型映射
crop_types = {
    0: '粮食（豆类）',  # 黄豆
    1: '粮食（豆类）',  # 黑豆
    2: '粮食（豆类）',  # 红豆
    3: '粮食（豆类）',  # 绿豆
    4: '粮食（豆类）',  # 爬豆
    5: '粮食',          # 小麦
    6: '粮食',          # 玉米
    7: '粮食',          # 谷子
    8: '粮食',          # 高粱
    9: '粮食',          # 黍子
    10: '粮食',         # 荞麦
    11: '粮食',         # 南瓜
    12: '粮食',         # 红薯
    13: '粮食',         # 莜麦
    14: '粮食',         # 大麦
    15: '粮食',         # 水稻
    16: '蔬菜（豆类）',  # 豇豆
    17: '蔬菜（豆类）',  # 刀豆
    18: '蔬菜（豆类）',  # 芸豆
    19: '蔬菜',         # 土豆
    20: '蔬菜',         # 西红柿
    21: '蔬菜',         # 茄子
    22: '蔬菜',         # 菠菜
    23: '蔬菜',         # 青椒
    24: '蔬菜',         # 菜花
    25: '蔬菜',         # 包菜
    26: '蔬菜',         # 油麦菜
    27: '蔬菜',         # 小青菜
    28: '蔬菜',         # 黄瓜
    29: '蔬菜',         # 生菜
    30: '蔬菜',         # 辣椒
    31: '蔬菜',         # 空心菜
    32: '蔬菜',         # 黄心菜
    33: '蔬菜',         # 芹菜
    34: '蔬菜',         # 大白菜
    35: '蔬菜',         # 白萝卜
    36: '蔬菜',         # 红萝卜
    37: '食用菌',       # 榆黄菇
    38: '食用菌',       # 香菇
    39: '食用菌',       # 白灵菇
    40: '食用菌'        # 羊肚菌
}

crops_names = [
    '黄豆', '黑豆', '红豆', '绿豆', '爬豆', '小麦', '玉米', '谷子', '高粱', '黍子', '荞麦', '南瓜', '红薯', '莜麦', '大麦',
    '水稻', '豇豆', '刀豆', '芸豆', '土豆', '西红柿', '茄子', '菠菜', '青椒', '菜花', '包菜', '油麦菜', '小青菜',
    '黄瓜', '生菜', '辣椒', '空心菜', '黄心菜', '芹菜', '大白菜', '白萝卜', '红萝卜', '榆黄菇', '香菇', '白灵菇', '羊肚菌'
]

# 初始化crops_matrix作物替代性、互补性矩阵
matrix_size = len(crops_names)
crops_matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]

# Define some complementary and substitutable relationships (simplified examples)
complementary_pairs = [  # 互补作物
    ('黄豆', '爬豆'), ('小麦', '玉米'), ('南瓜', '红薯'), ('西红柿', '青椒'), ('大白菜', '红萝卜'), ('榆黄菇', '香菇'),
    ('黄豆','小麦'),('黄豆','玉米'),('黄豆','大麦'),('黑豆','小麦'),('黑豆','玉米'),('黑豆','大麦'),
    ('红豆','小麦'),('红豆','玉米'),('红豆','大麦'),('爬豆','小麦'),('爬豆','玉米'),('爬豆','大麦'),
    ('豇豆','小麦'),('豇豆','玉米'),('豇豆','大麦'),('绿豆','小麦'),('绿豆','玉米'),('绿豆','大麦'),
    ('刀豆','小麦'),('刀豆','玉米'),('刀豆','大麦'),('芸豆','小麦'),('芸豆','玉米'),('芸豆','大麦'),
    ('南瓜','爬豆'),('南瓜','爬豆'),('南瓜','黄豆'),('南瓜','黑豆'),('南瓜','红豆'),('红薯','玉米'),
]

substitutable_pairs = [ # 可替代作物
    ('黄豆', '黑豆'),('红豆', '绿豆'),('豇豆', '刀豆'),('芸豆', '爬豆'),('小麦', '大麦'),('小麦', '莜麦'),
    ('小麦', '荞麦'),('玉米', '高粱'),('谷子', '黍子'),('红薯', '土豆'),('白萝卜', '红萝卜'),('菠菜', '空心菜'),
    ('包菜', '菜花'),('油麦菜', '生菜'),('青椒', '辣椒'),('黄瓜', '生菜'),('榆黄菇', '香菇'),('白灵菇', '羊肚菌'),
]

# Fill the matrix with complementary relationships
for crop1, crop2 in complementary_pairs:
    i = crops_names.index(crop1)
    j = crops_names.index(crop2)
    crops_matrix[i][j] = 1
    crops_matrix[j][i] = 1

# Fill the matrix with substitutable relationships
for crop1, crop2 in substitutable_pairs:
    i = crops_names.index(crop1)
    j = crops_names.index(crop2)
    crops_matrix[i][j] = 2
    crops_matrix[j][i] = 2


# 获取作物的销售价格（p），使用区间的平均值
p_base = data_stats['销售单价/(元/斤)'].apply(
    lambda x: (float(x.split('-')[0]) + float(x.split('-')[1])) / 2 if '-' in x else float(x)
).values

# 获取作物的种植成本（c）
c_base = data_stats['种植成本/(元/亩)'].values

# 获取作物的亩产量（q），将产量从‘斤’转换为‘千克’
q_base = (data_stats['亩产量/斤'].values / 2).astype(float)  # 1斤 = 0.5千克

# 读取2023年农作物种植情况，假设其为预期销售量（D）
data_crop_situation = pd.read_excel(file_path, sheet_name='2023年的农作物种植情况')

# 计算预期销售量 D：使用种植面积/亩 乘以 对应作物的亩产量（q）
D_base = (data_crop_situation['种植面积/亩'].values * q_base[data_crop_situation['作物编号'].values - 1])

# 输出目标函数2定义和参数读取部分
print("目标函数2已经定义，销售价格、种植成本、亩产量、预期销售量已从附件中读取。")

# 初始化
particles = np.random.rand(num_particles, num_crops, num_plots, num_seasons, num_years)  # 粒子的位置
velocities = np.random.rand(num_particles, num_crops, num_plots, num_seasons, num_years)  # 粒子的速度
p_best = np.copy(particles)  # 每个粒子的最佳位置
g_best = np.copy(particles[0])  # 全局最佳位置
best_fitness_over_time = []
iterations = []  # 用于保存对应的迭代次数


# 确保x是0.1的倍数
def ensure_tenth_multiples(x):
    return np.round(x * 10) / 10


# 模拟参数的生成函数
def simulate_parameters():
    p_sim = np.copy(p_base)
    c_sim = np.copy(c_base)
    q_sim = np.copy(q_base)
    D_sim = np.copy(D_base)
    # 小麦和玉米预期销售量的增长
    for t in range(1, num_years + 1):
        for i in range(num_crops):
            if i in [5, 6]:  # 小麦和玉米
                D_sim[i] *= (1 + np.random.normal(0.075, 0.015))  # 正态分布 N(7.5%, 1.5%)
            else:
                D_sim[i] *= (1 + np.random.normal(0, 0.03))  # 正态分布 N(0%, 3%)
            # 亩产量变化
            q_sim[i] *= (1 + np.random.normal(0, 0.05))  # 正态分布 N(0, 5%)
            # 种植成本增长
            c_sim[i] *= (1 + np.random.normal(0.05, 0.01))  # 正态分布 N(5%, 1%)
            # 销售价格变化
            if i <= 14:  # 粮食类作物
                p_sim[i] = p_base[i]  # 稳定不变
            elif i <= 36:  # 蔬菜类作物
                p_sim[i] *= (1 + np.random.normal(0.05, 0.02))  # 正态分布 N(5%, 2%)
            elif i < 40:  # 食用菌类作物
                p_sim[i] *= (1 - np.random.normal(0.03, 0.01))  # 正态分布 N(-3%, 1%)
            elif i == 40:  # 羊肚菌
                p_sim[i] *= (1 - 0.05)  # 固定下降5%
    return p_sim, c_sim, q_sim, D_sim


# 目标函数1
def objective_function1(x, p_sim, c_sim, q_sim, D_sim):
    Z1 = 0
    for t in range(num_years):
        for k in range(num_seasons):
            for i in range(num_crops):
                y_ikt = np.sum(x[i, :, k, t]) * q_sim[i]
                Z1 += p_sim[i] * min(y_ikt, D_sim[i]) - c_sim[i] * np.sum(x[i, :, k, t])
    return Z1  # 由于PSO算法是求最大化问题，直接返回Z1


def apply_constraints(particles):
    global A_j, plot_types, T_5, T_6
    increase_percent = 0.05
    for n in range(num_particles):
        for j in range(num_plots):

            for i in range(num_crops):  # 遍历每一种作物
                for t in range(num_years):  # 遍历每一年
                    # 互补性约束 ： 作物 i 与 j 若为互补作物，则它们的产量同步增加一定的百分比
                    for k in range(num_crops):
                        if crops_matrix[i][k] == 1:  # 判断是否为互补作物
                            if particles[n][i, j, 0, t] > 0 and particles[n][k, j, 0, t] > 0:
                                particles[n][i, j, 0, t] *= (1 + increase_percent)  # 增加 i 作物的产量
                                particles[n][k, j, 0, t] *= (1 + increase_percent)  # 增加互补作物 k 的产量

                            if particles[n][i, j, 1, t] > 0 and particles[n][k, j, 1, t] > 0:
                                particles[n][i, j, 1, t] *= (1 + increase_percent)  # 增加 i 作物的产量
                                particles[n][k, j, 1, t] *= (1 + increase_percent)  # 增加互补作物 k 的产量

                    # 可替代性约束 ： 作物 i 与 j 若为可替代作物，则一方产量增加，另一方减少
                    for k in range(num_crops):
                        if crops_matrix[i][k] == 2:  # 判断是否为可替代作物
                            if particles[n][i, j, 0, t] > 0:
                                particles[n][k, j, 0, t] *= (1 - increase_percent)  # 减少替代作物 k 的产量
                            if particles[n][i, j, 1, t] > 0:
                                particles[n][k, j, 1, t] *= (1 - increase_percent)  # 减少替代作物 k 的产量

                    # 约束(1)：食用菌类只能在秋冬季的普通大棚里种植
                    if crop_types[i] == 5:
                        if plot_types[j] == T_5:
                            particles[n][i, j, 0, t] = 0  # 第一季不允许种植
                        else:
                            particles[n][i, j, 0, t] = 0  # 其他地块不允许种植食用菌
                            particles[n][i, j, 1, t] = 0

                    # 约束(2)：智慧大棚每年种植两季作物，第一季可种植多种蔬菜（大白菜、白萝卜和红萝卜除外），第二季只能种植食用菌  done
                    if plot_types[j] == T_6 and i in [cabbage_id, white_radish_id, red_radish_id]:
                        particles[n][i, j, 0, t] = 0
                        particles[n][i, j, 1, t] = 0

                    # 约束(3)：普通大棚每年种植两季作物，第一季可种植多种蔬菜（大白菜、白萝卜和红萝卜除外），第二季只能种植食用菌
                    if plot_types[j] == T_5:  # 普通大棚   done
                        if (crop_types[i] != 3 and crop_types[i] != 4) and i in [cabbage_id, white_radish_id,
                                                                                 red_radish_id]:
                            particles[n][i, j, 0, t] = 0  # 第一季种植
                        else:
                            particles[n][i, j, 0, t] = 0  # 第一季不允许种植大白菜、白萝卜和红萝卜
                        if crop_types[i] != 5:
                            particles[n][i, j, 1, t] = 0

                    # 约束(4)：智慧大棚每年种植两季作物，第一季可种植多种蔬菜（大白菜、白萝卜和红萝卜除外），第二季只能种植食用菌
                    if plot_types[j] in [T_1, T_2, T_3]:  # 平旱地、梯田、山坡地
                        if (crop_types[i] == 2 or crop_types[i] ==1) and i != rice_id:  # 非水稻粮食类
                            if particles[n][i, j, 0, t] > 0:
                                particles[n][i, j, 1, t] = 0  # 第二季不能种植
                                # 如果第二季种植了粮食类作物，第一季不允许种植
                            elif particles[n][i, j, 1, t] > 0:
                                particles[n][i, j, 0, t] = 0  # 第一季不能种植


                    # 约束(5)：水浇地两季蔬菜的限制，第二季只能种植大白菜、白萝卜和红萝卜中的一种  done
                    if plot_types[j] == T_4 and particles[n][i, j, 0, t] > 0 and particles[n][i, j, 1, t] > 0:
                        if i in [cabbage_id, white_radish_id, red_radish_id]:  # 第一季不允许种植这些作物,如果种植多种蔬菜的话
                            particles[n][i, j, 0, t] = 0
                        # 第二季只允许种植一种
                        if i in [cabbage_id, white_radish_id, red_radish_id]:
                            if np.sum([particles[n][k, j, 1, t] for k in [cabbage_id, white_radish_id, red_radish_id]]) > 1:
                                selected_crop = np.random.choice([cabbage_id, white_radish_id, red_radish_id])
                                for k in [cabbage_id, white_radish_id, red_radish_id]:
                                    if k != selected_crop:
                                        particles[n][k, j, 1, t] = 0

                    # 约束(6)：大白菜、白萝卜和红萝卜只能在水浇地的第二季种植  done
                    if plot_types[j] != T_4 and i in [cabbage_id, white_radish_id, red_radish_id]:
                        # 在非水浇地的所有季节都不允许种植
                        particles[n][i, j, 0, t] = 0  # 第一季不允许
                        particles[n][i, j, 1, t] = 0  # 第二季也不允许
                    if plot_types[j] == T_4 and i in [cabbage_id, white_radish_id, red_radish_id]:
                        particles[n][i, j, 0, t] = 0  # 第一季不允许种植

                    # 约束(7)：第二季只能种植大白菜、白萝卜和红萝卜中的一种     ————此约束严重拖慢代码速度
                    if plot_types[j] == T_4:  # 只针对水浇地（T_4）
                        # 从三个作物中随机选择两个作物，并将它们的种植面积设为0
                        crops = [cabbage_id, white_radish_id, red_radish_id]
                        selected_to_zero = np.random.choice(crops, 2, replace=False)  # 随机选择两个作物
                        for crop_id in selected_to_zero:
                            particles[n][crop_id, j, 1, t] = 0  # 将这两个作物的种植面积设为0

            # 定义惩罚系数（例如，如果连续种植，产量减少10%）
            penalty_factor = 0.9

            # 软约束A 重茬种植惩罚
            for i in range(num_crops):
                for k in range(num_seasons):
                    for t in range(num_years - 1):
                        if particles[n][i, j, k, t] > 0 and particles[n][i, j, k, t + 1] > 0:
                            # 如果在连续两年种植同一种作物，下一年的产量减少
                            particles[n][i, j, k, t + 1] *= penalty_factor
                        if particles[n][i, j, k, t] < 0.1 and particles[n][i, j, k, t] > 0:
                            particles[n][i, j, k, t] = 0.1
                        particles[n][i, j, k, t] = ensure_tenth_multiples(particles[n][i, j, k, t])

            # 约束B 豆类作物种植频率限制
            for t in range(num_years - 2):
                if i in (range(0, 4) or range(16,18)) and np.sum(particles[n][i, j, :, t:t+3]) < A_j[j]:
                    particles[n][i, j, :, t:t+3] = A_j[j] / 3

            # 约束C 作物种植集中度约束
            for i in range(num_crops):
                for k in range(num_seasons):
                    for t in range(num_years):
                        if particles[n][i, j, k, t] < 0.1 and particles[n][i, j, k, t] > 0:
                            particles[n][i, j, k, t] = 0.1

            # 约束D 种植面积为0.1的倍数的约束 ————模型不收敛
            for i in range(num_crops):
                for k in range(num_seasons):
                    for t in range(num_years):
                        particles[n][i, j, k, t] = ensure_tenth_multiples(particles[n][i, j, k, t])

            # 约束E 地块种植面积不超过总面积  -----这个应该放在最后
            for k in range(num_seasons):
                for t in range(num_years):
                    if np.sum(particles[n][:, j, k, t]) > A_j[j]:
                        particles[n][:, j, k, t] *= A_j[j] / np.sum(particles[n][:, j, k, t])




# 粒子群算法主函数
def pso():
    global g_best, p_best, best_fitness_over_time
    for iter in range(max_iter):
        if iter % 100 == 0 and iter > 1:
            # 绘制核心图表
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, best_fitness_over_time, label='Best Fitness over Iterations')
            plt.xlabel('Iterations')
            plt.ylabel('Best Fitness')
            plt.title('Convergence of PSO')
            plt.legend()
            plt.grid()
            plt.show()

        for n in range(num_particles):
            # 进行多次模拟并求期望
            expected_value = 0
            for _ in range(num_simulations):
                p_sim, c_sim, q_sim, D_sim = simulate_parameters()
                fitness = objective_function1(particles[n], p_sim, c_sim, q_sim, D_sim)
                expected_value += fitness
            expected_value /= num_simulations

            # 更新个体最佳位置
            if expected_value > objective_function1(p_best[n], p_sim, c_sim, q_sim, D_sim):
                p_best[n] = particles[n]
            # 更新全局最佳位置
            if expected_value > objective_function1(g_best, p_sim, c_sim, q_sim, D_sim):
                g_best = particles[n]
            # 更新粒子的速度和位置
            velocities[n] = w * velocities[n] + c1 * np.random.rand() * (
                        p_best[n] - particles[n]) + c2 * np.random.rand() * (g_best - particles[n])
            particles[n] += velocities[n]
            # 非负性约束：强制所有位置为非负
            particles[n] = np.maximum(particles[n], 0)
            # 强制x为0.1的倍数
            particles[n] = ensure_tenth_multiples(particles[n])

            # 应用约束
            apply_constraints(particles)

        # 记录每次迭代后的全局最佳适应度
        best_fitness_over_time.append(objective_function1(g_best, p_sim, c_sim, q_sim, D_sim))
        iterations.append(iter)
        print("Iteration " + str(iter) + str(-objective_function1(g_best, p_sim, c_sim, q_sim, D_sim)))

    return g_best


# 运行粒子群算法
best_solution = pso()

best_value = objective_function1(best_solution, p_base, c_base, q_base, D_base)  # 计算最优解对应的目标函数值

# 创建文件夹，命名为当前日期+时间
folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(folder_name, exist_ok=True)

# 输出x解空间矩阵到多个csv文件
for t in range(num_years):
    for k in range(num_seasons):
        df = pd.DataFrame(best_solution[:, :, k, t], columns=[f"P_{j + 1}" for j in range(num_plots)],
                          index=[f"C_{i + 1}" for i in range(num_crops)])
        file_name = f"{folder_name}/{2024 + t}_Season_{k + 1}.csv"
        df.to_csv(file_name)

# 绘制核心图表
plt.figure(figsize=(10, 6))
plt.plot(range(max_iter), best_fitness_over_time, label='Best Fitness over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Best Fitness')
plt.title('Convergence of PSO')
plt.legend()
plt.grid()
plt.savefig(f"{folder_name}/Convergence_PSO.png")
plt.show()

print("最优解对应的总收益:", best_value)
print(f"解空间矩阵已输出到文件夹: {folder_name}")
