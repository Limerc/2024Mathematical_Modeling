import matplotlib.pyplot as plt
import pandas as pd
import pulp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------读取数据部分-------------------------
# 读取农作物信息
land_data = pd.read_excel('data_source.xlsx', sheet_name='乡村的现有耕地')
production_data1 = pd.read_excel('data_source.xlsx', sheet_name='2023年的农作物种植情况')

# 地块信息
area = {}
for index, row in land_data.iterrows():
    plot_name = row['地块名称']
    plot_type = row['地块类型']  # 地块类型
    plot_size = row['地块面积/亩']  # 地块大小

    # 将数据存储到字典中
    area[plot_name] = {
        '地块类型': plot_type,
        '地块大小': plot_size
    }

# 2023年农作物信息
crop_data_temp = production_data1[['作物编号', '作物名称', '种植季次', '作物类型', '地块类型', '亩产量/斤', '销售单价中间值', '种植成本/(元/亩)']].set_index('作物编号')

# 创建目标字典
crops = {}
grouped = crop_data_temp.groupby('作物编号')
for crop_id, group in grouped:
    crop_name = group['作物名称'].iloc[0]
    crop_season = group['种植季次'].iloc[0]
    crop_type = group['作物类型'].iloc[0]
    sale_price = group['销售单价中间值'].iloc[0]

    yield_per_acre = group.set_index('地块类型')['亩产量/斤'].to_dict()
    cost_per_acre = group.set_index('地块类型')['种植成本/(元/亩)'].to_dict()

    crops[crop_id] = {
        '作物名称': crop_name,
        '种植季次': crop_season,
        '作物类型': crop_type,
        '亩产量/斤': yield_per_acre,
        '种植成本/(元/亩)': cost_per_acre,
        '销售单价中间值': sale_price
    }

# 作物销售预期量
demand = {
    1: 57000, 2: 21850, 3: 22400, 4: 33040, 5: 9875, 6: 170840,
    7: 132750, 8: 71400, 9: 30000, 10: 12500, 11: 1500, 12: 35100,
    13: 36000, 14: 14000, 15: 10000, 16: 21000, 17: 36240, 18: 26880,
    19: 6240, 20: 30000, 21: 36210, 22: 45360, 23: 900, 24: 2610,
    25: 3480, 26: 3930, 27: 4500, 28: 35480, 29: 13050, 30: 2850,
    31: 1200, 32: 3300, 33: 1620, 34: 1800, 35: 150000, 36: 100000,
    37: 36000, 38: 9000, 39: 7200, 40: 18000, 41: 4200
}

# 将销售预期量插入到作物信息中
for crop_id in crops:
    if crop_id in demand:
        crops[crop_id]['销售预期量'] = demand[crop_id]

# --------------------------整数规划部分（添加时间维度）------------------------------------
int_problem = pulp.LpProblem("Crop_Allocation_Integer_Problem", pulp.LpMaximize)

years = list(range(1, 8))  # 7年规划

# 决策变量 X[(plot, crop, year)]：表示第year年在地块plot上种植作物crop的二进制变量
X = pulp.LpVariable.dicts("X", [(plot, crop, year) for i, (plot, plot_info) in enumerate(area.items()) if i < 26 for crop in crops for year in years], cat='Binary')

# 目标函数：最大化6年内的总收益 = 总销售收益 - 总种植成本
profit_terms = []

# 创建一个列表用于存储每年的总利润
annual_profits = []

for year in years:
    annual_profit = 0  # 每年开始时利润为0
    for crop_id, crop_info in crops.items():
        total_yield_year = 0  # 每种作物每年的总产量
        total_cost_year = 0  # 每种作物每年的总种植成本
        total_sales_year = 0  # 每种作物每年的销售额
        sale_price = crop_info['销售单价中间值']

        for i,(plot, plot_info) in enumerate(area.items()):
            if i >= 26:  # 只考虑前26个地块
                continue
            if area[plot]['地块类型'] in crop_info['亩产量/斤']:
                plot_type = area[plot]['地块类型']
                plot_size = area[plot]['地块大小']
                yield_per_acre = crop_info['亩产量/斤'][plot_type]
                cost_per_acre = crop_info['种植成本/(元/亩)'][plot_type]
                expected_sales = crop_info.get('销售预期量', float('inf'))

                # 计算每个季度的总产量
                total_yield_quarter = X[(plot, crop_id, year)] * yield_per_acre * plot_size
                total_yield_year += total_yield_quarter

                # 计算种植成本
                total_cost_quarter = X[(plot, crop_id, year)] * cost_per_acre * plot_size
                total_cost_year += total_cost_quarter

        # ---------------情况1：超出的部分直接浪费----------------
        # 计算销售收入，销售量不能超过预期销售量
        expected_sales = crop_info.get('销售预期量', float('inf'))

        # 实际销售量为总产量和销售预期量的较小值
        actual_sales = pulp.LpVariable(f"actual_sales_{crop_id}_{year}", lowBound=0, upBound=expected_sales)
        int_problem += actual_sales <= total_yield_year, f"Max_Actual_Sales_{crop_id}_{year}"

        # 销售收入 = 实际销售量 * 销售单价
        total_sales_year = actual_sales * sale_price

        # 超过预期的产量不会产生利润（超出部分的浪费）
        waste_yield = total_yield_year - actual_sales

        # 总利润 = 实际销售收入 - 总种植成本
        profit_terms.append(total_sales_year - total_cost_year)

        # 累加得到每年的总利润
        annual_profit += total_sales_year - total_cost_year

        # ---------------情况2：超出的部分按照5折销售--------------
        # # 计算销售收入，销售量不能超过预期销售量
        # expected_sales = crop_info.get('销售预期量', float('inf'))
        #
        # # 实际销售量为总产量和销售预期量的较小值
        # actual_sales = pulp.LpVariable(f"actual_sales_{crop_id}_{year}", lowBound=0, upBound=expected_sales)
        # int_problem += actual_sales <= total_yield_year, f"Max_Actual_Sales_{crop_id}_{year}"
        #
        # # 超出部分的销售量：超出的部分 = 总产量 - 实际销售量
        # excess_sales = pulp.LpVariable(f"excess_sales_{crop_id}_{year}", lowBound=0)
        # int_problem += excess_sales == total_yield_year - actual_sales, f"Excess_Sales_{crop_id}_{year}"
        #
        # # 销售收入分为两部分：
        # # 1. 按正常价格出售的部分 (最多为 expected_sales)
        # normal_sales_revenue = actual_sales * sale_price
        #
        # # 2. 超出预期的部分按5折出售
        # excess_sales_revenue = excess_sales * (sale_price * 0.5)
        #
        # # 总销售收入 = 正常销售收入 + 超出部分的折扣收入
        # total_sales_year = normal_sales_revenue + excess_sales_revenue
        #
        # # 总利润 = 总销售收入 - 总种植成本
        # profit_terms.append(total_sales_year - total_cost_year)
        #
        # # 累加得到每年的总利润
        # annual_profit += total_sales_year - total_cost_year

    # 每年计算结束后，将总利润存入列表
    annual_profits.append(annual_profit)

# 将目标函数添加到问题中
int_problem += pulp.lpSum(profit_terms), "Total_Profit"

# 约束1：每年每块地块只能种植一种作物
for i, plot in enumerate(area):
    if i >= 26:  # 只考虑前26个地块
        continue
    for year in years:
        int_problem += pulp.lpSum([X[(plot, crop_id, year)] for crop_id in crops]) <= 1, f"One_Crop_Per_Plot_{plot}_Year_{year}"

# 约束2：每3年必须种一次豆类作物
for i, plot in enumerate(area):
    if i >= 26:  # 只考虑前26个地块
        continue
    for start_year in range(1, 5):  # 保证有3年窗口
        int_problem += pulp.lpSum([X[(plot, crop_id, year)] for crop_id in crops if crops[crop_id]['作物类型'] == '粮食（豆类）' for year in range(start_year, start_year + 3)]) >= 1, f"Beans_Once_Every_Three_Years_{plot}_{start_year}"

# 约束3：避免作物重茬种植（同一作物不能连续两年在同一块土地上种植）
for i, plot in enumerate(area):
    if i < 26:  # 只考虑前26个地块
        for crop_id in crops:
            for year in range(1, 7):  # 防止连续两年种植同一作物 (只到第6年，因为第7年没有 year+1)
                # 强制避免连续种植同一作物，确保同一地块在第 year 和 year+1 年不能种植相同作物
                int_problem += X[(plot, crop_id, year)] + X[(plot, crop_id, year + 1)] <= 1, f"No_Replanting_{plot}_{crop_id}_{year}"

# 求解问题
int_problem.solve(pulp.PULP_CBC_CMD(timeLimit=100))  # 设定最大求解时间为100秒
int_problem.solve()

# 遍历所有地块，打印出每年的种植作物和面积，格式与线性规划一致
for i, (plot, plot_info) in enumerate(area.items()):
    if i >= 26:  # 只考虑前26个地块
        continue
    for year in years:
        for crop_id in crops:
            if pulp.value(X[(plot, crop_id, year)]) > 0:  # 如果种植了该作物
                crop_name = crops[crop_id]['作物名称']
                plot_size = plot_info['地块大小']
                print(f"Year {year}, Plot {plot}: Plant {crop_name}, Area = {plot_size} acres")
                break  # 每块地每年只能种植一种作物，找到后跳出循环

# 打印总利润
print(f"Total Integer Profit: {pulp.value(int_problem.objective)}")

# ------------------评价1：计算总收益和年度收益平衡度--------------------------
# 计算总收益（Total Profit）
total_profit = pulp.value(int_problem.objective)
print(f"Total Profit: {total_profit}")

# 计算每年的利润（Annual Profit）
annual_profits_values = [pulp.value(annual_profit) for annual_profit in annual_profits]
print(f"Annual Profits: {annual_profits_values}")

# 计算年度收益波动（Annual Profit Standard Deviation）
profit_std = np.std(annual_profits_values)
print(f"Annual Profit Standard Deviation: {profit_std}")


# ------------------评价2：计算超出预期的浪费和折扣销售比例--------------------------
# 计算浪费量（Waste Yield）和折扣销售比例（Discount Sales Rate）
waste_yield_total = 0
discount_sales_total = 0
total_yield_total = 0

for year in years:
    for crop_id, crop_info in crops.items():
        actual_sales_var = f"actual_sales_{crop_id}_{year}"
        total_yield_year = sum(
            pulp.value(X[(plot, crop_id, year)]) * crop_info['亩产量/斤'][area[plot]['地块类型']] * area[plot]['地块大小']
            for i, plot in enumerate(area) if i < 26 and plot in X  # 确保只统计前26个地块
        )

        actual_sales_value = pulp.value(int_problem.variablesDict()[actual_sales_var])
        waste_yield = total_yield_year - actual_sales_value
        waste_yield_total += waste_yield

        # 假设有折扣销售情况
        discount_sales = max(0, waste_yield)
        discount_sales_total += discount_sales
        total_yield_total += total_yield_year

# 计算浪费比例和折扣销售比例
waste_rate = waste_yield_total / total_yield_total if total_yield_total != 0 else 0
discount_sales_rate = discount_sales_total / total_yield_total if total_yield_total != 0 else 0

print(f"Waste Yield Rate: {waste_rate}")
print(f"Discount Sales Rate: {discount_sales_rate}")

# ------------------评价3：计算土地利用率--------------------------
# 计算土地利用率（Land Utilization Rate）
land_used_total = 0
land_total = 0

for year in years:
    for i, plot in enumerate(area):
        if i >= 26:  # 只考虑前26个地块
            continue
        plot_size = area[plot]['地块大小']
        land_total += plot_size

        # 计算这一年这块地是否有作物种植
        for crop_id in crops:
            if pulp.value(X[(plot, crop_id, year)]) > 0:
                land_used_total += plot_size
                break  # 每块地每年只能种一种作物，找到之后跳出

# 土地利用率
land_utilization_rate = land_used_total / land_total
print(f"Land Utilization Rate: {land_utilization_rate}")


# ------------------评价4：计算作物种类多样性--------------------------
# 计算作物种类多样性（Crop Diversity）
diversity_score_total = 0
plot_count = 0

for year in years:
    for i, plot in enumerate(area):
        if i >= 26:  # 只考虑前26个地块
            continue
        plot_count += 1
        crop_types_planted = set()

        for crop_id in crops:
            if pulp.value(X[(plot, crop_id, year)]) > 0:
                crop_types_planted.add(crops[crop_id]['作物类型'])

        diversity_score_total += len(crop_types_planted)

# 平均作物多样性
crop_diversity_score = diversity_score_total / plot_count if plot_count != 0 else 0
print(f"Crop Diversity Score: {crop_diversity_score}")


# ------------------评价5：计算作物适宜性匹配率--------------------------
# 计算作物适宜性匹配率（Crop Suitability Match Rate）
suitable_matches = 0
total_plots = 0

for year in years:
    for i, plot in enumerate(area):
        if i >= 26:  # 只考虑前26个地块
            continue
        total_plots += 1
        for crop_id in crops:
            if pulp.value(X[(plot, crop_id, year)]) > 0:
                if area[plot]['地块类型'] in crops[crop_id]['亩产量/斤']:
                    suitable_matches += 1

# 作物适宜性匹配率
crop_suitability_match_rate = suitable_matches / total_plots if total_plots != 0 else 0
print(f"Crop Suitability Match Rate: {crop_suitability_match_rate}")


# ------------------评价6：计算豆类作物种植频率--------------------------
# 计算豆类作物种植频率（Bean Planting Frequency）
bean_planting_years = 0
total_years = 0

for i, plot in enumerate(area):
    if i >= 26:  # 只考虑前26个地块
        continue
    for year in years:
        total_years += 1
        beans_planted = False
        for crop_id in crops:
            if pulp.value(X[(plot, crop_id, year)]) > 0 and crops[crop_id]['作物类型'] == '粮食（豆类）':
                beans_planted = True
                break
        if beans_planted:
            bean_planting_years += 1

# 豆类作物种植频率
bean_planting_frequency = bean_planting_years / total_years if total_years != 0 else 0
print(f"Bean Planting Frequency: {bean_planting_frequency}")


# ------------------写入到平均文件中txt------------------------
# 文件名可以根据需要调整
output_file = 'Int_Program_evaluation_results1.txt'

# 打开文件进行写入操作
with open(output_file, 'w') as f:
    # 写入总收益
    f.write(f"Total Profit: {total_profit}\n")

    # 写入年度收益
    f.write(f"Annual Profits: {annual_profits_values}\n")

    # 写入年度收益的波动
    f.write(f"Annual Profit Standard Deviation: {profit_std}\n")

    # 写入浪费率和折扣销售比例
    f.write(f"Waste Yield Rate: {waste_rate}\n")
    f.write(f"Discount Sales Rate: {discount_sales_rate}\n")

    # 写入土地利用率
    f.write(f"Land Utilization Rate: {land_utilization_rate}\n")

    # 写入作物多样性得分
    f.write(f"Crop Diversity Score: {crop_diversity_score}\n")

    # 写入作物适宜性匹配率
    f.write(f"Crop Suitability Match Rate: {crop_suitability_match_rate}\n")

    # 写入豆类作物种植频率
    f.write(f"Bean Planting Frequency: {bean_planting_frequency}\n")

print(f"Results have been written to {output_file}")

# -----------------------------作图部分-----------------------------------

# 定义作物名称映射
crop_names = {crop_id: crop_info['作物名称'] for crop_id, crop_info in crops.items()}

# 提取决策变量的值 (每个地块、每年、每个作物的种植情况)
plot_indices = list(area.keys())[:26]  # 取前26块地
years = list(range(1, 8))  # 7年
X_vals = []
Y_vals = []
Z_vals = []
C_vals = []

for i, plot in enumerate(plot_indices):
    for crop_id in crops:
        for year in years:
            if pulp.value(X[(plot, crop_id, year)]) == 1:
                X_vals.append(i)  # 地块编号
                Y_vals.append(year)  # 年份
                Z_vals.append(crop_id)  # 作物ID
                C_vals.append(crop_id)  # 用作物ID作为颜色映射

# 创建3D绘图
fig = plt.figure(figsize=(24, 16))
ax = fig.add_subplot(111, projection='3d')

# 使用scatter绘制3D散点图
sc = ax.scatter(X_vals, Y_vals, Z_vals, c=C_vals, cmap='viridis', s=100)

# 添加颜色条并标注
colorbar = plt.colorbar(sc)
colorbar.set_label('Crop ID')

# 设置轴标签
ax.set_xlabel('Plot Index (地块编号)')
ax.set_ylabel('Year (年份)')
ax.set_zlabel('Crop ID (作物编号)')

# 设置轴的刻度标签
ax.set_xticks(range(len(plot_indices)))
ax.set_xticklabels(plot_indices, rotation=90)

ax.set_yticks(years)

ax.set_zticks(list(crops.keys()))
ax.set_zticklabels([crop_names[crop_id] for crop_id in crops.keys()])

# 使用 plt.setp() 调整 Z 轴标签的字体大小
plt.setp(ax.get_zticklabels(), fontsize=7)

# 添加标题
ax.set_title('3D Visualization of Crop Allocation Over Years and Plots')

# 调整Z轴的比例（例如将Z轴放大3倍）
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.3, 1]))

plt.show()

# 绘制每年的利润变化图
# 绘制每年的利润变化图 (使用柱状图)
plt.figure(figsize=(10, 6))

# 使用柱状图代替折线图
plt.bar(years, [pulp.value(p) for p in annual_profits], color='b', label='Annual Profit')
# 设置轴标签和标题
plt.xlabel('Year')
plt.ylabel('Profit')
plt.title('Annual Profit Over 7 Years')
# 调整 y 轴刻度大小
plt.yticks(ticks=plt.yticks()[0], fontsize=10)
# 添加网格线和图例
plt.grid(True)
plt.legend()
# 显示图像
plt.show()


# --------------------------线性规划部分（添加时间维度）-------------------------------
lp_problem = pulp.LpProblem("Crop_Allocation_Linear_Problem", pulp.LpMaximize)

years = list(range(1, 8))  # 模型处理7年规划
quarters = [1, 2]  # 每年两个季度

# 线性规划的决策变量：种植面积，表示在 plot 地块上第 year 年第 quarter 季度种植 crop 作物的面积
X_lp = pulp.LpVariable.dicts("X_lp",
                             [(plot, crop, year, quarter) for i, (plot, plot_info) in enumerate(area.items()) if i >= 26 for crop in crops for year in years for quarter in quarters], lowBound=0)

# 目标函数：最大化6年总收益 = 总销售收益 - 总种植成本
profit_terms_lp = []
# 创建一个列表用于存储每年的总利润
annual_profits = []
# 创建一个列表用于存储每年的折扣浪费量（斤）
annual_discounts = []
# 创建一个列表用于存储每年的浪费量（斤）
annual_wates = []

# 软约束的奖励/惩罚系数（可以根据需要调整）
reward_coef = 0.05  # 种植推荐作物的奖励系数
penalty_coef = -0.05  # 种植非推荐作物的惩罚系数

for year in years:
    annual_profit = 0  # 每年开始时利润为0
    annual_discount = 0  # 每年开始时折扣销售量为0
    annual_waste = 0  # 每年开始时浪费量为0
    for crop_id, crop_info in crops.items():
            total_yield_year = 0  # 每种作物每年的总产量
            total_cost_year = 0  # 每种作物每年的总种植成本
            total_sales_year = 0  # 每种作物每年的销售额
            sale_price = crop_info['销售单价中间值']

            for i, (plot, plot_info) in enumerate(area.items()):
                if i < 26:  # 跳过前26个地块
                    continue
                if area[plot]['地块类型'] in crop_info['亩产量/斤']:
                    for quarter in quarters:
                        # 确保种植季次符合条件
                        if quarter == 2 and crop_info['种植季次'] == '单季':
                            # 跳过单季水稻，水稻需种植两季
                            continue
                        elif (quarter == 1 and crop_info['种植季次'] in ['单季', '第一季']) or (
                                quarter == 2 and crop_info['种植季次'] == '第二季'):
                            # 获取地块类型、产量、种植成本和销售价格
                            yield_per_acre = crop_info['亩产量/斤'][plot_info['地块类型']]
                            cost_per_acre = crop_info['种植成本/(元/亩)'][plot_info['地块类型']]

                            # 计算每个季度的总产量
                            total_yield_quarter = X_lp[(plot, crop_id, year, quarter)] * yield_per_acre
                            total_yield_year += total_yield_quarter

                            # 计算种植成本
                            total_cost_quarter = X_lp[(plot, crop_id, year, quarter)] * cost_per_acre
                            total_cost_year += total_cost_quarter

                            # -----------非强制性约束的软约束处理----------------
                            # 针对不同地块类型设置推荐作物（蔬菜、水稻、食用菌）

                            # 如果是普通大棚
                            if plot_info['地块类型'] == '普通大棚':
                                if crop_info['作物类型'] in ['蔬菜', '食用菌']:
                                    # 奖励种植推荐作物
                                    annual_profit += reward_coef * total_yield_quarter
                                else:
                                    # 惩罚种植非推荐作物
                                    annual_profit += penalty_coef * total_yield_quarter

                            # 如果是智慧大棚
                            elif plot_info['地块类型'] == '智慧大棚':
                                if crop_info['作物类型'] == '蔬菜':
                                    # 奖励种植蔬菜
                                    annual_profit += reward_coef * total_yield_quarter
                                else:
                                    # 惩罚种植非推荐作物
                                    annual_profit += penalty_coef * total_yield_quarter

                            # 如果是水浇地
                            elif plot_info['地块类型'] == '水浇地':
                                if crop_info['作物名称'] == '水稻' or crop_info['作物类型'] == '蔬菜':
                                    # 奖励种植水稻或蔬菜
                                    annual_profit += reward_coef * total_yield_quarter
                                else:
                                    # 惩罚种植非推荐作物
                                    annual_profit += penalty_coef * total_yield_quarter

            # ---------------情况1：超出的部分直接浪费----------------
            # 计算销售收入，销售量不能超过预期销售量
            expected_sales = crop_info.get('销售预期量', float('inf'))

            # 实际销售量为总产量和销售预期量的较小值
            actual_sales = pulp.LpVariable(f"actual_sales_{crop_id}_{year}", lowBound=0, upBound=expected_sales)
            lp_problem += actual_sales <= total_yield_year, f"Max_Actual_Sales_{crop_id}_{year}"

            # 销售收入 = 实际销售量 * 销售单价
            total_sales_year = actual_sales * sale_price

            # 超过预期的产量不会产生利润（超出部分的浪费）
            waste_yield = total_yield_year - actual_sales

            # 总利润 = 实际销售收入 - 总种植成本
            profit_terms_lp.append(total_sales_year - total_cost_year)

            # 总利润 = 实际销售收入 - 总种植成本
            annual_profit += total_sales_year - total_cost_year
            annual_waste += waste_yield

            # ---------------情况2：超出的部分按照5折销售--------------
            # # 计算销售收入，销售量不能超过预期销售量
            # expected_sales = crop_info.get('销售预期量', float('inf'))
            #
            # # 实际销售量为总产量和销售预期量的较小值
            # actual_sales = pulp.LpVariable(f"actual_sales_{crop_id}_{year}", lowBound=0, upBound=expected_sales)
            # lp_problem += actual_sales <= total_yield_year, f"Max_Actual_Sales_{crop_id}_{year}"
            #
            # # 超出部分的销售量：超出的部分 = 总产量 - 实际销售量
            # excess_sales = pulp.LpVariable(f"excess_sales_{crop_id}_{year}", lowBound=0)
            # lp_problem += excess_sales == total_yield_year - actual_sales, f"Excess_Sales_{crop_id}_{year}"
            #
            # # 销售收入分为两部分：
            # # 1. 按正常价格出售的部分 (最多为 expected_sales)
            # normal_sales_revenue = actual_sales * sale_price
            #
            # # 2. 超出预期的部分按5折出售
            # excess_sales_revenue = excess_sales * (sale_price * 0.5)
            #
            # # 总销售收入 = 正常销售收入 + 超出部分的折扣收入
            # total_sales_year = normal_sales_revenue + excess_sales_revenue
            #
            # # 总利润 = 总销售收入 - 总种植成本
            # profit_terms_lp.append(total_sales_year - total_cost_year)
            #
            # # 总利润 = 实际销售收入 - 总种植成本
            # annual_profit += total_sales_year - total_cost_year
            # annual_discount += excess_sales_revenue

    # 每年计算结束后，将总利润存入列表
    annual_profits.append(annual_profit)
    annual_discounts.append(annual_discount)
    annual_wates.append(annual_waste)

lp_problem += pulp.lpSum(profit_terms_lp), "Total_Linear_Profit"

# 约束1：每年每季度中同一块土地上所有作物的种植面积之和不能超过地块的实际面积
for year in years:
    for i, plot in enumerate(area):
        if i >= 26:  # 仅针对后面的地块
            plot_size = area[plot]['地块大小']  # 地块的总面积

            # 第一季度的面积约束
            lp_problem += pulp.lpSum([X_lp[(plot, crop_id, year, 1)]
                                      for crop_id in crops
                                      if area[plot]['地块类型'] in crops[crop_id]['亩产量/斤']]) <= plot_size, \
                f"Max_Area_{plot}_Year_{year}_Q1"

            # 计算第一季度种植水稻的面积
            rice_area_q1 = pulp.lpSum([X_lp[(plot, crop_id, year, 1)]
                                       for crop_id in crops
                                       if crops[crop_id]['作物名称'] == '水稻'])

            # 第二季度的面积约束（考虑第一季度的水稻面积）
            lp_problem += pulp.lpSum([X_lp[(plot, crop_id, year, 2)]
                                      for crop_id in crops
                                      if area[plot]['地块类型'] in crops[crop_id][
                                          '亩产量/斤']]) <= plot_size - rice_area_q1, \
                f"Max_Area_{plot}_Year_{year}_Q2_With_Rice_Adjustment"

# 约束2：每种作物每季度只能种植在适合的土地类型上
for i, plot in enumerate(area):
    if i >= 26:  # 仅针对后面的地块
        for crop_id in crops:
            for year in years:
                for quarter in quarters:
                    if area[plot]['地块类型'] not in crops[crop_id]['亩产量/斤']:
                        # 强制不允许在不适合的地块种植该作物
                        lp_problem += X_lp[(plot, crop_id, year, quarter)] == 0, f"Invalid_Crop_Location_{plot}_{crop_id}_{year}_{quarter}"


# 约束3：每块土地每年至少种植一次蔬菜（豆类）
for i, plot in enumerate(area):
    if i >= 26:  # 仅针对后面的地块
        plot_size = area[plot]['地块大小']  # 地块的总面积
        for start_year in range(1, 6):  # 保证有3年窗口
            lp_problem += pulp.lpSum([X_lp[(plot, crop_id, year, quarter)] for crop_id in crops if crops[crop_id]['作物类型'] == '蔬菜（豆类）' for year in range(start_year, start_year + 3) for quarter in quarters]) >= plot_size, f"Beans_Once_Every_Three_Years_{plot}_{start_year}"

# 约束4：避免作物重茬种植（同一作物不能连续两年种植）
# 理论逻辑：如果第一年第一季度种植了A面积的X作物，第二年第一季度种植了B面积的X作物，A+B <= 该地块的面积。
for i, plot in enumerate(area):
    if i >= 26:  # 仅针对后面的地块
        plot_size = area[plot]['地块大小']  # 地块的总面积
        for crop_id in crops:
            for year in range(1, 7):  # 防止连续两年种植同一作物（1到6年，最多到6年）
                for quarter in quarters:
                    # 限制同一作物在连续两年同一季度的种植面积之和不能超过该地块的总面积
                    lp_problem += X_lp[(plot, crop_id, year, quarter)] + X_lp[(plot, crop_id, year + 1, quarter)] <= plot_size, f"No_Replanting_{plot}_{crop_id}_{year}_{quarter}"

# 约束5：如果水浇地上第一季度种植了水稻，第二季度要种植等面积的水稻，且这部分水稻不算入总利润
for year in years:
    for i, plot in enumerate(area):
        if i >= 26:  # 仅针对后面的地块
            if area[plot]['地块类型'] == '水浇地':
                # 第一季度种植了水稻
                rice_area_q1 = pulp.lpSum([X_lp[(plot, crop_id, year, 1)]
                                           for crop_id in crops
                                           if crops[crop_id]['作物名称'] == '水稻'])

                # 第二季度种植等面积的水稻
                rice_area_q2 = pulp.lpSum([X_lp[(plot, crop_id, year, 2)]
                                           for crop_id in crops
                                           if crops[crop_id]['作物名称'] == '水稻'])

                # 第二季度种植的水稻面积约束
                lp_problem += rice_area_q2 == rice_area_q1, f"Rice_Area_Q2_{plot}_Year_{year}"


# 求解问题
lp_problem.solve(pulp.PULP_CBC_CMD(timeLimit=300))   # 同样设定线性规划的最大求解时间

# 输出求解状态
print(f"Status: {pulp.LpStatus[lp_problem.status]}")

# 输出每块土地每年的种植情况
for i, plot in enumerate(area):
    if i >= 26:  # 仅针对后面的地块
        for year in years:
            for quarter in quarters:
                for crop_id in crops:
                    if pulp.value(X_lp[(plot, crop_id, year, quarter)]) > 0:
                        crop_name = crops[crop_id]['作物名称']
                        print(f"Year {year}, Quarter {quarter}, Plot {plot}: Plant {crop_name}, Area = {pulp.value(X_lp[(plot, crop_id, year, quarter)])} acres")

# 输出6年的总利润
print(f"Total Linear Profit over 6 years: {pulp.value(lp_problem.objective)}")

# -----------------------------输出模型评价部分------------------------------
# 文件名可以根据需要调整
output_file = 'linear_model_evaluation_results1.txt'

# 提取所有变量到局部字典中
variables_dict = lp_problem.variablesDict()

# 打开文件进行写入操作
with open(output_file, 'w') as f:
    # 输出模型求解状态
    f.write(f"Status: {pulp.LpStatus[lp_problem.status]}\n")

    # 输出每年的总利润
    annual_profits_values = [pulp.value(annual_profit) for annual_profit in annual_profits]
    f.write(f"Annual Profits: {annual_profits_values}\n")

    # 计算总利润
    total_profit = pulp.value(lp_problem.objective)
    f.write(f"Total Linear Profit over 6 years: {total_profit}\n")

    # 计算年度收益波动性
    profit_std = np.std(annual_profits_values)
    f.write(f"Annual Profit Standard Deviation: {profit_std}\n")

    # 计算每年的浪费量和折扣量
    # 创建列表用于存储每年的实际浪费量和折扣量
    calculated_wastes = []
    calculated_discounts = []

    # 计算每年的浪费量和折扣量
    for year in range(len(years)):
        # 使用 pulp.value() 将线性表达式转化为实际值
        total_waste_value = pulp.value(annual_wates[year])
        total_discount_value = pulp.value(annual_discounts[year])

        calculated_wastes.append(total_waste_value)
        calculated_discounts.append(total_discount_value)

    # 遍历每年每季度，计算土地利用率、浪费率和折扣率
    for year in years:
        for quarter in quarters:
            land_used_total = 0
            land_total = 0
            total_yield = 0  # 每个季度的总产量

            # 遍历每个地块，计算土地利用率
            for i, (plot, plot_info) in enumerate(area.items()):
                if i >= 26:  # 只考虑后面的地块
                    plot_size = plot_info['地块大小']
                    land_total += plot_size

                    # 计算该地块在当前季度是否有作物种植
                    for crop_id in crops:
                        if pulp.value(X_lp[(plot, crop_id, year, quarter)]) > 0:
                            land_used_total += plot_size
                            # 获取该地块的产量
                            yield_per_acre = crops[crop_id]['亩产量/斤'][plot_info['地块类型']]
                            total_yield += pulp.value(X_lp[(plot, crop_id, year, quarter)]) * yield_per_acre
                            break  # 每个地块在每季度只能种植一种作物，找到后跳出

            # 计算土地利用率
            land_utilization_rate = land_used_total / land_total if land_total > 0 else 0

            # 获取对应年份的浪费量和折扣量
            total_waste = calculated_wastes[year - 1]  # year 从 1 开始，索引从 0 开始
            total_discount = calculated_discounts[year - 1]

            # 计算浪费率
            waste_rate = total_waste / total_yield if total_yield > 0 else 0

            # 计算折扣率
            discount_rate = total_discount / total_yield if total_yield > 0 else 0

            # 输出结果到文件
            f.write(f"Year {year}, Quarter {quarter}, Land Utilization Rate: {land_utilization_rate:.4f}\n")
            f.write(f"Year {year}, Quarter {quarter}, Waste Rate: {waste_rate:.4f}%\n")
            f.write(f"Year {year}, Quarter {quarter}, Discount Rate: {discount_rate:.4f}%\n")


# ---------------------------- 画图部分 ---------------------------------------
# 1. 生成每块地在每年种植的作物图
plots = [plot for i, plot in enumerate(area) if i >= 26]  # 只选择后面的地块
crop_names = {crop_id: crops[crop_id]['作物名称'] for crop_id in crops}
n_years = len(years)
n_quarters = len(quarters)

# 创建一个颜色映射作物种植情况
fig = plt.figure(figsize=(24, 16))
ax = fig.add_subplot(111, projection='3d')

# 定义作物对应的颜色
crop_colors = {crop_name: np.random.rand(3,) for crop_name in set(crop_names.values())}

# 遍历所有地块、年份、季度
for i, plot in enumerate(plots):
    for year in years:
        for quarter in quarters:
            for crop_id in crops:
                area_planted = pulp.value(X_lp[(plot, crop_id, year, quarter)])
                if area_planted > 0:
                    crop_name = crop_names[crop_id]
                    ax.bar3d(i, year + (quarter - 1) * 0.5, 0, 0.9, 0.4, area_planted,
                             color=crop_colors[crop_name], label=crop_name if quarter == 1 else "")

# 设置坐标轴标签
ax.set_xlabel('Plots')
ax.set_ylabel('Year')
ax.set_zlabel('Area Planted (acres)')
ax.set_xticks(range(len(plots)))
ax.set_xticklabels(plots)
ax.set_title('3D Bar Plot of Crop Allocation Over Years and Quarters')
# 使用 plt.setp() 调整 X 轴标签的字体大小
plt.setp(ax.get_xticklabels(), fontsize=7)

# 创建图例
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))  # 去除重复的标签
ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()

# 2. 利润随时间变化的趋势图
plt.figure(figsize=(10, 6))
# 绘制柱状图
plt.bar(years, [pulp.value(p) for p in annual_profits], color='b', label='Annual Profit')
plt.xlabel('Year')
plt.ylabel('Profit')
plt.title('Annual Profit Over 7 Years')
# 设置 y 轴刻度缩小
plt.yticks(ticks=plt.yticks()[0], fontsize=10)  # 将 y 轴刻度的字体大小调整为 10
plt.grid(True)
plt.legend()
plt.show()

# --------------------------输出Excel部分--------------------
crops_names = ['黄豆', '黑豆', '红豆', '绿豆', '爬豆', '小麦', '玉米', '谷子', '高粱', '黍子', '荞麦', '南瓜', '红薯', '莜麦', '大麦', '水稻', '豇豆', '刀豆', '芸豆', '土豆', '西红柿', '茄子', '菠菜 ', '青椒', '菜花', '包菜', '油麦菜', '小青菜', '黄瓜', '生菜', '辣椒', '空心菜', '黄心菜', '芹菜', '大白菜', '白萝卜', '红萝卜', '榆黄菇', '香菇', '白灵菇', '羊肚菌']
area_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'F1', 'F2', 'F3', 'F4', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'F1', 'F2', 'F3', 'F4']

# 创建 ExcelWriter 对象，用于写入 Excel 文件
with pd.ExcelWriter("Crop_Allocation_Results1.xlsx", engine='openpyxl') as writer:
    # 遍历每一年
    for year in years:
        # 创建空的 DataFrame，列为作物名，行为地块名
        df_first_season = pd.DataFrame(index=area_names[:26], columns=crops_names)  # 第一季次
        df_second_season = pd.DataFrame(index=area_names[26:], columns=crops_names)  # 第二季次

        # 填充第一季次的种植情况（整数规划解）
        for i, (plot, plot_info) in enumerate(area.items()):
            if i < 26:  # 只考虑前26个地块（第一季次）
                plot_size = plot_info['地块大小']  # 获取地块大小
                for crop_id, crop_info in crops.items():
                    crop_name = crop_info['作物名称']
                    if X.get((plot, crop_id, year)) and X[(plot, crop_id, year)].varValue == 1:
                        # 填写种植面积，整数规划结果
                        df_first_season.at[plot, crop_name] = plot_size
                    else:
                        df_first_season.at[plot, crop_name] = float('nan')  # 未种植作物

        # 填充第二季次的种植情况（线性规划解）
        for i, (plot, plot_info) in enumerate(area.items()):
            if i >= 26:  # 处理后面的地块（第二季次）
                plot_size = plot_info['地块大小']  # 获取地块大小
                for crop_id, crop_info in crops.items():
                    crop_name = crop_info['作物名称']
                    if pulp.value(X_lp[(plot, crop_id, year, 2)]) > 0:
                        # 填写种植面积，线性规划结果
                        df_second_season.at[plot, crop_name] = pulp.value(X_lp[(plot, crop_id, year, 2)])
                    else:
                        df_second_season.at[plot, crop_name] = float('nan')  # 未种植作物

        # 合并第一季次和第二季次的 DataFrame
        df_combined = pd.concat([df_first_season, df_second_season], axis=0)

        # 将结果写入 Excel 文件，每个年份一个工作表
        sheet_name = f"Year_{year}"
        df_combined.to_excel(writer, sheet_name=sheet_name)

print("结果已保存到 Excel 文件中。")


