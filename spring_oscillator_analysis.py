"""
弹簧振子周期实验数据分析
研究质量 m 与弹簧振子周期 T 的关系
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据准备与基础计算 ====================
print("=" * 60)
print("1. 数据准备与基础计算")
print("=" * 60)

# 创建质量数组 m (kg) - 模拟实验数据
m = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])

# 真实的弹簧劲度系数（用于生成模拟数据）
k_true = 25.0  # N/m

# 根据公式 T = 2π√(m/k) 生成理论周期，并添加一些随机噪声模拟真实测量
np.random.seed(42)  # 设置随机种子以保证可重复性
T_theory = 2 * np.pi * np.sqrt(m / k_true)
noise = np.random.normal(0, 0.02, len(m))  # 添加正态分布噪声
T = T_theory + noise

# 计算周期平方 T²
T_squared = T ** 2

# 输出前5组数据
print("\n前5组实验数据：")
print("-" * 40)
print(f"{'序号':<6}{'质量 m (kg)':<15}{'周期 T (s)':<15}{'周期平方 T² (s²)':<15}")
print("-" * 40)
for i in range(5):
    print(f"{i+1:<6}{m[i]:<15.3f}{T[i]:<15.3f}{T_squared[i]:<15.4f}")
print("-" * 40)

# ==================== 2. 线性回归分析 ====================
print("\n" + "=" * 60)
print("2. 线性回归分析")
print("=" * 60)

# 使用 scipy.stats.linregress 进行线性回归
# T² = (4π²/k) * m，所以 T² 与 m 呈线性关系
slope, intercept, r_value, p_value, std_err = stats.linregress(m, T_squared)

# 计算 R²
r_squared = r_value ** 2

print(f"\n线性回归结果 (T² vs m)：")
print(f"  斜率 (slope): {slope:.6f} s²/kg")
print(f"  截距 (intercept): {intercept:.6f} s²")
print(f"  相关系数 R²: {r_squared:.6f}")
print(f"  p值: {p_value:.6e}")
print(f"  标准误差: {std_err:.6f}")

# 根据斜率计算弹簧劲度系数 k
# 斜率 = 4π²/k  =>  k = 4π²/斜率
k_linear = (4 * np.pi ** 2) / slope
print(f"\n根据线性回归计算的弹簧劲度系数：")
print(f"  k = 4π² / 斜率 = {k_linear:.2f} N/m")

# ==================== 3. 非线性曲线拟合 ====================
print("\n" + "=" * 60)
print("3. 非线性曲线拟合")
print("=" * 60)

# 定义模型函数 T = 2π√(m/k)
def model(m, k):
    return 2 * np.pi * np.sqrt(m / k)

# 使用 scipy.optimize.curve_fit 进行非线性拟合
popt, pcov = curve_fit(model, m, T, p0=[20])  # p0 是初始猜测值
k_nonlinear = popt[0]
k_nonlinear_err = np.sqrt(pcov[0, 0])  # 拟合误差

print(f"\n非线性拟合结果 (T vs m)：")
print(f"  拟合得到的劲度系数 k: {k_nonlinear:.2f} ± {k_nonlinear_err:.2f} N/m")

# 对比分析
print(f"\n两种方法结果对比：")
print(f"  线性回归法: k = {k_linear:.2f} N/m")
print(f"  非线性拟合法: k = {k_nonlinear:.2f} N/m")
print(f"  真实值 (用于生成数据): k = {k_true:.2f} N/m")
print(f"  线性回归误差: {abs(k_linear - k_true):.2f} N/m ({abs(k_linear - k_true)/k_true*100:.2f}%)")
print(f"  非线性拟合误差: {abs(k_nonlinear - k_true):.2f} N/m ({abs(k_nonlinear - k_true)/k_true*100:.2f}%)")

# ==================== 4. 数据可视化 ====================
print("\n" + "=" * 60)
print("4. 数据可视化")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# (1) 左图：线性拟合（T² vs m）
ax1.scatter(m, T_squared, color='blue', s=50, label='实验数据', zorder=5)

# 绘制拟合直线
m_fit = np.linspace(m.min(), m.max(), 100)
T_squared_fit = slope * m_fit + intercept
ax1.plot(m_fit, T_squared_fit, 'r-', linewidth=2, label='线性拟合', zorder=3)

# 添加回归方程和 R² 标注
equation_text = f'$T^2$ = {slope:.4f}m + {intercept:.4f}\n$R^2$ = {r_squared:.4f}'
ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.set_xlabel('质量 m (kg)', fontsize=12)
ax1.set_ylabel('周期平方 T² (s²)', fontsize=12)
ax1.set_title('线性拟合：T² 与 m 的关系', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# (2) 右图：非线性拟合（T vs m）
ax2.scatter(m, T, color='blue', s=50, label='实验数据', zorder=5)

# 绘制拟合曲线
m_curve = np.linspace(m.min(), m.max(), 100)
T_curve = model(m_curve, k_nonlinear)
ax2.plot(m_curve, T_curve, 'g-', linewidth=2, label='非线性拟合', zorder=3)

# 标注拟合得到的 k 值
k_text = f'拟合结果: k = {k_nonlinear:.2f} N/m'
ax2.text(0.05, 0.95, k_text, transform=ax2.transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax2.set_xlabel('质量 m (kg)', fontsize=12)
ax2.set_ylabel('周期 T (s)', fontsize=12)
ax2.set_title('非线性拟合：T 与 m 的关系', fontsize=14)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spring_fit.png', dpi=150, bbox_inches='tight')
print("\n图像已保存为 'spring_fit.png'")

# ==================== 可选扩展：残差分析 ====================
print("\n" + "=" * 60)
print("可选扩展：残差分析")
print("=" * 60)

# 计算残差
T_squared_pred = slope * m + intercept
residuals_linear = T_squared - T_squared_pred

T_pred_nonlinear = model(m, k_nonlinear)
residuals_nonlinear = T - T_pred_nonlinear

print(f"\n线性拟合残差统计：")
print(f"  残差均值: {np.mean(residuals_linear):.6f}")
print(f"  残差标准差: {np.std(residuals_linear):.6f}")

print(f"\n非线性拟合残差统计：")
print(f"  残差均值: {np.mean(residuals_nonlinear):.6f}")
print(f"  残差标准差: {np.std(residuals_nonlinear):.6f}")

# 创建残差图
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 4))

# 线性拟合残差图
ax3.scatter(m, residuals_linear, color='blue', s=50)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax3.set_xlabel('质量 m (kg)', fontsize=12)
ax3.set_ylabel('残差 (s²)', fontsize=12)
ax3.set_title('线性拟合残差图', fontsize=14)
ax3.grid(True, alpha=0.3)

# 非线性拟合残差图
ax4.scatter(m, residuals_nonlinear, color='blue', s=50)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax4.set_xlabel('质量 m (kg)', fontsize=12)
ax4.set_ylabel('残差 (s)', fontsize=12)
ax4.set_title('非线性拟合残差图', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spring_residuals.png', dpi=150, bbox_inches='tight')
print("\n残差图已保存为 'spring_residuals.png'")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
