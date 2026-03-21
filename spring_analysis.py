import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 1. 数据准备与基础计算
# 实验数据（示例数据，可根据实际实验数据修改）
m = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])  # 质量（kg）
T = np.array([0.45, 0.63, 0.77, 0.89, 1.00, 1.09, 1.18, 1.26, 1.33, 1.40])   # 周期（秒）

# 计算周期平方
T_squared = T ** 2

# 输出前5组数据
print("=" * 50)
print("1. 数据准备与基础计算")
print("=" * 50)
print(f"{'质量m (kg)':<12} {'周期T (s)':<12} {'周期平方T² (s²)':<16}")
print("-" * 40)
for i in range(min(5, len(m))):
    print(f"{m[i]:<12.4f} {T[i]:<12.4f} {T_squared[i]:<16.4f}")

# 2. 线性回归分析
print("\n" + "=" * 50)
print("2. 线性回归分析")
print("=" * 50)

# 线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(m, T_squared)
r_squared = r_value ** 2

# 计算弹簧劲度系数
k_linear = (4 * np.pi ** 2) / slope

print(f"斜率: {slope:.4f} s²/kg")
print(f"截距: {intercept:.4f} s²")
print(f"相关系数R²: {r_squared:.4f}")
print(f"p值: {p_value:.4e}")
print(f"弹簧劲度系数k (线性回归): {k_linear:.2f} N/m")

# 3. 非线性曲线拟合
print("\n" + "=" * 50)
print("3. 非线性曲线拟合")
print("=" * 50)

# 定义模型函数
def model(m, k):
    return 2 * np.pi * np.sqrt(m / k)

# 初始猜测值
initial_guess = [10.0]

# 曲线拟合
popt, pcov = curve_fit(model, m, T, p0=initial_guess)
k_nonlinear = popt[0]

print(f"弹簧劲度系数k (非线性拟合): {k_nonlinear:.2f} N/m")

# 对比分析
print("\n对比分析:")
print(f"线性回归得到的k: {k_linear:.2f} N/m")
print(f"非线性拟合得到的k: {k_nonlinear:.2f} N/m")
print(f"两者差值: {abs(k_linear - k_nonlinear):.4f} N/m")

# 4. 数据可视化
print("\n" + "=" * 50)
print("4. 数据可视化")
print("=" * 50)

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：线性拟合（T² vs m）
ax1.scatter(m, T_squared, color='blue', label='实验数据')
m_fit = np.linspace(min(m), max(m), 100)
T_squared_fit = slope * m_fit + intercept
ax1.plot(m_fit, T_squared_fit, 'r-', label='拟合直线')

ax1.set_title('线性拟合：T² 与 m 的关系', fontsize=14)
ax1.set_xlabel('质量 m (kg)', fontsize=12)
ax1.set_ylabel('周期平方 T² (s²)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 添加回归方程和R²标注
equation_text = f'T² = {slope:.4f}m + {intercept:.4f}\nR² = {r_squared:.4f}'
ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

# 右图：非线性拟合（T vs m）
ax2.scatter(m, T, color='blue', label='实验数据')
T_fit = model(m_fit, k_nonlinear)
ax2.plot(m_fit, T_fit, 'g-', label='拟合曲线')

ax2.set_title('非线性拟合：T 与 m 的关系', fontsize=14)
ax2.set_xlabel('质量 m (kg)', fontsize=12)
ax2.set_ylabel('周期 T (s)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

# 添加k值标注
k_text = f'k = {k_nonlinear:.2f} N/m'
ax2.text(0.05, 0.95, k_text, transform=ax2.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('spring_fit.png', dpi=300, bbox_inches='tight')
print("图像已保存为 spring_fit.png")

# 显示图像
plt.show()

# 可选扩展：残差分析
print("\n" + "=" * 50)
print("可选扩展：残差分析")
print("=" * 50)

# 线性回归残差
T_squared_pred = slope * m + intercept
residuals_linear = T_squared - T_squared_pred

# 非线性拟合残差
T_pred = model(m, k_nonlinear)
residuals_nonlinear = T - T_pred

print(f"线性回归残差平方和: {np.sum(residuals_linear ** 2):.6f}")
print(f"非线性拟合残差平方和: {np.sum(residuals_nonlinear ** 2):.6f}")

# 绘制残差图
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

ax3.scatter(m, residuals_linear, color='red')
ax3.axhline(y=0, color='black', linestyle='--')
ax3.set_title('线性拟合残差图', fontsize=14)
ax3.set_xlabel('质量 m (kg)', fontsize=12)
ax3.set_ylabel('残差 (s²)', fontsize=12)
ax3.grid(True, alpha=0.3)

ax4.scatter(m, residuals_nonlinear, color='red')
ax4.axhline(y=0, color='black', linestyle='--')
ax4.set_title('非线性拟合残差图', fontsize=14)
ax4.set_xlabel('质量 m (kg)', fontsize=12)
ax4.set_ylabel('残差 (s)', fontsize=12)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
print("残差图已保存为 residual_plots.png")
