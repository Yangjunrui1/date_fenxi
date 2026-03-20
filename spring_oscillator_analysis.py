import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("弹簧振子周期实验数据分析")
print("=" * 60)

print("\n" + "=" * 60)
print("任务1：数据准备与基础计算")
print("=" * 60)

m = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
T = np.array([0.31, 0.44, 0.54, 0.63, 0.70, 0.77, 0.83, 0.89, 0.94, 1.00])

T_squared = T ** 2

print("\n前5组数据：")
print("-" * 45)
print(f"{'质量 m (kg)':<15}{'周期 T (s)':<15}{'周期平方 T² (s²)':<15}")
print("-" * 45)
for i in range(5):
    print(f"{m[i]:<15.2f}{T[i]:<15.2f}{T_squared[i]:<15.4f}")

print("\n" + "=" * 60)
print("任务2：线性回归分析")
print("=" * 60)

slope, intercept, r_value, p_value, std_err = linregress(m, T_squared)

r_squared = r_value ** 2

print(f"\n线性回归结果 (T² vs m)：")
print(f"  斜率 (slope): {slope:.6f} s²/kg")
print(f"  截距 (intercept): {intercept:.6f} s²")
print(f"  相关系数 R²: {r_squared:.6f}")
print(f"  p值: {p_value:.6e}")

k_linear = (4 * np.pi ** 2) / slope
print(f"\n根据斜率计算的弹簧劲度系数：")
print(f"  k = 4π² / 斜率 = {k_linear:.2f} N/m")

print("\n" + "=" * 60)
print("任务3：非线性曲线拟合")
print("=" * 60)

def model(m, k):
    return 2 * np.pi * np.sqrt(m / k)

popt, pcov = curve_fit(model, m, T, p0=[20])
k_nonlinear = popt[0]
k_nonlinear_err = np.sqrt(pcov[0, 0])

print(f"\n非线性曲线拟合结果 (T vs m)：")
print(f"  劲度系数 k = {k_nonlinear:.2f} N/m")
print(f"  拟合误差 = ±{k_nonlinear_err:.2f} N/m")

print(f"\n两种方法结果对比：")
print(f"  线性回归法: k = {k_linear:.2f} N/m")
print(f"  非线性拟合法: k = {k_nonlinear:.2f} N/m")
print(f"  相对误差: {abs(k_linear - k_nonlinear) / k_linear * 100:.2f}%")

print("\n" + "=" * 60)
print("任务4：数据可视化")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(m, T_squared, color='blue', s=60, label='实验数据', zorder=3)
m_fit = np.linspace(0, max(m) * 1.1, 100)
T_squared_fit = slope * m_fit + intercept
ax1.plot(m_fit, T_squared_fit, 'r-', linewidth=2, label='拟合直线', zorder=2)

equation_text = f'T^2 = {slope:.4f}m + {intercept:.4f}'
r2_text = f'R^2 = {r_squared:.4f}'
ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.text(0.05, 0.85, r2_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.set_xlabel('质量 m (kg)', fontsize=12)
ax1.set_ylabel('周期平方 T^2 (s^2)', fontsize=12)
ax1.set_title('线性拟合：T^2 与 m 的关系', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max(m) * 1.1)
ax1.set_ylim(0, max(T_squared) * 1.1)

ax2.scatter(m, T, color='blue', s=60, label='实验数据', zorder=3)
T_fit = model(m_fit, k_nonlinear)
ax2.plot(m_fit, T_fit, 'g-', linewidth=2, label='拟合曲线', zorder=2)

k_text = f'k = {k_nonlinear:.2f} N/m'
ax2.text(0.05, 0.95, k_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax2.set_xlabel('质量 m (kg)', fontsize=12)
ax2.set_ylabel('周期 T (s)', fontsize=12)
ax2.set_title('非线性拟合：T 与 m 的关系', fontsize=14)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max(m) * 1.1)
ax2.set_ylim(0, max(T) * 1.1)

plt.tight_layout()
plt.savefig('spring_fit.png', dpi=150, bbox_inches='tight')
print("\n图像已保存为 spring_fit.png")

print("\n" + "=" * 60)
print("可选扩展：误差分析与残差图")
print("=" * 60)

T_squared_pred = slope * m + intercept
residuals_linear = T_squared - T_squared_pred
T_pred = model(m, k_nonlinear)
residuals_nonlinear = T - T_pred

print(f"\n线性拟合残差统计：")
print(f"  残差均值: {np.mean(residuals_linear):.6f} s²")
print(f"  残差标准差: {np.std(residuals_linear):.6f} s²")
print(f"  残差范围: [{np.min(residuals_linear):.4f}, {np.max(residuals_linear):.4f}] s²")

print(f"\n非线性拟合残差统计：")
print(f"  残差均值: {np.mean(residuals_nonlinear):.6f} s")
print(f"  残差标准差: {np.std(residuals_nonlinear):.6f} s")
print(f"  残差范围: [{np.min(residuals_nonlinear):.4f}, {np.max(residuals_nonlinear):.4f}] s")

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

ax3.scatter(m, residuals_linear, color='blue', s=60, zorder=3)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
ax3.set_xlabel('质量 m (kg)', fontsize=12)
ax3.set_ylabel('残差 (s^2)', fontsize=12)
ax3.set_title('线性拟合残差图', fontsize=14)
ax3.grid(True, alpha=0.3)

ax4.scatter(m, residuals_nonlinear, color='green', s=60, zorder=3)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
ax4.set_xlabel('质量 m (kg)', fontsize=12)
ax4.set_ylabel('残差 (s)', fontsize=12)
ax4.set_title('非线性拟合残差图', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals.png', dpi=150, bbox_inches='tight')
print("\n残差图已保存为 residuals.png")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
print(f"\n最终结果汇总：")
print(f"  弹簧劲度系数 (线性回归): k = {k_linear:.2f} N/m")
print(f"  弹簧劲度系数 (非线性拟合): k = {k_nonlinear:.2f} N/m")
print(f"  两种方法相对误差: {abs(k_linear - k_nonlinear) / k_linear * 100:.2f}%")
