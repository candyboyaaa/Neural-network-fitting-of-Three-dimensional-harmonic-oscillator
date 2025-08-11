import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

# 物理常数
hbar = 197.3269804  # MeV·fm
mu = 939  # MeV/c²
omega = 10 / hbar  # MeV

# 计算alpha参数
alpha = mu * omega / hbar

# 输出物理参数信息
print(f"物理参数:")
print(f"  hbar = {hbar} MeV·fm")
print(f"  mu = {mu} MeV/c²")
print(f"  omega = {omega:.6f} MeV")
print(f"  alpha = {alpha:.6f} fm⁻²")
print()

# 三维谐振子基态和激发态解析解
def psi_3d_analytic(n_x, n_y, n_z, x, y, z, alpha=alpha):
    # 1D谐振子本征函数
    from scipy.special import hermite, factorial
    def psi_1d(n, x, alpha):
        Hn = hermite(n)
        norm = 1.0 / np.sqrt(2.0**n * factorial(n)) * (alpha/np.pi)**0.25
        return norm * np.exp(-0.5*alpha*x**2) * Hn(np.sqrt(alpha)*x)
    return psi_1d(n_x, x, alpha) * psi_1d(n_y, y, alpha) * psi_1d(n_z, z, alpha)

# 网格参数，与plot.py一致
x_min, x_max = -5, 5
y_min, y_max = -5, 5
z_min, z_max = -5, 5
offset = 0.1
x = np.linspace(x_min, x_max, 80)
y = np.linspace(y_min, y_max, 80)
z = np.linspace(z_min, z_max, 80)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 选择态：基态(0,0,0)，第一激发态(1,0,0)
psi_gs = psi_3d_analytic(0, 0, 0, X, Y, Z)
psi_ex = psi_3d_analytic(1, 0, 0, X, Y, Z)

# 物理归一化
# 计算体元体积
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]
dV = dx * dy * dz
# 计算归一化因子
norm_gs = np.sqrt(np.sum(np.abs(psi_gs)**2) * dV)
norm_ex = np.sqrt(np.sum(np.abs(psi_ex)**2) * dV)
# 归一化波函数
psi_gs /= norm_gs
psi_ex /= norm_ex

# 归一化检验
print("基态归一化检验:", np.sum(np.abs(psi_gs)**2) * dV)
print("第一激发态归一化检验:", np.sum(np.abs(psi_ex)**2) * dV)
# 概率密度
prob_gs = np.abs(psi_gs)**2
prob_ex = np.abs(psi_ex)**2

# 体渲染参数与plot.py一致
color = [
    [0, "rgba(255, 255, 255, 0)"],
    [0.1, "rgb(0, 0, 128)"],
    [0.3, "rgb(0, 255, 255)"],
    [0.6, "rgb(255, 255, 0)"],
    [1, "rgb(128, 0, 0)"]
]
opc = 0.3
surface_count = 15
vmin = 0
vmax = 0.021

# ------------------- 三维体密度图 -------------------
fig_gs_vol = go.Figure(go.Volume(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=prob_gs.flatten(),
    opacity=opc,  # 体渲染透明度
    surface_count=surface_count,
    colorscale=color,
    isomin=vmin,
    isomax=vmax,
    showscale=True,
    colorbar=dict(title='|ψ₀₀₀|²'),
    name='ground state |ψ₀₀₀|²',
))

# 添加主切面，与plot.py一致
from scipy.interpolate import griddata
opc2 = 0.4
vmax2 = vmax * 1.1
# y=0切面
mask_y = np.isclose(Y.flatten(), 0, atol=0.1)
x_slice_y = X.flatten()[mask_y]
z_slice_y = Z.flatten()[mask_y]
density_slice_y = prob_gs.flatten()[mask_y]
x_unique_y = np.linspace(x_min, x_max, 100)
z_unique_y = np.linspace(z_min, z_max, 100)
x_grid_y, z_grid_y = np.meshgrid(x_unique_y, z_unique_y)
points_y = np.vstack((x_slice_y, z_slice_y)).T
density_grid_y = griddata(points_y, density_slice_y, (x_grid_y, z_grid_y), method='linear')
density_grid_y = np.nan_to_num(density_grid_y, nan=0.0)
fig_gs_vol.add_trace(go.Surface(
    x=x_grid_y,
    y=np.full_like(x_grid_y, y_min-offset),
    z=z_grid_y,
    cmin=vmin,
    cmax=vmax2,
    surfacecolor=density_grid_y,
    colorscale=color,
    opacity=opc2,
    showscale=False
))
# z=0切面
mask_z = np.isclose(Z.flatten(), 0, atol=0.1)
x_slice_z = X.flatten()[mask_z]
y_slice_z = Y.flatten()[mask_z]
density_slice_z = prob_gs.flatten()[mask_z]
x_unique_z = np.linspace(x_min, x_max, 100)
y_unique_z = np.linspace(y_min, y_max, 100)
x_grid_z, y_grid_z = np.meshgrid(x_unique_z, y_unique_z)
points_z = np.vstack((x_slice_z, y_slice_z)).T
density_grid_z = griddata(points_z, density_slice_z, (x_grid_z, y_grid_z), method='linear')
density_grid_z = np.nan_to_num(density_grid_z, nan=0.0)
fig_gs_vol.add_trace(go.Surface(
    x=x_grid_z,
    y=y_grid_z,
    z=np.full_like(x_grid_z, z_min-offset),
    cmin=vmin,
    cmax=vmax2,
    surfacecolor=density_grid_z,
    colorscale=color,
    opacity=opc2,
    showscale=False
))
# x=0切面
mask_x = np.isclose(X.flatten(), 0, atol=0.1)
y_slice_x = Y.flatten()[mask_x]
z_slice_x = Z.flatten()[mask_x]
density_slice_x = prob_gs.flatten()[mask_x]
y_unique_x = np.linspace(y_min, y_max, 100)
z_unique_x = np.linspace(z_min, z_max, 100)
y_grid_x, z_grid_x = np.meshgrid(y_unique_x, z_unique_x)
points_x = np.vstack((y_slice_x, z_slice_x)).T
density_grid_x = griddata(points_x, density_slice_x, (y_grid_x, z_grid_x), method='linear')
density_grid_x = np.nan_to_num(density_grid_x, nan=0.0)
fig_gs_vol.add_trace(go.Surface(
    x=np.full_like(y_grid_x, x_min-offset),
    y=y_grid_x,
    z=z_grid_x,
    cmin=vmin,
    cmax=vmax2,
    surfacecolor=density_grid_x,
    colorscale=color,
    opacity=opc2,
    showscale=False
))
# 第一激发态同理
fig_ex_vol = go.Figure(go.Volume(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=prob_ex.flatten(),
    opacity=opc,
    surface_count=surface_count,
    colorscale=color,
    isomin=vmin,
    isomax=vmax,
    showscale=True,
    colorbar=dict(title='|ψ₁₀₀|²'),
    name='first excited state |ψ₁₀₀|²',
))
# y=0切面
mask_y = np.isclose(Y.flatten(), 0, atol=0.1)
x_slice_y = X.flatten()[mask_y]
z_slice_y = Z.flatten()[mask_y]
density_slice_y = prob_ex.flatten()[mask_y]
x_unique_y = np.linspace(x_min, x_max, 100)
z_unique_y = np.linspace(z_min, z_max, 100)
x_grid_y, z_grid_y = np.meshgrid(x_unique_y, z_unique_y)
points_y = np.vstack((x_slice_y, z_slice_y)).T
density_grid_y = griddata(points_y, density_slice_y, (x_grid_y, z_grid_y), method='linear')
density_grid_y = np.nan_to_num(density_grid_y, nan=0.0)
fig_ex_vol.add_trace(go.Surface(
    x=x_grid_y,
    y=np.full_like(x_grid_y, y_min-offset),
    z=z_grid_y,
    cmin=vmin,
    cmax=vmax2,
    surfacecolor=density_grid_y,
    colorscale=color,
    opacity=opc2,
    showscale=False
))
# z=0切面
mask_z = np.isclose(Z.flatten(), 0, atol=0.1)
x_slice_z = X.flatten()[mask_z]
y_slice_z = Y.flatten()[mask_z]
density_slice_z = prob_ex.flatten()[mask_z]
x_unique_z = np.linspace(x_min, x_max, 100)
y_unique_z = np.linspace(y_min, y_max, 100)
x_grid_z, y_grid_z = np.meshgrid(x_unique_z, y_unique_z)
points_z = np.vstack((x_slice_z, y_slice_z)).T
density_grid_z = griddata(points_z, density_slice_z, (x_grid_z, y_grid_z), method='linear')
density_grid_z = np.nan_to_num(density_grid_z, nan=0.0)
fig_ex_vol.add_trace(go.Surface(
    x=x_grid_z,
    y=y_grid_z,
    z=np.full_like(x_grid_z, z_min-offset),
    cmin=vmin,
    cmax=vmax2,
    surfacecolor=density_grid_z,
    colorscale=color,
    opacity=opc2,
    showscale=False
))
# x=0切面
mask_x = np.isclose(X.flatten(), 0, atol=0.1)
y_slice_x = Y.flatten()[mask_x]
z_slice_x = Z.flatten()[mask_x]
density_slice_x = prob_ex.flatten()[mask_x]
y_unique_x = np.linspace(y_min, y_max, 100)
z_unique_x = np.linspace(z_min, z_max, 100)
y_grid_x, z_grid_x = np.meshgrid(y_unique_x, z_unique_x)
points_x = np.vstack((y_slice_x, z_slice_x)).T
density_grid_x = griddata(points_x, density_slice_x, (y_grid_x, z_grid_x), method='linear')
density_grid_x = np.nan_to_num(density_grid_x, nan=0.0)
fig_ex_vol.add_trace(go.Surface(
    x=np.full_like(y_grid_x, x_min-offset),
    y=y_grid_x,
    z=z_grid_x,
    cmin=vmin,
    cmax=vmax2,
    surfacecolor=density_grid_x,
    colorscale=color,
    opacity=opc2,
    showscale=False
))
fig_gs_vol.update_layout(
    title="Analytical solution of the ground state of a three-dimensional harmonic oscillator",
    scene=dict(
        xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='cube',
        xaxis=dict(range=[x_min-offset, x_max]),
        yaxis=dict(range=[y_min-offset, y_max]),
        zaxis=dict(range=[z_min-offset, z_max])
    )
)

fig_ex_vol.update_layout(
    title="Analytical solution of the first excited state of a three-dimensional harmonic oscillator",
    scene=dict(
        xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='cube',
        xaxis=dict(range=[x_min-offset, x_max]),
        yaxis=dict(range=[y_min-offset, y_max]),
        zaxis=dict(range=[z_min-offset, z_max])
    )
)

# 输出为html
pio.write_html(fig_gs_vol, file="Analytical solution of the ground state of a three-dimensional harmonic oscillator.html", auto_open=False)
pio.write_html(fig_ex_vol, file="Analytical solution of the first excited state of a three-dimensional harmonic oscillator.html", auto_open=False)
print("基态三维体密度图已保存为 Analytical solution of the ground state of a three-dimensional harmonic oscillator.html")
print("第一激发态三维体密度图已保存为 Analytical solution of the first excited state of a three-dimensional harmonic oscillator.html")
