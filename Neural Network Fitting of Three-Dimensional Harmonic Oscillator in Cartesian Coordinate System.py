import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from functools import partial
import plotly.graph_objs as go
import plotly.subplots as sp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 常数定义
hbar = 197.3269804
mu = 939.0
omega = 10.0 / hbar

# 球谐函数计算
def spherical_harmonics(xyz):
    """
    计算球谐函数Y₁₊₁和Y₁₋₁
    xyz: (..., 3) 笛卡尔坐标
    返回: (..., 2) 包含Y₁₊₁和Y₁₋₁的数组
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    
    # 避免除零
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    
    # 计算球坐标
    theta = jnp.arccos(z / r_safe)
    phi = jnp.arctan2(y, x)
    
    # 球谐函数Y₁₊₁ = -sqrt(3/(8π)) * sin(θ) * exp(iφ)
    Y_1_plus_1 = -jnp.sqrt(3/(8*jnp.pi)) * jnp.sin(theta) * jnp.exp(1j * phi)
    
    # 球谐函数Y₁₋₁ = sqrt(3/(8π)) * sin(θ) * exp(-iφ)
    Y_1_minus_1 = jnp.sqrt(3/(8*jnp.pi)) * jnp.sin(theta) * jnp.exp(-1j * phi)
    
    return jnp.stack([Y_1_plus_1, Y_1_minus_1], axis=-1)

# swish激活函数
def swish(x):
    return x * nn.sigmoid(x)

# 神经网络定义
class PsiNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = swish(x)
        x = nn.Dense(128)(x)
        x = swish(x)
        x = nn.Dense(64)(x)
        x = swish(x)
        # 输出为复数：实部和虚部分开输出
        x = nn.Dense(2)(x)
        return x[..., 0] + 1j * x[..., 1]

# 第二个神经网络定义（与第一个完全一致）
class PsiNet2(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = swish(x)
        x = nn.Dense(128)(x)
        x = swish(x)
        x = nn.Dense(64)(x)
        x = swish(x)
        # 输出为复数：实部和虚部分开输出
        x = nn.Dense(2)(x)
        return x[..., 0] + 1j * x[..., 1]

# 哈密顿量作用
def hamiltonian(psi_fn, params, xyz):
    def psi_real(x):
        return jnp.real(psi_fn(params, x))
    def psi_imag(x):
        return jnp.imag(psi_fn(params, x))
    # 二阶导数
    grad2_real = jax.jacfwd(jax.jacfwd(psi_real))(xyz)
    grad2_imag = jax.jacfwd(jax.jacfwd(psi_imag))(xyz)
    laplacian = jnp.trace(grad2_real, axis1=-2, axis2=-1) + 1j * jnp.trace(grad2_imag, axis1=-2, axis2=-1)
    potential = 0.5 * mu * omega**2 * jnp.sum(xyz**2, axis=-1)
    return (-hbar**2 / (2 * mu)) * laplacian + potential * psi_fn(params, xyz)

# 蒙特卡洛采样
def sample_xyz(key, n_points):
    return jax.random.uniform(key, (n_points, 3), minval=-10.0, maxval=10.0)

# 损失函数
def loss_fn(params, apply_fn, xyz):
    psi = apply_fn(params, xyz)
    H_psi = jax.vmap(lambda x: hamiltonian(apply_fn, params, x))(xyz)
    numerator = jnp.vdot(psi, H_psi)
    denominator = jnp.vdot(psi, psi)
    return jnp.real(numerator / denominator)

# 第二个神经网络的损失函数（包含正交惩罚项）
def loss_fn_orthogonal(params2, apply_fn2, params1, apply_fn1, xyz):
    # 基础能量损失
    psi2 = apply_fn2(params2, xyz)
    H_psi2 = jax.vmap(lambda x: hamiltonian(apply_fn2, params2, x))(xyz)
    numerator = jnp.vdot(psi2, H_psi2)
    denominator = jnp.vdot(psi2, psi2)
    energy_loss = jnp.real(numerator / denominator)
    
    # 与第一个神经网络的正交惩罚项
    psi1 = apply_fn1(params1, xyz)
    overlap1 = jnp.vdot(psi1, psi2)
    norm1 = jnp.sqrt(jnp.vdot(psi1, psi1))
    norm2 = jnp.sqrt(jnp.vdot(psi2, psi2))
    
    # 使用更强的正交约束
    normalized_overlap1 = overlap1 / (norm1 * norm2)
    orthogonal_penalty1 = 50.0 * jnp.abs(normalized_overlap1)**2
    
    # 与球谐函数Y₁₊₁和Y₁₋₁的正交惩罚项
    Y_harmonics = spherical_harmonics(xyz)  # 形状: (..., 2)
    Y_1_plus_1 = Y_harmonics[..., 0]  # Y₁₊₁
    Y_1_minus_1 = Y_harmonics[..., 1]  # Y₁₋₁
    
    # 计算与Y₁₊₁的重叠积分
    overlap_Y_plus = jnp.vdot(Y_1_plus_1, psi2)
    norm_Y_plus = jnp.sqrt(jnp.vdot(Y_1_plus_1, Y_1_plus_1))
    normalized_overlap_Y_plus = overlap_Y_plus / (norm_Y_plus * norm2)
    orthogonal_penalty_Y_plus = 30.0 * jnp.abs(normalized_overlap_Y_plus)**2
    
    # 计算与Y₁₋₁的重叠积分
    overlap_Y_minus = jnp.vdot(Y_1_minus_1, psi2)
    norm_Y_minus = jnp.sqrt(jnp.vdot(Y_1_minus_1, Y_1_minus_1))
    normalized_overlap_Y_minus = overlap_Y_minus / (norm_Y_minus * norm2)
    orthogonal_penalty_Y_minus = 30.0 * jnp.abs(normalized_overlap_Y_minus)**2
    
    # 总损失：能量损失 + 与第一个神经网络的正交惩罚 + 与球谐函数的正交惩罚
    total_loss = (jnp.real(energy_loss) + 
                  jnp.real(orthogonal_penalty1) + 
                  jnp.real(orthogonal_penalty_Y_plus) + 
                  jnp.real(orthogonal_penalty_Y_minus))
    
    return total_loss

# 训练状态
def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, 3)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# 训练循环
@jax.jit
def train_step(state, xyz):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, xyz)
    state = state.apply_gradients(grads=grads)
    return state, loss

# 第二个神经网络的训练步骤（包含正交惩罚）
@jax.jit
def train_step_orthogonal(state2, state1, xyz):
    grad_fn = jax.value_and_grad(loss_fn_orthogonal)
    loss, grads = grad_fn(state2.params, state2.apply_fn, state1.params, state1.apply_fn, xyz)
    state2 = state2.apply_gradients(grads=grads)
    return state2, loss

# 主程序
def main():
    rng = jax.random.PRNGKey(0)
    model1 = PsiNet()
    model2 = PsiNet2()
    state1 = create_train_state(rng, model1, 1e-3)
    state2 = create_train_state(rng, model2, 1e-3)
    n_points = 8192
    n_epochs = 5000
    
    print("开始训练第一个神经网络...")
    # 记录第一个神经网络完整的训练过程
    all_losses1 = []
    
    for epoch in range(n_epochs):
        rng, subkey = jax.random.split(rng)
        xyz = sample_xyz(subkey, n_points)
        state1, loss1 = train_step(state1, xyz)
        
        # 记录每次训练的损失
        all_losses1.append(float(loss1))
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss1: {loss1:.6f}")
    
    # 计算第一个神经网络最后500次的平均值
    final_losses1 = all_losses1[-500:]
    avg_final_loss1 = np.mean(final_losses1)
    print(f"\n第一个神经网络最后500次训练的平均值:")
    print(f"平均损失函数: {avg_final_loss1:.6f}")
    
    print("开始训练第二个神经网络（带正交惩罚）...")
    # 记录第二个神经网络完整的训练过程
    all_losses2 = []
    all_overlaps = []
    all_overlaps_Y_plus = []
    all_overlaps_Y_minus = []
    
    for epoch in range(n_epochs):
        rng, subkey = jax.random.split(rng)
        xyz = sample_xyz(subkey, n_points)
        state2, loss2 = train_step_orthogonal(state2, state1, xyz)
        
        # 计算当前的重叠积分
        psi1_current = model1.apply(state1.params, xyz)
        psi2_current = model2.apply(state2.params, xyz)
        overlap_current = jnp.vdot(psi1_current, psi2_current)
        norm1_current = jnp.sqrt(jnp.vdot(psi1_current, psi1_current))
        norm2_current = jnp.sqrt(jnp.vdot(psi2_current, psi2_current))
        normalized_overlap_current = jnp.abs(overlap_current) / (norm1_current * norm2_current)
        
        # 计算与球谐函数的重叠积分
        Y_harmonics_current = spherical_harmonics(xyz)
        Y_1_plus_1_current = Y_harmonics_current[..., 0]
        Y_1_minus_1_current = Y_harmonics_current[..., 1]
        
        overlap_Y_plus_current = jnp.vdot(Y_1_plus_1_current, psi2_current)
        norm_Y_plus_current = jnp.sqrt(jnp.vdot(Y_1_plus_1_current, Y_1_plus_1_current))
        normalized_overlap_Y_plus_current = jnp.abs(overlap_Y_plus_current) / (norm_Y_plus_current * norm2_current)
        
        overlap_Y_minus_current = jnp.vdot(Y_1_minus_1_current, psi2_current)
        norm_Y_minus_current = jnp.sqrt(jnp.vdot(Y_1_minus_1_current, Y_1_minus_1_current))
        normalized_overlap_Y_minus_current = jnp.abs(overlap_Y_minus_current) / (norm_Y_minus_current * norm2_current)
        
        # 记录每次训练的数据
        all_losses2.append(float(loss2))
        all_overlaps.append(float(jnp.real(normalized_overlap_current)))
        all_overlaps_Y_plus.append(float(jnp.real(normalized_overlap_Y_plus_current)))
        all_overlaps_Y_minus.append(float(jnp.real(normalized_overlap_Y_minus_current)))
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss2: {loss2:.6f}")
    
    # 计算最后500次的平均值
    final_losses = all_losses2[-500:]
    final_overlaps = all_overlaps[-500:]
    final_overlaps_Y_plus = all_overlaps_Y_plus[-500:]
    final_overlaps_Y_minus = all_overlaps_Y_minus[-500:]
    avg_final_loss = np.mean(final_losses)
    avg_final_overlap = np.mean(final_overlaps)
    avg_final_overlap_Y_plus = np.mean(final_overlaps_Y_plus)
    avg_final_overlap_Y_minus = np.mean(final_overlaps_Y_minus)
    print(f"\n第二个神经网络最后500次训练的平均值:")
    print(f"平均损失函数: {avg_final_loss:.6f}")
    
    # 绘制第一个神经网络的损失函数图像
    epochs = list(range(n_epochs))
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, all_losses1, label='energy', color='blue', linewidth=2)
    plt.title('Energy for Neural Network Fitting of the Ground State of a Three-Dimensional Harmonic Oscillator in the Cartesian Coordinate System')
    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("已展示第一个神经网络损失函数图像")
    
    # 绘制第二个神经网络的损失函数图像
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, all_losses2, label='energy', color='red', linewidth=2)
    plt.title('Energy for Neural Network Fitting of the First Excited State of a Three-Dimensional Harmonic Oscillator in the Cartesian Coordinate System')
    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("已展示第二个神经网络损失函数图像")
    
    # 归一化输出
    grid_n = 40
    x = np.linspace(-5, 5, grid_n)
    y = np.linspace(-5, 5, grid_n)
    z = np.linspace(-5, 5, grid_n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    xyz_grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    # 计算两个波函数
    psi_pred1 = model1.apply(state1.params, jnp.array(xyz_grid))
    psi_pred2 = model2.apply(state2.params, jnp.array(xyz_grid))
    psi_pred1 = np.array(psi_pred1)
    psi_pred2 = np.array(psi_pred2)
    
    # 归一化
    norm1 = np.sqrt(np.sum(np.abs(psi_pred1)**2))
    norm2 = np.sqrt(np.sum(np.abs(psi_pred2)**2))
    psi_pred_norm1 = psi_pred1 / norm1
    psi_pred_norm2 = psi_pred2 / norm2
    
    # 计算重叠积分
    overlap = np.sum(np.conj(psi_pred_norm1) * psi_pred_norm2)
    print(f"归一化后的重叠积分: {np.abs(overlap):.6f}")
    
    psi_mod2_1 = np.abs(psi_pred_norm1)**2
    psi_mod2_2 = np.abs(psi_pred_norm2)**2
    psi_mod2_grid1 = psi_mod2_1.reshape((grid_n, grid_n, grid_n))
    psi_mod2_grid2 = psi_mod2_2.reshape((grid_n, grid_n, grid_n))

    # 自定义颜色映射（类似Jet）
    color = [
        [0, "rgba(255, 255, 255, 0)"],       # 低值映射为白色
        [0.1, "rgb(0, 0, 128)"],  # 深蓝色
        [0.3, "rgb(0, 255, 255)"],  # 青色
        [0.6, "rgb(255, 255, 0)"],  # 黄色
        [1, "rgb(128, 0, 0)"]       # 深红色
    ]
    
    # 设置参数
    opc = 0.3
    opc2 = 0.4
    offset = 0.1
    vmin = 0
    vmax = psi_mod2_grid1.max() * 1.0
    vmax2 = vmax * 1.1

    # 创建3D体积图
    fig = go.Figure(data=go.Volume(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=psi_mod2_grid1.ravel(),
        isomin=vmin,
        isomax=vmax,
        opacity=opc,
        surface_count=15,
        colorscale=color,
    ))

    # 定义数据边界
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[x_min-offset, x_max]),  # 设置x轴边界
            yaxis=dict(range=[y_min-offset, y_max]),  # 设置y轴边界
            zaxis=dict(range=[z_min-offset, z_max])   # 设置z轴边界
        ),
        title='|ψ(x,y,z)|² 3D分布'
    )

    # ---------------------------
    # 添加 y = 0 平面
    # ---------------------------
    from scipy.interpolate import griddata
    
    # 定义要添加切片的y平面
    y_plane = 0  # 切片处的y值

    # 提取 y = y_plane 处的数据
    y_idx = np.argmin(np.abs(y - y_plane))
    x_slice_y, z_slice_y = np.meshgrid(x, z, indexing='ij')
    density_slice_y = psi_mod2_grid1[:, y_idx, :]

    # 在x和z上创建网格
    x_unique_y = np.linspace(x_min, x_max, 100)
    z_unique_y = np.linspace(z_min, z_max, 100)
    x_grid_y, z_grid_y = np.meshgrid(x_unique_y, z_unique_y)

    # 将密度值插值到网格上
    points_y = np.vstack((x_slice_y.flatten(), z_slice_y.flatten())).T
    density_grid_y = griddata(points_y, density_slice_y.flatten(), (x_grid_y, z_grid_y), method='linear')

    # 将NaN替换为零（或其他合适的值）
    density_grid_y = np.nan_to_num(density_grid_y, nan=0.0)

    # 将表面添加到图中
    fig.add_trace(go.Surface(
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

    # ---------------------------
    # 添加 z = 0 平面
    # ---------------------------

    # 定义要添加切片的z平面
    z_plane = 0  # 切片处的z值

    # 提取 z = z_plane 处的数据
    z_idx = np.argmin(np.abs(z - z_plane))
    x_slice_z, y_slice_z = np.meshgrid(x, y, indexing='ij')
    density_slice_z = psi_mod2_grid1[:, :, z_idx]

    # 在x和y上创建网格
    x_unique_z = np.linspace(x_min, x_max, 100)
    y_unique_z = np.linspace(y_min, y_max, 100)
    x_grid_z, y_grid_z = np.meshgrid(x_unique_z, y_unique_z)

    # 将密度值插值到网格上
    points_z = np.vstack((x_slice_z.flatten(), y_slice_z.flatten())).T
    density_grid_z = griddata(points_z, density_slice_z.flatten(), (x_grid_z, y_grid_z), method='linear')

    # 将NaN替换为零（或其他合适的值）
    density_grid_z = np.nan_to_num(density_grid_z, nan=0.0)

    # 将表面添加到图中
    fig.add_trace(go.Surface(
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

    # ---------------------------
    # 添加 x = 0 平面
    # ---------------------------

    # 定义要添加切片的x平面
    x_plane = 0  # 切片处的x值

    # 提取 x = x_plane 处的数据
    x_idx = np.argmin(np.abs(x - x_plane))
    y_slice_x, z_slice_x = np.meshgrid(y, z, indexing='ij')
    density_slice_x = psi_mod2_grid1[x_idx, :, :]

    # 在y和z上创建网格
    y_unique_x = np.linspace(y_min, y_max, 100)
    z_unique_x = np.linspace(z_min, z_max, 100)
    y_grid_x, z_grid_x = np.meshgrid(y_unique_x, z_unique_x)

    # 将密度值插值到网格上
    points_x = np.vstack((y_slice_x.flatten(), z_slice_x.flatten())).T
    density_grid_x = griddata(points_x, density_slice_x.flatten(), (y_grid_x, z_grid_x), method='linear')

    # 将NaN替换为零（或其他合适的值）
    density_grid_x = np.nan_to_num(density_grid_x, nan=0.0)

    # 将表面添加到图中
    fig.add_trace(go.Surface(
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

    # 输出为html
    fig.write_html("Neural Network Fitting of the Ground State of a Three-Dimensional Harmonic Oscillator in the Cartesian Coordinate System.html")
    print("已保存为Neural Network Fitting of the Ground State of a Three-Dimensional Harmonic Oscillator in the Cartesian Coordinate System.html")
    
    # 创建第二个波函数的3D图
    fig2 = go.Figure(data=go.Volume(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=psi_mod2_grid2.ravel(),
        isomin=vmin,
        isomax=vmax,
        opacity=opc,
        surface_count=15,
        colorscale=color,
    ))
    
    fig2.update_layout(
        scene=dict(
            xaxis=dict(range=[x_min-offset, x_max]),
            yaxis=dict(range=[y_min-offset, y_max]),
            zaxis=dict(range=[z_min-offset, z_max])
        ),
        title='|ψ₂(x,y,z)|² 3D分布（第二个神经网络）'
    )
    
    # 为第二个图添加相同的切片平面
    # y = 0 平面
    density_slice_y2 = psi_mod2_grid2[:, y_idx, :]
    density_grid_y2 = griddata(points_y, density_slice_y2.flatten(), (x_grid_y, z_grid_y), method='linear')
    density_grid_y2 = np.nan_to_num(density_grid_y2, nan=0.0)
    
    fig2.add_trace(go.Surface(
        x=x_grid_y,
        y=np.full_like(x_grid_y, y_min-offset),
        z=z_grid_y,
        cmin=vmin,
        cmax=vmax2,
        surfacecolor=density_grid_y2,
        colorscale=color,
        opacity=opc2,
        showscale=False
    ))
    
    # z = 0 平面
    density_slice_z2 = psi_mod2_grid2[:, :, z_idx]
    density_grid_z2 = griddata(points_z, density_slice_z2.flatten(), (x_grid_z, y_grid_z), method='linear')
    density_grid_z2 = np.nan_to_num(density_grid_z2, nan=0.0)
    
    fig2.add_trace(go.Surface(
        x=x_grid_z,
        y=y_grid_z,
        z=np.full_like(x_grid_z, z_min-offset),
        cmin=vmin,
        cmax=vmax2,
        surfacecolor=density_grid_z2,
        colorscale=color,
        opacity=opc2,
        showscale=False
    ))
    
    # x = 0 平面
    density_slice_x2 = psi_mod2_grid2[x_idx, :, :]
    density_grid_x2 = griddata(points_x, density_slice_x2.flatten(), (y_grid_x, z_grid_x), method='linear')
    density_grid_x2 = np.nan_to_num(density_grid_x2, nan=0.0)
    
    fig2.add_trace(go.Surface(
        x=np.full_like(y_grid_x, x_min-offset),
        y=y_grid_x,
        z=z_grid_x,
        cmin=vmin,
        cmax=vmax2,
        surfacecolor=density_grid_x2,
        colorscale=color,
        opacity=opc2,
        showscale=False
    ))
    
    fig2.write_html("Neural Network Fitting of the First Excited State of a Three-Dimensional Harmonic Oscillator in the Cartesian Coordinate System.html")
    print("已保存为Neural Network Fitting of the First Excited State of a Three-Dimensional Harmonic Oscillator in the Cartesian Coordinate System.html")

if __name__ == "__main__":
    main()
