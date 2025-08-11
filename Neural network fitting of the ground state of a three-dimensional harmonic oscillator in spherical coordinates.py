import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad, vmap
from functools import partial
import optax
import matplotlib.pyplot as plt
import numpy as np

# 常数定义
hbar = 197.3269804
mu = 939.0
omega = 10.0 / hbar

# swish激活函数
def swish(x):
    return x * nn.sigmoid(x)

# 神经网络定义
class RadialNet(nn.Module):
    @nn.compact
    def __call__(self, r):
        x = nn.Dense(128)(r)
        x = swish(x)
        x = nn.Dense(128)(x)
        x = swish(x)
        x = nn.Dense(64)(x)
        x = swish(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)

# 径向能量损失函数
def energy_loss(params, model, r, key):
    psi = model.apply(params, r)
    # 自动微分求导 - 使用vmap处理向量化求导
    def psi_fn_scalar(r_scalar):
        return model.apply(params, r_scalar.reshape(1, -1)).squeeze()
    
    # 使用vmap对每个r值求导
    dpsi_dr = vmap(grad(psi_fn_scalar))(r.squeeze())
    d2psi_dr2 = vmap(grad(grad(psi_fn_scalar)))(r.squeeze())
    
    # 势能项
    V = 0.5 * mu * omega**2 * r.squeeze()**2
    
    # 处理r=0的奇点问题
    r_safe = jnp.maximum(r.squeeze(), 1e-8)
    
    # 正确的径向拉普拉斯算符：∇²ψ = ∂²ψ/∂r² + (2/r) * ∂ψ/∂r
    kinetic = - (hbar**2) / (2 * mu) * (d2psi_dr2 + 2/r_safe * dpsi_dr)
    potential = V * psi
    energy_density = psi * (kinetic + potential)
    
    # 蒙特卡洛积分（均匀采样）
    r_max = 8.0  # 积分上限
    # 对于均匀采样，权重是 r_max，积分核是 4πr²
    # 但需要除以采样密度来得到正确的积分
    integrand = energy_density * 4 * jnp.pi * r.squeeze()**2
    total_energy = jnp.mean(integrand) * r_max  # 乘以积分区间长度
    
    # 计算归一化因子
    norm_integrand = (psi**2) * 4 * jnp.pi * r.squeeze()**2
    norm_factor = jnp.sqrt(jnp.mean(norm_integrand) * r_max)
    
    # 返回归一化后的能量期望值
    return total_energy / (norm_factor**2)

# 归一化处理
def normalize(params, model, key, num_samples=10000, r_max=8.0):
    r = random.uniform(key, (num_samples, 1), minval=0.0, maxval=r_max)
    psi = model.apply(params, r)
    # 正确的归一化计算：积分 ψ²(r) * 4πr² dr
    # 对于均匀采样，权重是 r_max，积分核是 4πr²
    norm = jnp.sqrt(jnp.mean((psi**2) * 4 * jnp.pi * r.squeeze()**2) * r_max)
    return norm

# 训练主流程
def train():
    key = random.PRNGKey(0)
    model = RadialNet()
    r_max = 8.0
    num_samples = 8192
    r = random.uniform(key, (num_samples, 1), minval=0.0, maxval=r_max)
    params = model.init(key, r)
    lr = 1e-4
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, key):
        loss, grads = jax.value_and_grad(energy_loss)(params, model, r, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 记录损失函数值
    losses = []
    steps = []
    
    for i in range(2000):
        key, subkey = random.split(key)
        params, opt_state, loss = step(params, opt_state, subkey)
        
        # 记录损失值
        losses.append(float(loss))
        steps.append(i)
        
        if i % 100 == 0:
            print(f"step {i}, loss: {loss}")
    
    # 计算最后500次损失函数平均值
    if len(losses) >= 500:
        last_500_losses = losses[-500:]
        avg_loss = np.mean(last_500_losses)
        print(f"最后500次损失函数平均值: {avg_loss:.6f}")
    else:
        print(f"训练步数不足500步，当前只有{len(losses)}步")
        if len(losses) > 0:
            avg_loss = np.mean(losses)
            print(f"所有损失函数平均值: {avg_loss:.6f}")
    
    # 绘制损失函数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, 'b-', linewidth=1)
    plt.xlabel('steps')
    plt.ylabel('energy')
    plt.title('Energy for neural network fitting of the ground state of a three-dimensional harmonic oscillator in spherical coordinates')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    norm = normalize(params, model, key)
    print("归一化因子:", norm)
    
    # 生成直角坐标网格并计算波函数值
    def generate_cartesian_wavefunction(params, model, x_range=(-5, 5), y_range=(-5, 5), z_range=(-5, 5), resolution=25):
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        z = np.linspace(z_range[0], z_range[1], resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        psi_values = np.zeros_like(X)
        
        # 计算每个点的波函数值
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    r = np.sqrt(X[i,j,k]**2 + Y[i,j,k]**2 + Z[i,j,k]**2)
                    if r <= 5.0:  # 只计算在积分范围内的点
                        psi_val = model.apply(params, np.array([[r]]))
                        psi_values[i,j,k] = float(psi_val.squeeze()) / norm  # 应用归一化
        
        return X, Y, Z, psi_values
    
    # 生成直角坐标波函数
    print("正在生成直角坐标波函数...")
    X, Y, Z, psi_values = generate_cartesian_wavefunction(params, model)
    
    # 保存为DS3文件格式
    def save_ds3_file(X, Y, Z, psi_values, filename='Neural network fitting of the ground state of a three-dimensional harmonic oscillator in spherical coordinates.ds3'):
        with open(filename, 'w') as f:
            # 写入文件头
            f.write("# DS3文件格式 - 三维谐振子基态波函数\n")
            f.write(f"# 网格大小: {X.shape[0]} x {X.shape[1]} x {X.shape[2]}\n")
            f.write(f"# X范围: {X.min():.3f} 到 {X.max():.3f}\n")
            f.write(f"# Y范围: {Y.min():.3f} 到 {Y.max():.3f}\n")
            f.write(f"# Z范围: {Z.min():.3f} 到 {Z.max():.3f}\n")
            f.write("# X Y Z |psi|^2\n")
            
            # 写入数据
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k in range(X.shape[2]):
                        x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                        psi_squared = psi_values[i,j,k]**2  # 概率密度
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {psi_squared:.6f}\n")
    
    # 保存DS3文件
    save_ds3_file(X, Y, Z, psi_values)
    print(f"波函数已保存为 Neural network fitting of the ground state of a three-dimensional harmonic oscillator in spherical coordinates.ds3")
    
    return params, model, norm

# 运行训练
if __name__ == "__main__":
    params, model, norm = train()
