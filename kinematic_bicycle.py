import torch
import torch.nn as nn
import numpy as np

class Kinematic_Bicycle_MPC(nn.Module):
    """Kinematic bicycle model with side slip angle"""
    def __init__(self, dt=0.05, wheelbase=3.45, lr=3.45, lf=0.00):
        super().__init__()
        self.s_dim = 4  # [x, y, psi, v]
        self.a_dim = 2  # [a, delta_f]
        self._dt = dt
        self._L = wheelbase
        self._lr = lr
        self._lf = lf
        
        # 动作限制
        self.a_min = -10.0  # 最小加速度 (m/s²)
        self.a_max = 10.0   # 最大加速度 (m/s²)
        self.delta_f_min = -np.deg2rad(70)  # 最小前轮转向角 (rad)
        self.delta_f_max = np.deg2rad(70)   # 最大前轮转向角 (rad)
        
    def clip_action(self, action):
        """限制动作在允许范围内"""
        a, delta_f = action.split(1, dim=-1)
        
        # 限制加速度
        a_clipped = torch.clamp(a, self.a_min, self.a_max)
        
        # 限制前轮转向角
        delta_f_clipped = torch.clamp(delta_f, self.delta_f_min, self.delta_f_max)
        
        return torch.cat([a_clipped, delta_f_clipped], dim=-1)
        
    def forward(self, X, action):
        """使用四阶龙格库塔方法进行状态积分"""
        # 首先限制动作范围
        action_clipped = self.clip_action(action)
        
        DT = self._dt
        
        # 四阶龙格库塔积分
        k1 = self._f(X, action_clipped)
        k2 = self._f(X + 0.5 * DT * k1, action_clipped)
        k3 = self._f(X + 0.5 * DT * k2, action_clipped)
        k4 = self._f(X + DT * k3, action_clipped)
        
        # RK4公式：X_next = X + (DT/6) * (k1 + 2*k2 + 2*k3 + k4)
        X_next = X + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # 对航向角psi进行角度归一化，限制在[-π, π]范围内
        X_next = self._normalize_angle(X_next)
        
        return X_next
    
    def _normalize_angle(self, X):
        """将航向角psi归一化到[-π, π]范围"""
        # 避免原地操作，使用torch.cat重建张量
        x = X[..., 0:1]  # x坐标
        y = X[..., 1:2]  # y坐标
        psi = X[..., 2:3]  # 航向角
        v = X[..., 3:4]  # 速度
        
        # 归一化航向角，避免原地操作
        psi_normalized = torch.atan2(torch.sin(psi), torch.cos(psi))
        
        # 重建张量而不是原地修改
        X_normalized = torch.cat([x, y, psi_normalized, v], dim=-1)
        return X_normalized
    
    def _f(self, state, action):
        """State derivatives based on equations (1a)-(1e)"""
        psi = state[..., 2] 
        v = state[..., 3] 
        
        a, delta_f = action.split(1, dim=-1)
        a = a.squeeze(-1)
        delta_f = delta_f.squeeze(-1)
        
        # Side slip angle: β = arctan((lr/(lf+lr)) * tan(δf))
        beta = torch.atan((self._lr / (self._lf + self._lr)) * torch.tan(delta_f))
        
        dstate = torch.zeros_like(state)
        dstate[..., 0] = v * torch.cos(psi + beta)  # ẋ = v cos(ψ + β)
        dstate[..., 1] = v * torch.sin(psi + beta)  # ẏ = v sin(ψ + β)
        dstate[..., 2] = (v / self._L) * torch.sin(beta)  # 交换：ψ̇ = (v/L) sin(β)
        dstate[..., 3] = a                          # 交换：v̇ = a
        
        return dstate
    
    def grad_input(self, X, action):
        """计算状态转移矩阵，使得 X_next = A_d @ X + B_d @ u
        这是一个时变线性系统，每个时间步都要重新计算
        """
        action_clipped = self.clip_action(action)
        DT = self._dt
        batch_shape = X.shape[:-1]
        
        psi = X[..., 2]  
        v = X[..., 3] 
        
        a, delta_f = action_clipped.split(1, dim=-1)
        a = a.squeeze(-1)
        delta_f = delta_f.squeeze(-1)
        
        # 计算侧滑角 β (只与控制输入 δf 有关)
        tan_delta = torch.tan(delta_f)
        beta = torch.atan((self._lr / (self._lf + self._lr)) * tan_delta)
        
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)
        cos_psi_beta = torch.cos(psi + beta)
        sin_psi_beta = torch.sin(psi + beta)
        
        # 初始化矩阵
        A_d = X.new_zeros(*batch_shape, self.s_dim, self.s_dim)
        B_d = X.new_zeros(*batch_shape, self.s_dim, self.a_dim)
        C_d = X.new_zeros(*batch_shape, self.s_dim)  # 常数项
        
        # 构造状态转移矩阵
        # x_next = x + dt * v * cos(ψ + β)
        # 注意：cos(ψ + β) 不能分解为关于 x,y,ψ,v 的线性组合
        # 但我们可以在给定当前 ψ 和 β 的情况下线性化
        
        # 对于 x 方程：x_next = x + dt * v * cos(ψ + β)
        # 其中 ψ 和 β 在当前时刻是已知的常数
        A_d[..., 0, 0] = 1.0                      # ∂x_next/∂x
        A_d[..., 0, 1] = 0.0                      # ∂x_next/∂y
        A_d[..., 0, 2] = 0.0                      # ∂x_next/∂ψ (线性化后为0)
        A_d[..., 0, 3] = DT * cos_psi_beta        # ∂x_next/∂v
        
        # 对于 y 方程：y_next = y + dt * v * sin(ψ + β)
        A_d[..., 1, 0] = 0.0                      # ∂y_next/∂x
        A_d[..., 1, 1] = 1.0                      # ∂y_next/∂y
        A_d[..., 1, 2] = 0.0                      # ∂y_next/∂ψ (线性化后为0)
        A_d[..., 1, 3] = DT * sin_psi_beta        # ∂y_next/∂v
        
        # 对于 ψ 方程：ψ_next = ψ + dt * (v/L) * sin(β)
        A_d[..., 2, 0] = 0.0                      # ∂ψ_next/∂x
        A_d[..., 2, 1] = 0.0                      # ∂ψ_next/∂y
        A_d[..., 2, 2] = 1.0                      # ∂ψ_next/∂ψ
        A_d[..., 2, 3] = DT * sin_beta / self._L  # ∂ψ_next/∂v
        
        # 对于 v 方程：v_next = v + dt * a
        A_d[..., 3, 0] = 0.0                      # ∂v_next/∂x
        A_d[..., 3, 1] = 0.0                      # ∂v_next/∂y
        A_d[..., 3, 2] = 0.0                      # ∂v_next/∂ψ
        A_d[..., 3, 3] = 1.0                      # ∂v_next/∂v
        
        # 控制矩阵 B_d
        # 注意：由于 β 依赖于 δf，所有包含 β 的项都会受影响
        
        # 由于 x_next = x + dt * v * cos(ψ + β(δf))
        # 我们需要把 β 的影响放入 B_d 中
        # 但这很复杂，因为 β 是 δf 的非线性函数
        
        # 简化方案：将控制输入的影响直接编码
        # 对于加速度 a 的影响
        B_d[..., 0, 0] = 0.0  # x 不直接依赖 a
        B_d[..., 1, 0] = 0.0  # y 不直接依赖 a
        B_d[..., 2, 0] = 0.0  # ψ 不直接依赖 a
        B_d[..., 3, 0] = DT    # v_next = v + dt * a
        
        # 对于转向角 δf 的影响（通过 β）
        # 这里需要计算 ∂β/∂δf
        coeff = self._lr / (self._lf + self._lr)
        sec2_delta = 1.0 / (torch.cos(delta_f) ** 2 + 1e-8)
        dbeta_ddelta = coeff * sec2_delta / (1.0 + (coeff * tan_delta) ** 2 + 1e-8)
        
        # x 通过 β 受 δf 影响
        B_d[..., 0, 1] = -DT * v * sin_psi_beta * dbeta_ddelta
        # y 通过 β 受 δf 影响
        B_d[..., 1, 1] = DT * v * cos_psi_beta * dbeta_ddelta
        # ψ 通过 β 受 δf 影响
        B_d[..., 2, 1] = DT * v * cos_beta * dbeta_ddelta / self._L
        # v 不受 δf 影响
        B_d[..., 3, 1] = 0.0
        
        return A_d, B_d
    
    def get_beta(self, delta_f):
        """Compute side slip angle"""
        return torch.atan((self._lr / (self._lf + self._lr)) * torch.tan(delta_f))
    
    def get_action_bounds(self):
        """返回动作的上下界，便于MPC优化器使用"""
        return {
            'a_min': self.a_min,
            'a_max': self.a_max, 
            'delta_f_min': self.delta_f_min,
            'delta_f_max': self.delta_f_max
        }