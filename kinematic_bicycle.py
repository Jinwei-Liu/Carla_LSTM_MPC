import torch
import torch.nn as nn

class Kinematic_Bicycle_MPC(nn.Module):
    """Kinematic bicycle model with side slip angle"""
    def __init__(self, dt=0.05, wheelbase=3.76, lr=2.00, lf=1.76):
        super().__init__()
        self.s_dim = 4  # [x, y, v, psi]
        self.a_dim = 2  # [a, delta_f]
        self._dt = dt
        self._L = wheelbase
        self._lr = lr
        self._lf = lf
        
    def forward(self, X, action):
        DT = self._dt
        k1 = DT * self._f(X, action)
        X_next = X + k1
        return X_next
    
    def _f(self, state, action):
        """State derivatives based on equations (1a)-(1e)"""
        v = state[..., 2]
        psi = state[..., 3]
        
        a, delta_f = action.split(1, dim=-1)
        a = a.squeeze(-1)
        delta_f = delta_f.squeeze(-1)
        
        # Side slip angle: β = arctan((lr/(lf+lr)) * tan(δf))
        beta = torch.atan((self._lr / (self._lf + self._lr)) * torch.tan(delta_f))
        
        dstate = torch.zeros_like(state)
        dstate[..., 0] = v * torch.cos(psi + beta)  # ẋ = v cos(ψ + β)
        dstate[..., 1] = v * torch.sin(psi + beta)  # ẏ = v sin(ψ + β)
        dstate[..., 2] = a                          # v̇ = a
        dstate[..., 3] = (v / self._L) * torch.sin(beta)  # ψ̇ = (v/L) sin(β)
        
        return dstate
    
    def grad_input(self, X, action):
        """Compute Jacobian matrices A and B for linearization"""
        DT = self._dt
        batch_shape = X.shape[:-1]
        
        v = X[..., 2]
        psi = X[..., 3]
        
        a, delta_f = action.split(1, dim=-1)
        delta_f = delta_f.squeeze(-1)
        
        # Side slip angle and its derivative
        tan_delta = torch.tan(delta_f)
        beta = torch.atan((self._lr / (self._lf + self._lr)) * tan_delta)
        
        coeff = self._lr / (self._lf + self._lr)
        sec2_delta = 1.0 / (torch.cos(delta_f) ** 2)
        dbeta_ddelta = coeff * sec2_delta / (1.0 + (coeff * tan_delta) ** 2)
        
        A = X.new_zeros(*batch_shape, self.s_dim, self.s_dim)
        B = X.new_zeros(*batch_shape, self.s_dim, self.a_dim)
        
        cos_psi_beta = torch.cos(psi + beta)
        sin_psi_beta = torch.sin(psi + beta)
        
        # A matrix
        A[..., 0, 2] = cos_psi_beta
        A[..., 0, 3] = -v * sin_psi_beta
        A[..., 1, 2] = sin_psi_beta
        A[..., 1, 3] = v * cos_psi_beta
        A[..., 3, 2] = (1.0 / self._L) * torch.sin(beta)
        
        # B matrix
        B[..., 0, 1] = -v * sin_psi_beta * dbeta_ddelta
        B[..., 1, 1] = v * cos_psi_beta * dbeta_ddelta
        B[..., 2, 0] = 1.0
        B[..., 3, 1] = (v / self._L) * torch.cos(beta) * dbeta_ddelta
        
        # Discretization
        eye = torch.eye(self.s_dim, dtype=X.dtype, device=X.device)
        eye = eye.expand(*batch_shape, -1, -1)
        A_d = eye + DT * A
        B_d = DT * B
        
        return A_d, B_d
    
    def get_beta(self, delta_f):
        """Compute side slip angle"""
        return torch.atan((self._lr / (self._lf + self._lr)) * torch.tan(delta_f))