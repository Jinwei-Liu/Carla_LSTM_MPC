"""
Vehicle LSTM-MPC Controller for CARLA with Multiple Target Points
Modified to support multiple target points across the MPC horizon
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Import kinematic bicycle model and MPC solver
from kinematic_bicycle import Kinematic_Bicycle_MPC
from mpc import mpc
from mpc.mpc import QuadCost, GradMethods

class VehicleLSTMMPCDataset(Dataset):
    """Vehicle LSTM-MPC dataset using relative coordinates without normalization"""
    
    def __init__(self, sequences):
        self.sequences = sequences
        print(f"Dataset initialized with {len(sequences)} sequences")
        print("Using raw relative coordinate data (no normalization)")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # History states and controls in relative coordinates
        hist_states = torch.FloatTensor(seq['hist_states'])  # (61, 4) [x, y, yaw, speed]
        hist_controls = torch.FloatTensor(seq['hist_controls'])  # (61, 3) [throttle, brake, steer]
        
        # Convert controls to [a, delta_f] format
        max_accel = 10.0  # m/s^2
        max_steer = np.deg2rad(70)
        
        accelerations = (hist_controls[:, 0] - hist_controls[:, 1]) * max_accel
        steer_angles = hist_controls[:, 2] * max_steer
        
        hist_actions = torch.stack([accelerations, steer_angles], dim=1)  # (61, 2)
        
        # Current state at origin
        current_state = torch.FloatTensor(seq['current_state'])  # (4,) [0, 0, 0, speed]
        
        # Future states and controls
        future_states = torch.FloatTensor(seq['future_states'])  # (100, 4) [x, y, yaw, speed]
        future_controls = torch.FloatTensor(seq['future_controls'])  # (100, 3)
        
        # Convert future controls
        future_accelerations = (future_controls[:, 0] - future_controls[:, 1]) * max_accel
        future_steer_angles = future_controls[:, 2] * max_steer
        future_actions = torch.stack([future_accelerations, future_steer_angles], dim=1)  # (100, 2)
        
        # Concatenate states and actions for LSTM input
        input_seq = torch.cat([hist_states, hist_actions], dim=1)  # (61, 6) [x, y, yaw, speed, a, delta_f]
        
        return {
            'input_seq': input_seq,
            'current_state': current_state,
            'future_states': future_states,
            'future_actions': future_actions
        }

class VehicleLSTMMPC(nn.Module):
    """LSTM network for predicting MPC parameters with multiple targets"""
    
    def __init__(self, 
                 input_dim=6,      # [x, y, yaw, speed, a, delta_f]
                 hidden_dim=128,
                 num_layers=1,
                 state_dim=4,      # [x, y, yaw, speed]
                 control_dim=2,    # [a, delta_f]
                 num_targets=1):   # Number of target points
        super(VehicleLSTMMPC, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.num_targets = num_targets
        
        # Control bounds
        self.a_bound = 10.0  # acceleration bound: [-10, 10]
        self.delta_f_bound = np.deg2rad(70)  # steering angle bound: [-70°, 70°]
        
        # LSTM encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True)
        
        # State weight prediction head
        self.state_weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
            nn.Softplus()  # Ensure positive weights
        )
        
        # Control weight prediction head
        self.control_weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, control_dim),
            nn.Softplus()  # Ensure positive weights
        )
        
        # Target state prediction head - now outputs multiple targets
        self.target_state_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim * num_targets)  # Output multiple targets
        )
        
        # Target control prediction head - now outputs multiple targets
        self.target_control_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, control_dim * num_targets)  # Output multiple targets
        )
    
    def apply_control_bounds(self, control_raw):
        """Apply bounds to control outputs using tanh activation"""
        # Reshape to handle multiple targets: (batch_size, num_targets, control_dim)
        batch_size = control_raw.size(0)
        control_reshaped = control_raw.view(batch_size, self.num_targets, self.control_dim)
        
        # Split the control outputs
        a_raw = control_reshaped[:, :, 0:1]  # acceleration
        delta_f_raw = control_reshaped[:, :, 1:2]  # steering angle
        
        # Apply bounds using tanh
        a_bounded = torch.tanh(a_raw) * self.a_bound
        delta_f_bounded = torch.tanh(delta_f_raw) * self.delta_f_bound
        
        # Concatenate back
        control_bounded = torch.cat([a_bounded, delta_f_bounded], dim=2)
        
        return control_bounded
    
    def apply_state_bounds(self, state_raw):
        """Apply bounds to state outputs, specifically for yaw angle"""
        # Reshape to handle multiple targets: (batch_size, num_targets, state_dim)
        batch_size = state_raw.size(0)
        state_reshaped = state_raw.view(batch_size, self.num_targets, self.state_dim)
        
        # Split the state outputs: [x, y, yaw, speed]
        x = state_reshaped[:, :, 0:1]      # x position (no bounds)
        y = state_reshaped[:, :, 1:2]      # y position (no bounds) 
        yaw_raw = state_reshaped[:, :, 2:3]    # yaw angle (需要限制)
        speed_raw = state_reshaped[:, :, 3:4]  # speed (限制为正值)
        
        # Apply yaw angle bounds: limit to [-π, π]
        yaw_bounded = torch.atan2(torch.sin(yaw_raw), torch.cos(yaw_raw))
        
        # Apply speed bounds: ensure non-negative speed
        speed_bounded = torch.relu(speed_raw)
        
        # Concatenate back
        state_bounded = torch.cat([x, y, yaw_bounded, speed_bounded], dim=2)
        
        return state_bounded
    
    def forward(self, x):
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Predict MPC parameters
        state_weights = self.state_weight_head(last_hidden)  # (batch_size, 4)


        control_weights = self.control_weight_head(last_hidden)  # (batch_size, 2)

        # Predict targets separately - now multiple targets
        target_state_raw = self.target_state_head(last_hidden)  # (batch_size, 4 * num_targets)
        target_control_raw = self.target_control_head(last_hidden)  # (batch_size, 2 * num_targets)
        
        # Apply bounds to targets
        target_state = self.apply_state_bounds(target_state_raw)  # (batch_size, 4 * num_targets)
        target_control = self.apply_control_bounds(target_control_raw)  # (batch_size, 2 * num_targets)
        
        # Combine state and control targets
        target = torch.cat([target_state, target_control], dim=2)  # (batch_size, num_targets, 6)

        return state_weights, control_weights, target.view(target.size(0), -1)

class VehicleMPCSolver:
    """MPC solver using mpc.pytorch library with kinematic bicycle model and multiple targets"""
    
    def __init__(self, dt=0.05, horizon=100, device='cuda', downsample_factor=1, num_targets=1):
        """
        Args:
            dt: Original sampling time (0.05s)
            horizon: Original horizon steps (100)
            device: Computing device
            downsample_factor: Factor to downsample MPC computation
            num_targets: Number of target points to use
        """
        self.original_dt = dt
        self.downsample_factor = downsample_factor
        self.mpc_dt = dt * downsample_factor  # MPC sampling time
        self.mpc_horizon = horizon // downsample_factor + 1  # Downsampled horizon
        self.device = device
        self.num_targets = num_targets
        
        # Create kinematic bicycle model with MPC sampling time
        self.model = Kinematic_Bicycle_MPC(dt=self.mpc_dt)
        self.n_state = self.model.s_dim  # 4: [x, y, psi, v]
        self.n_ctrl = self.model.a_dim   # 2: [a, delta_f]
        
        # Control bounds
        self.u_min = torch.tensor([-10.0, -np.deg2rad(70)], device=device)  # [a_min, delta_min]
        self.u_max = torch.tensor([10.0, np.deg2rad(70)], device=device)    # [a_max, delta_max]
        
        print(f"MPC Solver initialized:")
        print(f"  Original dt: {self.original_dt}s, MPC dt: {self.mpc_dt}s")
        print(f"  Downsample factor: {self.downsample_factor}")
        print(f"  MPC horizon: {self.mpc_horizon} steps")
        print(f"  Number of targets: {self.num_targets}")
        
    def distribute_targets_across_horizon(self, targets):
        """
        Distribute multiple targets across the MPC horizon
        
        Args:
            targets: (batch_size, 6 * num_targets) - concatenated state and control targets
            
        Returns:
            distributed_targets: (mpc_horizon, batch_size, 6) - targets for each time step
        """
        batch_size = targets.size(0)
        
        # Reshape targets to (batch_size, num_targets, 6)
        targets_reshaped = targets.view(batch_size, self.num_targets, 6)
        
        # Calculate how many time steps each target should cover
        steps_per_target = self.mpc_horizon // self.num_targets
        remaining_steps = self.mpc_horizon % self.num_targets
        
        # Create the distributed targets tensor
        distributed_targets = torch.zeros(self.mpc_horizon, batch_size, 6, 
                                        device=targets.device, dtype=targets.dtype)
        
        current_step = 0
        for target_idx in range(self.num_targets):
            # Calculate how many steps this target covers
            steps_for_this_target = steps_per_target
            if target_idx < remaining_steps:  # Distribute remaining steps to first targets
                steps_for_this_target += 1
            
            # Assign this target to the corresponding time steps
            end_step = current_step + steps_for_this_target
            distributed_targets[current_step:end_step] = targets_reshaped[:, target_idx, :].unsqueeze(0)
            
            current_step = end_step
        
        return distributed_targets
        
    def solve(self, initial_state, state_weights, control_weights, target):
        """
        Solve MPC optimization using mpc.pytorch with multiple targets
        
        Args:
            initial_state: (batch_size, 4) [x, y, yaw, speed]
            state_weights: (batch_size, 4) state weights
            control_weights: (batch_size, 2) control weights
            target: (batch_size, 6 * num_targets) multiple target states and controls
            
        Returns:
            predicted_states: (batch_size, mpc_horizon-1, 4)
            optimal_controls: (batch_size, mpc_horizon-1, 2)
        """
        batch_size = initial_state.size(0)

        # Setup bounds for MPC
        u_lower = self.u_min.unsqueeze(0).unsqueeze(0).repeat(self.mpc_horizon, batch_size, 1)
        u_upper = self.u_max.unsqueeze(0).unsqueeze(0).repeat(self.mpc_horizon, batch_size, 1)
        
        # Build quadratic cost matrices
        weights = torch.cat([state_weights, control_weights], dim=1)  # (batch, 6)
         
        # Create diagonal cost matrix C for each batch
        weight_diag = torch.diag_embed(weights)
        C = weight_diag.unsqueeze(0).repeat(self.mpc_horizon, 1, 1, 1)  # [T, batch_size, 6, 6]

        # Distribute targets across horizon
        distributed_targets = self.distribute_targets_across_horizon(target)  # [T, batch_size, 6]
        
        # Build cost vectors with distributed targets
        c = -torch.sqrt(weights).unsqueeze(0) * distributed_targets  # [T, batch_size, 6]

        # Create QuadCost object
        cost = QuadCost(C, c)
        
        # Setup MPC controller
        ctrl = mpc.MPC(
            n_state=self.n_state,
            n_ctrl=self.n_ctrl,
            T=self.mpc_horizon,
            lqr_iter=10,
            grad_method=GradMethods.ANALYTIC,
            exit_unconverged=False,
            detach_unconverged=False,
            backprop=True,
            verbose=0
        )
        
        # Solve MPC problem
        x_pred, u_opt, _ = ctrl(initial_state, cost, self.model)

        # Transpose to (batch, time, dim)
        predicted_states = x_pred.transpose(0, 1)[:,1:,:]
        optimal_controls = u_opt.transpose(0, 1)[:,1:,:]
        
        return predicted_states, optimal_controls
        
class VehicleLSTMMPCTrainer:
    """Trainer for Vehicle LSTM-MPC model with multiple targets support"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 downsample_factor=1, num_targets=1):
        self.model = model.to(device)
        self.device = device
        self.downsample_factor = downsample_factor
        self.num_targets = num_targets
        self.train_losses = []
        self.val_losses = []
        
        # MPC solver with downsampling and multiple targets
        self.mpc_solver = VehicleMPCSolver(
            dt=0.05, 
            horizon=100, 
            device=device,
            downsample_factor=downsample_factor,
            num_targets=num_targets
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        print(f"Trainer initialized with downsample_factor={downsample_factor}, num_targets={num_targets}")
        
    def train_model(self, train_loader, val_loader, epochs=100, lr=1e-3, patience=15, save_dir='models'):
        
        os.makedirs(save_dir, exist_ok=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Downsample factor: {self.downsample_factor}")
        print(f"Number of targets: {self.num_targets}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_trajectory_loss = 0.0
            train_control_loss = 0.0
            train_reg_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                input_seq = batch['input_seq'].to(self.device)
                current_state = batch['current_state'].to(self.device)
                future_states = batch['future_states'].to(self.device)
                future_actions = batch['future_actions'].to(self.device)

                # Forward pass
                state_weights, control_weights, target = self.model(input_seq)
                
                # Compute loss
                loss_dict = self.compute_loss(
                    state_weights, control_weights, target,
                    current_state, future_states, future_actions
                )
                
                loss = loss_dict['total']
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_trajectory_loss += loss_dict['trajectory'].item()
                train_control_loss += loss_dict['control'].item()
                train_reg_loss += loss_dict['regularization'].item()
            
            train_loss /= len(train_loader)
            train_trajectory_loss /= len(train_loader)
            train_control_loss /= len(train_loader)
            train_reg_loss /= len(train_loader)
            
            # Validation phase
            val_loss, val_trajectory_loss, val_control_loss, val_reg_loss = self.evaluate(val_loader)
            
            scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.6f} (Traj: {train_trajectory_loss:.6f}, Ctrl: {train_control_loss:.6f}, Reg: {train_reg_loss:.6f})")
            print(f"  Val Loss: {val_loss:.6f} (Traj: {val_trajectory_loss:.6f}, Ctrl: {val_control_loss:.6f}, Reg: {val_reg_loss:.6f})")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                model_path = os.path.join(save_dir, f'best_vehicle_lstm_mpc_ds{self.downsample_factor}_targets{self.num_targets}.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': {
                        'input_dim': self.model.input_dim,
                        'hidden_dim': self.model.hidden_dim,
                        'num_layers': self.model.num_layers,
                        'state_dim': self.model.state_dim,
                        'control_dim': self.model.control_dim,
                        'num_targets': self.model.num_targets
                    },
                    'downsample_factor': self.downsample_factor,
                    'num_targets': self.num_targets,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, model_path)
                print(f"  ✓ Best model saved to {model_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
    def compute_loss(self, state_weights, control_weights, target,
                     current_state, future_states, future_actions):
        """Compute MPC-based loss with multiple targets"""
        try:
            # Use MPC solver with predicted parameters
            mpc_states, mpc_controls = self.mpc_solver.solve(
                current_state,
                state_weights,
                control_weights,
                target
            )
            
            # Downsample ground truth for comparison
            downsampled_indices = torch.arange(
                0, 
                min(future_states.size(1), mpc_states.size(1) * self.downsample_factor), 
                self.downsample_factor,
                device=self.device
            )
            
            # Ensure we don't exceed MPC prediction length
            downsampled_indices = downsampled_indices[:mpc_states.size(1)]
            
            downsampled_future_states = future_states[:, downsampled_indices, :]
            downsampled_future_actions = future_actions[:, downsampled_indices, :]
            
            # Truncate MPC predictions to match downsampled ground truth length
            mpc_states_truncated = mpc_states[:, :len(downsampled_indices), :]
            mpc_controls_truncated = mpc_controls[:, :len(downsampled_indices), :]
            
            # State trajectory loss (position and velocity)
            position_loss = self.mse_loss(
                mpc_states_truncated[:, :, :2],  # x, y
                downsampled_future_states[:, :, :2]
            )
            
            heading_loss = self.mse_loss(
                mpc_states_truncated[:, :, 2],  # yaw (index 2)
                downsampled_future_states[:, :, 2]
            )
            
            velocity_loss = self.mse_loss(
                mpc_states_truncated[:, :, 3],  # speed (index 3)
                downsampled_future_states[:, :, 3]
            )
            
            trajectory_loss = position_loss + 0.1 * heading_loss + 0.1 * velocity_loss
            
            # Control loss
            control_loss = self.mse_loss(
                mpc_controls_truncated,
                downsampled_future_actions
            )

        except Exception as e:
            # If MPC fails, use only regularization loss
            print(f"WARNING: MPC failed in loss computation: {e}")
            trajectory_loss = torch.tensor(10.0, device=self.device, requires_grad=True)
            control_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        # Parameter regularization
        state_weight_reg = torch.mean((state_weights - 0.0).pow(2))
        control_weight_reg = torch.mean((control_weights - 0.0).pow(2))
        target_reg = torch.mean(target.pow(2)) * 0.01
        
        regularization = state_weight_reg + control_weight_reg + target_reg
        
        # Total loss
        total_loss = trajectory_loss + 0.0 * control_loss + 0.01 * regularization

        return {
            'total': total_loss,
            'trajectory': trajectory_loss,
            'control': control_loss,
            'regularization': regularization
        }
    
    def evaluate(self, val_loader):
        """Evaluate model"""
        self.model.eval()
        val_loss = 0.0
        val_trajectory_loss = 0.0
        val_control_loss = 0.0
        val_reg_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_seq = batch['input_seq'].to(self.device)
                current_state = batch['current_state'].to(self.device)
                future_states = batch['future_states'].to(self.device)
                future_actions = batch['future_actions'].to(self.device)
                
                state_weights, control_weights, target = self.model(input_seq)
                
                loss_dict = self.compute_loss(
                    state_weights, control_weights, target,
                    current_state, future_states, future_actions
                )
                
                val_loss += loss_dict['total'].item()
                val_trajectory_loss += loss_dict['trajectory'].item()
                val_control_loss += loss_dict['control'].item()
                val_reg_loss += loss_dict['regularization'].item()
        
        return (val_loss / len(val_loader), 
                val_trajectory_loss / len(val_loader),
                val_control_loss / len(val_loader),
                val_reg_loss / len(val_loader))

class VehicleLSTMMPCPredictor:
    """Online predictor for vehicle control with multiple targets support"""
    
    def __init__(self, model_path, device='cpu', horizon=100):
        self.device = device
        self.horizon = horizon
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get parameters
        self.downsample_factor = checkpoint.get('downsample_factor', 1)
        self.num_targets = checkpoint.get('num_targets', 1)
        
        # Rebuild model
        config = checkpoint['model_config']
        self.model = VehicleLSTMMPC(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            state_dim=config['state_dim'],
            control_dim=config['control_dim'],
            num_targets=config.get('num_targets', 1)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # MPC solver with multiple targets
        self.mpc_solver = VehicleMPCSolver(
            dt=0.05, 
            horizon=horizon, 
            device=device,
            downsample_factor=self.downsample_factor,
            num_targets=self.num_targets
        )
        
        print(f"Model loaded from {model_path}")
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")
        print(f"Downsample factor: {self.downsample_factor}")
        print(f"Number of targets: {self.num_targets}")
    
    def predict_control(self, hist_states, hist_actions, current_state):
        """
        Predict MPC parameters and solve for optimal control with multiple targets
        """
        
        # Prepare input
        input_seq = np.concatenate([hist_states, hist_actions], axis=1)
        input_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
        current_state = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Predict MPC parameters
            state_weights, control_weights, target = self.model(input_seq)
            
            # Solve MPC
            predicted_states, optimal_controls = self.mpc_solver.solve(
                current_state, state_weights, control_weights, target
            )
        
        # Convert to numpy
        optimal_controls = optimal_controls.cpu().numpy().squeeze()
        predicted_states = predicted_states.cpu().numpy().squeeze()
        
        # Parse multiple targets for visualization
        target_np = target.cpu().numpy().squeeze()
        targets_parsed = []
        
        for i in range(self.num_targets):
            start_idx = i * 6
            target_state = target_np[start_idx:start_idx+4]
            target_control = target_np[start_idx+4:start_idx+6]
            targets_parsed.append({
                'state': target_state,
                'control': target_control
            })
        
        mpc_params = {
            'state_weights': state_weights.cpu().numpy().squeeze(),
            'control_weights': control_weights.cpu().numpy().squeeze(),
            'targets': targets_parsed,
            'num_targets': self.num_targets,
            'downsample_factor': self.downsample_factor,
            'mpc_dt': self.mpc_solver.mpc_dt,
            'mpc_horizon': self.mpc_solver.mpc_horizon
        }
        
        return optimal_controls, predicted_states, mpc_params
    
    def visualize_prediction(self, hist_states, predicted_states, true_states=None, mpc_params=None):
        """Visualize prediction results with multiple targets"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Trajectory plot
        ax = axes[0, 0]
        ax.plot(hist_states[:, 0], hist_states[:, 1], 'b-', label='History', linewidth=2)
        ax.plot(predicted_states[:, 0], predicted_states[:, 1], 'r--', 
                label=f'MPC Predicted (dt={mpc_params["mpc_dt"]:.2f}s)', linewidth=2)
        if true_states is not None:
            ax.plot(true_states[:, 0], true_states[:, 1], 'g-', label='Ground Truth', linewidth=2)
        
        # Plot multiple targets
        if mpc_params is not None and 'targets' in mpc_params:
            colors = ['red', 'orange', 'purple', 'brown', 'pink']
            for i, target_info in enumerate(mpc_params['targets']):
                target_state = target_info['state']
                color = colors[i % len(colors)]
                ax.plot(target_state[0], target_state[1], '*', markersize=15, 
                       color=color, label=f'Target {i+1}')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Vehicle Trajectory (LSTM-MPC, DS={mpc_params.get("downsample_factor", 1)}, Targets={mpc_params.get("num_targets", 1)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Speed profile
        ax = axes[0, 1]
        time_hist = np.arange(len(hist_states)) * 0.05
        time_pred = np.arange(len(predicted_states)) * mpc_params['mpc_dt']
        time_true = np.arange(len(true_states)) * 0.05 if true_states is not None else None
        
        ax.plot(time_hist, hist_states[:, 3] * 3.6, 'b-', label='History', linewidth=2)
        ax.plot(time_pred, predicted_states[:, 3] * 3.6, 'r--', label='MPC Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(time_true, true_states[:, 3] * 3.6, 'g-', label='Ground Truth', linewidth=2)
        
        # Plot target speeds
        if mpc_params is not None and 'targets' in mpc_params:
            colors = ['red', 'orange', 'purple', 'brown', 'pink']
            for i, target_info in enumerate(mpc_params['targets']):
                target_speed = target_info['state'][3] * 3.6
                color = colors[i % len(colors)]
                ax.axhline(y=target_speed, color=color, linestyle=':', 
                          alpha=0.7, label=f'Target Speed {i+1}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Speed Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Yaw angle
        ax = axes[1, 0]
        ax.plot(time_hist, np.rad2deg(hist_states[:, 2]), 'b-', label='History', linewidth=2)
        ax.plot(time_pred, np.rad2deg(predicted_states[:, 2]), 'r--', label='MPC Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(time_true, np.rad2deg(true_states[:, 2]), 'g-', label='Ground Truth', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw Angle (degrees)')
        ax.set_title('Yaw Angle')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MPC parameters
        if mpc_params is not None:
            ax = axes[1, 1]
            
            weights_labels = ['X', 'Y', 'Yaw', 'Speed', 'Accel', 'Steer']
            weights_values = list(mpc_params['state_weights']) + list(mpc_params['control_weights'])
            
            colors = ['blue'] * 4 + ['orange'] * 2
            bars = ax.bar(weights_labels, weights_values, color=colors, alpha=0.7)
            
            ax.set_ylabel('Weight Value')
            ax.set_title(f'MPC Weights (DS={mpc_params["downsample_factor"]}, Targets={mpc_params["num_targets"]})')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# Keep all other functions unchanged (calculate_temporal_errors, evaluate_full_dataset, etc.)
def calculate_temporal_errors(pred_states, true_states, time_step=0.05, downsample_factor=1):
    """Calculate average errors at different future time points"""
    time_points = [1, 2, 3, 4, 5]  # seconds
    time_indices = [int(t / time_step) - 1 for t in time_points]  # convert to indices
    
    error_results = {}
    
    for i, (t, idx) in enumerate(zip(time_points, time_indices)):
        if idx >= pred_states.shape[1]:
            continue
            
        # Position error
        position_error = np.sqrt(
            (pred_states[:, idx, 0] - true_states[:, idx, 0])**2 + 
            (pred_states[:, idx, 1] - true_states[:, idx, 1])**2
        )
        
        # Yaw error (index 2 for states: [x, y, yaw, speed])
        yaw_error = np.abs(pred_states[:, idx, 2] - true_states[:, idx, 2])
        yaw_error = np.minimum(yaw_error, 2*np.pi - yaw_error)
        yaw_error = np.rad2deg(yaw_error)
        
        # Speed error (index 3 for states: [x, y, yaw, speed])
        speed_error = np.abs(pred_states[:, idx, 3] - true_states[:, idx, 3]) * 3.6
        
        error_results[f'{t}s'] = {
            'position_error_mean': np.mean(position_error),
            'position_error_std': np.std(position_error),
            'position_error_max': np.max(position_error),
            'speed_error_mean': np.mean(speed_error),
            'speed_error_std': np.std(speed_error),
            'speed_error_max': np.max(speed_error),
            'yaw_error_mean': np.mean(yaw_error),
            'yaw_error_std': np.std(yaw_error),
            'yaw_error_max': np.max(yaw_error),
            'sample_count': len(position_error)
        }
    
    return error_results

def evaluate_full_dataset(predictor, test_dataset_full, device='cpu'):
    """Evaluate entire test dataset and compute temporal error statistics"""
    print("Starting full dataset evaluation...")
    print(f"Total test sequences: {len(test_dataset_full['training_sequences'])}")
    print(f"Using downsample factor: {predictor.downsample_factor}")
    print(f"Using number of targets: {predictor.num_targets}")
    
    all_pred_states = []
    all_true_states = []
    
    for i, seq in enumerate(tqdm(test_dataset_full['training_sequences'], desc="Processing sequences")):
        try:
            # Convert controls to [a, delta_f] format
            max_accel = 10.0
            max_steer = np.deg2rad(70)
            hist_controls = seq['hist_controls']
            accelerations = (hist_controls[:, 0] - hist_controls[:, 1]) * max_accel
            steer_angles = hist_controls[:, 2] * max_steer
            hist_actions = np.column_stack([accelerations, steer_angles])
            
            # Predict using LSTM-MPC
            optimal_controls, predicted_states, mpc_params = predictor.predict_control(
                seq['hist_states'], hist_actions, seq['current_state']
            )
            
            # Downsample ground truth to match MPC predictions
            ds_indices = np.arange(0, len(seq['future_states']), predictor.downsample_factor)
            downsampled_future_states = seq['future_states'][ds_indices[:len(predicted_states)]]
            
            all_pred_states.append(predicted_states)
            all_true_states.append(downsampled_future_states)
            
        except Exception as e:
            print(f"Error processing sequence {i}: {e}")
            continue
    
    if not all_pred_states:
        print("No valid predictions generated!")
        return None
    
    # Convert to numpy arrays and ensure same shape
    min_length = min(len(pred) for pred in all_pred_states)
    all_pred_states = np.array([pred[:min_length] for pred in all_pred_states])
    all_true_states = np.array([true[:min_length] for true in all_true_states])
    
    print(f"Successfully processed {len(all_pred_states)} sequences")
    print(f"Prediction horizon: {min_length} steps (MPC dt = {predictor.mpc_solver.mpc_dt}s)")
    
    temporal_errors = calculate_temporal_errors(
        all_pred_states, all_true_states, 
        time_step=predictor.mpc_solver.mpc_dt,
        downsample_factor=1  # Already handled in MPC solver
    )
    
    return temporal_errors

def print_error_statistics(error_results, downsample_factor=1, num_targets=1):
    """Print error statistics results"""
    print("\n" + "="*80)
    print(f"TEMPORAL ERROR ANALYSIS RESULTS (LSTM-MPC, DS={downsample_factor}, Targets={num_targets})")
    print("="*80)
    
    print(f"{'Time':<6} {'Pos.Error(m)':<15} {'Speed Error(km/h)':<18} {'Yaw Error(°)':<15} {'Samples':<8}")
    print(f"{'Point':<6} {'Mean±Std (Max)':<15} {'Mean±Std (Max)':<18} {'Mean±Std (Max)':<15} {'Count':<8}")
    print("-"*80)
    
    for time_point, errors in error_results.items():
        pos_mean = errors['position_error_mean']
        pos_std = errors['position_error_std']
        pos_max = errors['position_error_max']
        
        speed_mean = errors['speed_error_mean']
        speed_std = errors['speed_error_std']
        speed_max = errors['speed_error_max']
        
        yaw_mean = errors['yaw_error_mean']
        yaw_std = errors['yaw_error_std']
        yaw_max = errors['yaw_error_max']
        
        sample_count = errors['sample_count']
        
        print(f"{time_point:<6} "
              f"{pos_mean:.3f}±{pos_std:.3f} ({pos_max:.3f}){'':<1} "
              f"{speed_mean:.2f}±{speed_std:.2f} ({speed_max:.2f}){'':<4} "
              f"{yaw_mean:.2f}±{yaw_std:.2f} ({yaw_max:.2f}){'':<2} "
              f"{sample_count:<8}")
    
    print("="*80)

def plot_error_trends(error_results, downsample_factor=1, num_targets=1):
    """Plot error trends over time"""
    time_points = [1, 2, 3, 4, 5]
    pos_means = [error_results[f'{t}s']['position_error_mean'] for t in time_points]
    speed_means = [error_results[f'{t}s']['speed_error_mean'] for t in time_points]
    yaw_means = [error_results[f'{t}s']['yaw_error_mean'] for t in time_points]
    
    pos_stds = [error_results[f'{t}s']['position_error_std'] for t in time_points]
    speed_stds = [error_results[f'{t}s']['speed_error_std'] for t in time_points]
    yaw_stds = [error_results[f'{t}s']['yaw_error_std'] for t in time_points]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Position error
    ax = axes[0]
    ax.errorbar(time_points, pos_means, yerr=pos_stds, marker='o', capsize=5, capthick=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title(f'Position Error vs Time (LSTM-MPC, DS={downsample_factor}, Targets={num_targets})')
    ax.grid(True, alpha=0.3)
    
    # Speed error
    ax = axes[1]
    ax.errorbar(time_points, speed_means, yerr=speed_stds, marker='s', capsize=5, capthick=2, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed Error (km/h)')
    ax.set_title(f'Speed Error vs Time (LSTM-MPC, DS={downsample_factor}, Targets={num_targets})')
    ax.grid(True, alpha=0.3)
    
    # Yaw error
    ax = axes[2]
    ax.errorbar(time_points, yaw_means, yerr=yaw_stds, marker='^', capsize=5, capthick=2, color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw Error (degrees)')
    ax.set_title(f'Yaw Error vs Time (LSTM-MPC, DS={downsample_factor}, Targets={num_targets})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def load_dataset_from_folder(data_folder, dataset_name):
    """Load dataset from folder"""
    dataset_path = os.path.join(data_folder, dataset_name)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset loaded: {dataset_path}")
    print(f"  Sequences: {len(dataset['training_sequences'])}")
    if 'config' in dataset:
        config = dataset['config']
        print(f"  Configuration:")
        print(f"    - Coordinate system: {config.get('coordinate_system', 'unknown')}")
        print(f"    - History steps: {config.get('history_steps', 'unknown')}")
        print(f"    - Predict steps: {config.get('predict_steps', 'unknown')}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Vehicle LSTM-MPC Training and Prediction with Multiple Targets')
    
    # Data parameters
    parser.add_argument('--data_folder', default='vehicle_datasets', help='Folder containing datasets')
    parser.add_argument('--train_file', default='vehicle_train_dataset.pkl', help='Training dataset')
    parser.add_argument('--test_file', default='vehicle_test_dataset.pkl', help='Test dataset')
    
    # Model parameters
    parser.add_argument('--save_dir', default='models', help='Model save directory')
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    
    # MPC parameters
    parser.add_argument('--downsample_factor', type=int, default=10, 
                       help='Downsample factor for MPC (e.g., 2 means MPC dt = 0.1s)')
    parser.add_argument('--num_targets', type=int, default=1, 
                       help='Number of target points to use across MPC horizon')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Run mode
    parser.add_argument('--mode', choices=['train', 'test', 'evaluate'], default='train',
                       help='Mode: train, test, or evaluate')
    parser.add_argument('--save_results', action='store_true', help='Save error results to file')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"=== Vehicle LSTM-MPC Controller with Multiple Targets ===")
    print(f"Mode: {args.mode}")
    print(f"Downsample Factor: {args.downsample_factor}")
    print(f"Number of Targets: {args.num_targets}")
    print(f"MPC dt: {0.05 * args.downsample_factor:.2f}s")
    print(f"MPC horizon: {100 // args.downsample_factor} steps")
    
    if args.mode == 'train':
        # Load datasets
        train_dataset_full = load_dataset_from_folder(args.data_folder, args.train_file)
        val_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
        
        # Create datasets
        train_dataset = VehicleLSTMMPCDataset(train_dataset_full['training_sequences'])
        val_dataset = VehicleLSTMMPCDataset(val_dataset_full['training_sequences'])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Create model with multiple targets
        model = VehicleLSTMMPC(
            input_dim=6,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            state_dim=4,
            control_dim=2,
            num_targets=args.num_targets
        )
        
        # Create trainer with multiple targets
        trainer = VehicleLSTMMPCTrainer(model, 
                                      downsample_factor=args.downsample_factor,
                                      num_targets=args.num_targets)
        
        # Train model
        trainer.train_model(
            train_loader, val_loader, 
            epochs=args.epochs, 
            lr=args.lr, 
            patience=args.patience,
            save_dir=args.save_dir
        )
        
    elif args.mode == 'test':
        # Load test dataset
        test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
        
        # Load model
        model_path = os.path.join(args.save_dir, f'best_vehicle_lstm_mpc_ds{args.downsample_factor}_targets{args.num_targets}.pth')
        predictor = VehicleLSTMMPCPredictor(model_path)
        
        # Test samples
        for i in range(6800, 10000, 10):
            seq = test_dataset_full['training_sequences'][i]
            
            # Prepare data
            hist_states = seq['hist_states']
            current_state = seq['current_state']
            future_states = seq['future_states']
            
            # Convert controls
            max_accel = 10.0
            max_steer = np.deg2rad(70)
            hist_controls = seq['hist_controls']
            accelerations = (hist_controls[:, 0] - hist_controls[:, 1]) * max_accel
            steer_angles = hist_controls[:, 2] * max_steer
            hist_actions = np.column_stack([accelerations, steer_angles])
            
            # Predict
            optimal_controls, predicted_states, mpc_params = predictor.predict_control(
                hist_states, hist_actions, current_state
            )
            
            print(f"\nTest Sample {i+1}:")
            print(f"  Current state: {current_state}")
            print(f"  Number of targets: {mpc_params['num_targets']}")
            for j, target_info in enumerate(mpc_params['targets']):
                print(f"  Target {j+1} state: {target_info['state']}")
                print(f"  Target {j+1} control: {target_info['control']}")
            print(f"  State weights: {mpc_params['state_weights']}")
            print(f"  Control weights: {mpc_params['control_weights']}")
            print(f"  MPC dt: {mpc_params['mpc_dt']:.2f}s")
            print(f"  MPC horizon: {mpc_params['mpc_horizon']} steps")
            
            # Visualize
            # For comparison, downsample ground truth
            ds_indices = np.arange(0, len(future_states), predictor.downsample_factor)
            downsampled_future_states = future_states[ds_indices[:len(predicted_states)]]
            
            predictor.visualize_prediction(
                hist_states, 
                predicted_states, 
                downsampled_future_states,
                mpc_params
            )
    
    elif args.mode == 'evaluate':
        print("Loading test dataset and model...")
        
        # Load test dataset
        test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
        
        # Load model
        model_path = os.path.join(args.save_dir, f'best_vehicle_lstm_mpc_ds{args.downsample_factor}_targets{args.num_targets}.pth')
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first.")
            return
            
        predictor = VehicleLSTMMPCPredictor(model_path)
        
        # Evaluate entire test dataset
        error_results = evaluate_full_dataset(predictor, test_dataset_full)
        
        if error_results:
            # Print results
            print_error_statistics(error_results, 
                                 downsample_factor=predictor.downsample_factor,
                                 num_targets=predictor.num_targets)
            
            # Plot trends
            plot_error_trends(error_results, 
                            downsample_factor=predictor.downsample_factor,
                            num_targets=predictor.num_targets)
            
            # Save results (optional)
            if args.save_results:
                results_path = os.path.join(args.save_dir, f'lstm_mpc_temporal_error_results_ds{predictor.downsample_factor}_targets{predictor.num_targets}.pkl')
                with open(results_path, 'wb') as f:
                    pickle.dump({
                        'error_results': error_results,
                        'downsample_factor': predictor.downsample_factor,
                        'num_targets': predictor.num_targets,
                        'mpc_dt': predictor.mpc_solver.mpc_dt,
                        'mpc_horizon': predictor.mpc_solver.mpc_horizon
                    }, f)
                print(f"Results saved to: {results_path}")
        else:
            print("Evaluation failed!")

if __name__ == "__main__":
    main()