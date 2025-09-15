"""
Vehicle LSTM-MPC Controller for CARLA with Multiple Target Points and Cost Probability Heatmap Visualization
Modified to support multiple target points across the MPC horizon with enhanced probability visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib.colorbar import ColorbarBase
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
    """LSTM network for predicting MPC parameters with decoder-based target prediction"""
    
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
        
        # NEW: Decoder LSTM for target prediction
        self.target_decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                          batch_first=True)
        
        # NEW: Output heads for decoder (now single target per timestep)
        self.target_state_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)  # Output single target state per timestep
        )
        
        self.target_control_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, control_dim)  # Output single target control per timestep
        )
    
    def apply_control_bounds(self, control_raw):
        """Apply bounds to control outputs using tanh activation"""
        # Now control_raw is (batch_size, num_targets, control_dim)
        # Split the control outputs
        a_raw = control_raw[:, :, 0:1]  # acceleration
        delta_f_raw = control_raw[:, :, 1:2]  # steering angle
        
        # Apply bounds using tanh
        a_bounded = torch.tanh(a_raw) * self.a_bound
        delta_f_bounded = torch.tanh(delta_f_raw) * self.delta_f_bound
        
        # Concatenate back
        control_bounded = torch.cat([a_bounded, delta_f_bounded], dim=2)
        
        return control_bounded
    
    def apply_state_bounds(self, state_raw):
        """Apply bounds to state outputs, specifically for yaw angle"""
        # Now state_raw is (batch_size, num_targets, state_dim)
        # Split the state outputs: [x, y, yaw, speed]
        x = state_raw[:, :, 0:1]      # x position (no bounds)
        y = state_raw[:, :, 1:2]      # y position (no bounds) 
        yaw_raw = state_raw[:, :, 2:3]    # yaw angle (需要限制)
        speed_raw = state_raw[:, :, 3:4]  # speed (限制为正值)
        
        # Apply yaw angle bounds: limit to [-π, π]
        yaw_bounded = torch.atan2(torch.sin(yaw_raw), torch.cos(yaw_raw))
        
        # Apply speed bounds: ensure non-negative speed
        speed_bounded = torch.relu(speed_raw)
        
        # Concatenate back
        state_bounded = torch.cat([x, y, yaw_bounded, speed_bounded], dim=2)
        
        return state_bounded
    
    def decode_targets(self, context):
        """Decode targets using LSTM decoder"""
        batch_size = context.size(0)
        
        # Expand context to all target timesteps
        decoder_input = context.unsqueeze(1).repeat(1, self.num_targets, 1)  # (batch, num_targets, hidden_dim)
        
        # Pass through decoder LSTM
        decoder_out, _ = self.target_decoder_lstm(decoder_input)  # (batch, num_targets, hidden_dim)
        
        # Generate target states and controls for each timestep
        target_states_raw = self.target_state_fc(decoder_out)  # (batch, num_targets, state_dim)
        target_controls_raw = self.target_control_fc(decoder_out)  # (batch, num_targets, control_dim)
        
        # Apply bounds
        target_states = self.apply_state_bounds(target_states_raw)
        target_controls = self.apply_control_bounds(target_controls_raw)
        
        return target_states, target_controls
    
    def forward(self, x):
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Predict MPC parameters
        state_weights = self.state_weight_head(last_hidden)  # (batch_size, 4)
        control_weights = self.control_weight_head(last_hidden) # (batch_size, 2)

        # NEW: Decode targets using LSTM decoder
        target_states, target_controls = self.decode_targets(last_hidden)  # (batch, num_targets, state_dim), (batch, num_targets, control_dim)
        
        # Combine state and control targets
        target = torch.cat([target_states, target_controls], dim=2)  # (batch_size, num_targets, 6)
        
        # Flatten for compatibility with existing code
        target_flat = target.view(target.size(0), -1)  # (batch_size, num_targets * 6)
        
        return state_weights, control_weights, target_flat

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
            lqr_iter=30,
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
                self.downsample_factor - 1, 
                future_states.size(1),
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
        state_weight_reg = torch.mean((state_weights - 1.0).pow(2))
        control_weight_reg = torch.mean((control_weights - 0.1).pow(2))
        target_reg = torch.mean(target.pow(2)) * 0.0
        
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
    """Online predictor for vehicle control with multiple targets support and cost probability heatmap visualization"""
    
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
    
    def calculate_mpc_costs(self, predicted_states, optimal_controls, mpc_params):
        """
        计算MPC轨迹上每个点的cost详细信息
        
        Args:
            predicted_states: MPC预测的状态轨迹 (T, 4)
            optimal_controls: MPC预测的控制轨迹 (T, 2) 
            mpc_params: MPC参数字典
        
        Returns:
            cost_breakdown: 每个时间步的cost分解
        """
        state_weights = mpc_params['state_weights']
        control_weights = mpc_params['control_weights']
        targets = mpc_params['targets']
        
        T = len(predicted_states)
        cost_breakdown = {
            'total_cost': np.zeros(T),
            'state_cost': np.zeros(T),
            'control_cost': np.zeros(T),
            'position_cost': np.zeros(T),
            'yaw_cost': np.zeros(T), 
            'speed_cost': np.zeros(T),
            'accel_cost': np.zeros(T),
            'steer_cost': np.zeros(T)
        }
        
        # 为每个时间步分配目标（简化：使用第一个目标）
        if targets:
            target_state = targets[0]['state']  # [x, y, yaw, speed]
            target_control = targets[0]['control']  # [a, delta_f]
        else:
            target_state = np.zeros(4)
            target_control = np.zeros(2)
        
        for t in range(T):
            # 状态误差
            state_error = predicted_states[t] - target_state
            control_error = optimal_controls[t] - target_control
            
            # 各分量的cost
            position_cost = state_weights[0] * (state_error[0]**2) + state_weights[1] * (state_error[1]**2)
            yaw_cost = state_weights[2] * (state_error[2]**2)
            speed_cost = state_weights[3] * (state_error[3]**2)
            
            accel_cost = control_weights[0] * (control_error[0]**2)
            steer_cost = control_weights[1] * (control_error[1]**2)
            
            # 存储结果
            cost_breakdown['position_cost'][t] = position_cost
            cost_breakdown['yaw_cost'][t] = yaw_cost
            cost_breakdown['speed_cost'][t] = speed_cost
            cost_breakdown['accel_cost'][t] = accel_cost
            cost_breakdown['steer_cost'][t] = steer_cost
            
            cost_breakdown['state_cost'][t] = position_cost + yaw_cost + speed_cost
            cost_breakdown['control_cost'][t] = accel_cost + steer_cost
            cost_breakdown['total_cost'][t] = cost_breakdown['state_cost'][t] + cost_breakdown['control_cost'][t]
        
        return cost_breakdown
    
    def _plot_cost_probability_heatmap(self, ax, predicted_states, cost_breakdown):
        """
        绘制基于cost的概率热力图
        cost越低的位置，概率权重越高（更"可能"被选中）
        """
        x = predicted_states[:, 0]
        y = predicted_states[:, 1] 
        cost = cost_breakdown['total_cost']
        
        # 将cost转换为概率权重 (cost越低，权重越高)
        # 使用softmax的逆向思维：cost越高，概率越低
        weights = np.exp(-cost / (np.std(cost) + 1e-8))  # 负指数变换
        weights = weights / np.sum(weights)  # 归一化为概率
        
        # 创建更精细的网格
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # 扩展边界以便更好地显示
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # 创建网格
        grid_size = 100
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        xi_mesh, yi_mesh = np.meshgrid(xi, yi)
        
        # 计算每个网格点的概率密度
        # 使用高斯核密度估计的思想
        sigma = min(x_range, y_range) * 0.05  # 核函数的标准差
        probability_map = np.zeros((grid_size, grid_size))
        
        for i in range(len(x)):
            # 计算每个轨迹点对网格的影响
            distances_sq = (xi_mesh - x[i])**2 + (yi_mesh - y[i])**2
            influence = weights[i] * np.exp(-distances_sq / (2 * sigma**2))
            probability_map += influence
        
        # 归一化概率图
        if np.sum(probability_map) > 0:
            probability_map = probability_map / np.sum(probability_map)
        
        # 绘制概率热力图
        levels = np.linspace(0, np.max(probability_map), 20)
        contour = ax.contourf(xi_mesh, yi_mesh, probability_map, levels=levels, 
                             cmap='YlOrRd', alpha=0.8)
        
        # 添加等概率线
        contour_lines = ax.contour(xi_mesh, yi_mesh, probability_map, levels=8, 
                                  colors='black', alpha=0.3, linewidths=0.5)
        
        # 绘制轨迹点，大小表示概率权重
        scatter = ax.scatter(x, y, c=weights, s=weights*1000, cmap='YlOrRd', 
                            edgecolors='black', linewidth=0.5, alpha=0.7)
        
        # 绘制轨迹线
        ax.plot(x, y, 'k--', alpha=0.6, linewidth=1, label='Trajectory')
        
        # 添加colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Probability Density', rotation=270, labelpad=15)
        
        # 标注高概率区域
        max_prob_idx = np.unravel_index(np.argmax(probability_map), probability_map.shape)
        max_prob_x = xi_mesh[max_prob_idx]
        max_prob_y = yi_mesh[max_prob_idx]
        ax.plot(max_prob_x, max_prob_y, 'w*', markersize=15, 
               markeredgecolor='black', markeredgewidth=1, label='Peak Probability')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Cost-based Trajectory Probability Heatmap')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    def visualize_prediction_with_heatmap(self, hist_states, predicted_states, true_states=None, mpc_params=None):
        """
        增强版可视化：包含cost概率热力图
        """
        # 计算cost详情
        dt = mpc_params.get('mpc_dt', 0.05)
        
        # 简化：从状态变化推算控制量
        controls_approx = np.zeros((len(predicted_states), 2))
        for i in range(len(predicted_states)-1):
            # 加速度近似
            controls_approx[i, 0] = (predicted_states[i+1, 3] - predicted_states[i, 3]) / dt
            # 转向角近似（从偏航角变化率推算）
            if predicted_states[i, 3] > 0.1:
                controls_approx[i, 1] = (predicted_states[i+1, 2] - predicted_states[i, 2]) / dt * 3.45 / predicted_states[i, 3]
            else:
                controls_approx[i, 1] = 0
        
        cost_breakdown = self.calculate_mpc_costs(predicted_states, controls_approx, mpc_params)
        
        # 创建图形
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 主轨迹图 - 带总cost热力图
        ax1 = plt.subplot(3, 4, (1, 2))
        self._plot_trajectory_heatmap(ax1, hist_states, predicted_states, true_states, 
                                     cost_breakdown['total_cost'], 'Total Cost Heatmap', mpc_params)
        
        # 2. 位置cost热力图
        ax2 = plt.subplot(3, 4, 3)
        self._plot_trajectory_heatmap(ax2, hist_states, predicted_states, true_states,
                                     cost_breakdown['position_cost'], 'Position Cost', mpc_params, compact=True)
        
        # 3. 速度cost热力图  
        ax3 = plt.subplot(3, 4, 4)
        self._plot_trajectory_heatmap(ax3, hist_states, predicted_states, true_states,
                                     cost_breakdown['speed_cost'], 'Speed Cost', mpc_params, compact=True)
        
        # 4. Cost随时间变化
        ax4 = plt.subplot(3, 4, (5, 6))
        self._plot_cost_timeline(ax4, cost_breakdown, mpc_params)
        
        # 5. 控制量cost热力图
        ax5 = plt.subplot(3, 4, 7)
        self._plot_trajectory_heatmap(ax5, hist_states, predicted_states, true_states,
                                     cost_breakdown['control_cost'], 'Control Cost', mpc_params, compact=True)
        
        # 6. 偏航角cost热力图
        ax6 = plt.subplot(3, 4, 8)
        self._plot_trajectory_heatmap(ax6, hist_states, predicted_states, true_states,
                                     cost_breakdown['yaw_cost'], 'Yaw Cost', mpc_params, compact=True)
        
        # 7. Cost分量饼图
        ax7 = plt.subplot(3, 4, 9)
        self._plot_cost_breakdown_pie(ax7, cost_breakdown)
        
        # 8. MPC权重可视化
        ax8 = plt.subplot(3, 4, 10)
        self._plot_mpc_weights(ax8, mpc_params)
        
        # 9. Cost概率分布热力图（固定使用probability样式）
        ax9 = plt.subplot(3, 4, 11)
        self._plot_cost_probability_heatmap(ax9, predicted_states, cost_breakdown)
        
        # 10. Cost统计信息
        ax10 = plt.subplot(3, 4, 12)
        self._plot_cost_statistics(ax10, cost_breakdown)
        
        plt.tight_layout()
        plt.suptitle(f'MPC Cost Analysis with Probability Heatmap (DS={mpc_params.get("downsample_factor", 1)}, Targets={mpc_params.get("num_targets", 1)})', 
                     fontsize=16, y=0.98)
        plt.show()
    
    def _plot_trajectory_heatmap(self, ax, hist_states, pred_states, true_states, cost_values, title, mpc_params, compact=False):
        """绘制轨迹热力图"""
        # 历史轨迹
        ax.plot(hist_states[:, 0], hist_states[:, 1], 'b-', linewidth=3, label='History', alpha=0.8)
        
        # 真实轨迹
        if true_states is not None:
            ax.plot(true_states[:, 0], true_states[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
        
        # 预测轨迹 - 用cost值着色
        if len(cost_values) > 0:
            # 归一化cost值到[0,1]
            cost_norm = (cost_values - np.min(cost_values)) / (np.ptp(cost_values) + 1e-8)
            
            # 创建热力图颜色映射
            cmap = plt.cm.Reds  # 红色表示高cost
            
            # 绘制轨迹点，颜色表示cost
            scatter = ax.scatter(pred_states[:, 0], pred_states[:, 1], 
                               c=cost_norm, cmap=cmap, s=50, alpha=0.8, 
                               edgecolors='darkred', linewidth=0.5)
            
            # 连接线
            ax.plot(pred_states[:, 0], pred_states[:, 1], 'r--', linewidth=1.5, alpha=0.6)
            
            # 添加颜色条
            if not compact:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
                cbar.set_label('Cost Value', rotation=270, labelpad=15)
        
        # 绘制目标点
        if mpc_params and 'targets' in mpc_params:
            colors = ['red', 'orange', 'purple', 'brown', 'pink']
            for i, target_info in enumerate(mpc_params['targets']):
                target_state = target_info['state']
                color = colors[i % len(colors)]
                ax.plot(target_state[0], target_state[1], '*', markersize=12 if compact else 15, 
                       color=color, label=f'Target {i+1}' if not compact else None,
                       markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        if not compact:
            ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    def _plot_cost_timeline(self, ax, cost_breakdown, mpc_params):
        """绘制cost随时间变化"""
        mpc_dt = mpc_params.get('mpc_dt', 0.05)
        time = np.arange(len(cost_breakdown['total_cost'])) * mpc_dt
        
        ax.plot(time, cost_breakdown['total_cost'], 'k-', linewidth=3, label='Total Cost')
        ax.plot(time, cost_breakdown['state_cost'], 'b--', linewidth=2, label='State Cost')  
        ax.plot(time, cost_breakdown['control_cost'], 'r--', linewidth=2, label='Control Cost')
        ax.fill_between(time, cost_breakdown['position_cost'], alpha=0.3, label='Position')
        ax.fill_between(time, cost_breakdown['speed_cost'], alpha=0.3, label='Speed')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cost Value')
        ax.set_title('Cost Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_cost_breakdown_pie(self, ax, cost_breakdown):
        """绘制cost分量饼图"""
        # 计算各分量的平均cost
        avg_costs = {
            'Position': np.mean(cost_breakdown['position_cost']),
            'Yaw': np.mean(cost_breakdown['yaw_cost']),
            'Speed': np.mean(cost_breakdown['speed_cost']),
            'Acceleration': np.mean(cost_breakdown['accel_cost']),
            'Steering': np.mean(cost_breakdown['steer_cost'])
        }
        
        # 过滤掉接近零的值
        filtered_costs = {k: v for k, v in avg_costs.items() if v > 1e-6}
        
        if filtered_costs:
            labels = list(filtered_costs.keys())
            sizes = list(filtered_costs.values())
            colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold', 'plum']
            
            ax.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
            ax.set_title('Average Cost Breakdown')
        else:
            ax.text(0.5, 0.5, 'No significant\ncost variation', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cost Breakdown')

    def _plot_mpc_weights(self, ax, mpc_params):
        """绘制MPC权重"""
        if mpc_params:
            weights_labels = ['X', 'Y', 'Yaw', 'Speed', 'Accel', 'Steer']
            weights_values = list(mpc_params['state_weights']) + list(mpc_params['control_weights'])
            
            colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'orange', 'orange']
            bars = ax.bar(weights_labels, weights_values, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_ylabel('Weight Value')
            ax.set_title('MPC Weights')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    def _plot_cost_statistics(self, ax, cost_breakdown):
        """绘制cost统计信息"""
        ax.axis('off')
        
        # 计算统计量
        total_cost = np.sum(cost_breakdown['total_cost'])
        max_cost = np.max(cost_breakdown['total_cost'])
        mean_cost = np.mean(cost_breakdown['total_cost'])
        std_cost = np.std(cost_breakdown['total_cost'])
        
        # 找到最高cost的时间点
        max_cost_idx = np.argmax(cost_breakdown['total_cost'])
        
        stats_text = f"""
    Cost Statistics:

    Total Cost: {total_cost:.3f}
    Maximum Cost: {max_cost:.3f} (at step {max_cost_idx})
    Average Cost: {mean_cost:.3f}
    Cost Std Dev: {std_cost:.3f}

    Cost Distribution:
    • State Cost: {np.mean(cost_breakdown['state_cost']):.3f}
    • Control Cost: {np.mean(cost_breakdown['control_cost']):.3f}
    • Position Cost: {np.mean(cost_breakdown['position_cost']):.3f}
    • Speed Cost: {np.mean(cost_breakdown['speed_cost']):.3f}
    • Yaw Cost: {np.mean(cost_breakdown['yaw_cost']):.3f}

    High Cost Regions:
    • Steps with cost > mean+std: {np.sum(cost_breakdown['total_cost'] > mean_cost + std_cost)}
    • Peak cost location: step {max_cost_idx}

    Probability Heatmap Info:
    • Blue regions: High probability (Low cost)
    • Red regions: Low probability (High cost)  
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def visualize_prediction(self, hist_states, predicted_states, true_states=None, mpc_params=None):
        """Visualize prediction results with multiple targets (original simple version)"""
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
            ds_indices = np.arange(predictor.downsample_factor - 1, len(seq['future_states']), predictor.downsample_factor)
            downsampled_future_states = seq['future_states'][ds_indices]
            
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
    parser = argparse.ArgumentParser(description='Vehicle LSTM-MPC Training and Prediction with Cost Probability Heatmap')
    
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
    parser.add_argument('--num_targets', type=int, default=10, 
                       help='Number of target points to use across MPC horizon')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Run mode
    parser.add_argument('--mode', choices=['train', 'test', 'evaluate'], default='train',
                       help='Mode: train, test, or evaluate')
    parser.add_argument('--save_results', action='store_true', help='Save error results to file')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"=== Vehicle LSTM-MPC Controller with Cost Probability Heatmap ===")
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
        for i in range(0, 10000, 100):
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
            
            # For comparison, downsample ground truth
            ds_indices = np.arange(predictor.downsample_factor-1, len(future_states), predictor.downsample_factor)
            downsampled_future_states = future_states[ds_indices[:len(predicted_states)]]
            
            # Use the cost probability heatmap visualization
            print(f"  Generating cost probability heatmap visualization...")
            predictor.visualize_prediction_with_heatmap(
                hist_states, 
                predicted_states, 
                downsampled_future_states,
                mpc_params
            )
            
            # Optional: Also show the original simple visualization for comparison
            show_simple_viz = input("Show simple visualization too? (y/n): ").lower().strip()
            if show_simple_viz == 'y':
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