"""
Vehicle LSTM-MPC Controller for CARLA
Predicts MPC parameters using LSTM for adaptive vehicle control
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
        max_accel = 5.0  # m/s^2
        max_steer = np.deg2rad(70)
        
        accelerations = (hist_controls[:, 0] - hist_controls[:, 1]) * max_accel
        steer_angles = hist_controls[:, 2] * max_steer
        
        hist_actions = torch.stack([accelerations, steer_angles], dim=1)  # (61, 2)
        
        # Current state at origin
        current_state = torch.FloatTensor(seq['current_state'])  # (4,) [0, 0, 0, speed]
        
        # Future states and controls
        future_states = torch.FloatTensor(seq['future_states'])  # (100, 4)
        future_controls = torch.FloatTensor(seq['future_controls'])  # (100, 3)
        
        # Convert future controls
        future_accelerations = (future_controls[:, 0] - future_controls[:, 1]) * max_accel
        future_steer_angles = future_controls[:, 2] * max_steer
        future_actions = torch.stack([future_accelerations, future_steer_angles], dim=1)  # (100, 2)
        
        # Concatenate states and actions for LSTM input
        input_seq = torch.cat([hist_states, hist_actions], dim=1)  # (61, 6)
        
        return {
            'input_seq': input_seq,
            'current_state': current_state,
            'future_states': future_states,
            'future_actions': future_actions
        }

class VehicleLSTMMPC(nn.Module):
    """LSTM network for predicting MPC parameters"""
    
    def __init__(self, 
                 input_dim=6,      # [x, y, yaw, speed, a, delta_f]
                 hidden_dim=128,
                 num_layers=1,
                 state_dim=4,      # [x, y, yaw, speed]
                 control_dim=2):   # [a, delta_f]
        super(VehicleLSTMMPC, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.control_dim = control_dim
        
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
        
        # Target state prediction head (combined state and control targets)
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim + control_dim)
        )
    
    def forward(self, x): 
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Predict MPC parameters
        state_weights = self.state_weight_head(last_hidden)  # (batch_size, 4)
        control_weights = self.control_weight_head(last_hidden)  # (batch_size, 2)
        target = self.target_head(last_hidden)  # (batch_size, 6)
        
        return state_weights, control_weights, target

class VehicleMPCSolver:
    """MPC solver using mpc.pytorch library with kinematic bicycle model"""
    
    def __init__(self, dt=0.05, horizon=100, device='cuda'):
        self.dt = dt
        self.horizon = horizon
        self.device = device
        
        # Create kinematic bicycle model
        self.model = Kinematic_Bicycle_MPC(dt=dt)
        self.n_state = self.model.s_dim  # 4: [x, y, psi, v]
        self.n_ctrl = self.model.a_dim   # 2: [a, delta_f]
        
        # Control bounds
        self.u_min = torch.tensor([-100.0, -np.deg2rad(70)], device=device)  # [a_min, delta_min]
        self.u_max = torch.tensor([100.0, np.deg2rad(70)], device=device)    # [a_max, delta_max]
        
    def solve(self, initial_state, state_weights, control_weights, target):
        """
        Solve MPC optimization using mpc.pytorch
        
        Args:
            initial_state: (batch_size, 4) [x, y, v, psi]
            state_weights: (batch_size, 4) state weights
            control_weights: (batch_size, 2) control weights
            target: (batch_size, 6) target state and controls
            
        Returns:
            predicted_states: (batch_size, horizon, 4)
            optimal_controls: (batch_size, horizon, 2)
        """
        batch_size = initial_state.size(0)

        # Setup bounds for MPC
        u_lower = self.u_min.unsqueeze(0).unsqueeze(0).repeat(self.horizon, batch_size, 1)
        u_upper = self.u_max.unsqueeze(0).unsqueeze(0).repeat(self.horizon, batch_size, 1)
        
        # Build quadratic cost matrices
        weights = torch.cat([state_weights, control_weights], dim=1)  # (batch, 6)
         
        # Create diagonal cost matrix C for each batch
        weight_diag = torch.diag_embed(weights)

        C = weight_diag.unsqueeze(0).repeat(self.horizon, 1, 1, 1)  # [T, batch_size, n_state+n_ctrl, n_state+n_ctrl]

        target = target.unsqueeze(0).repeat(self.horizon, 1, 1)  # [T, batch_size, n_state+n_ctrl]
        # Build cost vectors
        c = -torch.sqrt(weights).unsqueeze(0) * target  # [T, batch_size, n_state+n_ctrl]

        # Create QuadCost object
        cost = QuadCost(C, c)
        
        # Setup MPC controller
        ctrl = mpc.MPC(
            n_state=self.n_state,
            n_ctrl=self.n_ctrl,
            T=self.horizon,
            # u_lower=u_lower,
            # u_upper=u_upper,
            lqr_iter=3,
            grad_method=GradMethods.ANALYTIC,
            exit_unconverged=False,
            detach_unconverged=False,
            backprop=True,
            verbose=0
        )
        
        # Solve MPC problem
        x_pred, u_opt, _ = ctrl(initial_state, cost, self.model)
        
        # Transpose to (batch, time, dim)
        predicted_states = x_pred.transpose(0, 1)
        optimal_controls = u_opt.transpose(0, 1)
        
        return predicted_states, optimal_controls
        
class VehicleLSTMMPCTrainer:
    """Trainer for Vehicle LSTM-MPC model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
        # MPC solver
        self.mpc_solver = VehicleMPCSolver(dt=0.05, horizon=100, device=device)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def train_model(self, train_loader, val_loader, epochs=100, lr=1e-3, patience=15, save_dir='models'):
        
        os.makedirs(save_dir, exist_ok=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
                
                model_path = os.path.join(save_dir, 'best_vehicle_lstm_mpc.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': {
                        'input_dim': self.model.input_dim,
                        'hidden_dim': self.model.hidden_dim,
                        'num_layers': self.model.num_layers,
                        'state_dim': self.model.state_dim,
                        'control_dim': self.model.control_dim
                    },
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
        """Compute MPC-based loss"""
        
        try:
            # Use MPC solver with predicted parameters
            mpc_states, mpc_controls = self.mpc_solver.solve(
                current_state,
                state_weights,
                control_weights,
                target
            )
            
            # State trajectory loss (position and velocity)
            position_loss = self.mse_loss(
                mpc_states[:, :, :2],  # x, y
                future_states[:, :, :2]
            )
            
            velocity_loss = self.mse_loss(
                mpc_states[:, :, 2],  # speed
                future_states[:, :, 2]
            )
            
            heading_loss = self.mse_loss(
                mpc_states[:, :, 3],  # yaw
                future_states[:, :, 3]
            )
            
            trajectory_loss = position_loss + 0.01 * velocity_loss + 0.01 * heading_loss
            
            # Control loss
            control_loss = self.mse_loss(
                mpc_controls[:, :],
                future_actions[:, :]
            )
            
        except Exception as e:
            # If MPC fails, use only regularization loss
            # This signals that the predicted parameters are problematic
            print(f"WARNING: MPC failed in loss computation: {e}")
            trajectory_loss = torch.tensor(10.0, device=self.device, requires_grad=True)
            control_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        # Parameter regularization
        state_weight_reg = torch.mean((state_weights - 1.0).pow(2))
        control_weight_reg = torch.mean((control_weights - 0.1).pow(2))
        target_reg = torch.mean(target.pow(2)) * 0.01
        
        regularization = 0.001 * (state_weight_reg + control_weight_reg + target_reg)
        
        # Total loss
        total_loss = trajectory_loss + 0.0 * control_loss + regularization
        
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
    """Online predictor for vehicle control"""
    
    def __init__(self, model_path, device='cpu', horizon=100):
        self.device = device
        self.horizon = horizon
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Rebuild model
        config = checkpoint['model_config']
        self.model = VehicleLSTMMPC(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            state_dim=config['state_dim'],
            control_dim=config['control_dim']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # MPC solver
        self.mpc_solver = VehicleMPCSolver(dt=0.05, horizon=horizon, device=device)
        
        print(f"Model loaded from {model_path}")
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    
    def predict_control(self, hist_states, hist_actions, current_state):
        """
        Predict MPC parameters and solve for optimal control
        
        Args:
            hist_states: (seq_len, 4) historical states
            hist_actions: (seq_len, 2) historical actions
            current_state: (4,) current state
            
        Returns:
            optimal_controls: (horizon, 2) optimal control sequence
            predicted_states: (horizon, 4) predicted state trajectory
            mpc_params: dict of MPC parameters
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
        
        mpc_params = {
            'state_weights': state_weights.cpu().numpy().squeeze(),
            'control_weights': control_weights.cpu().numpy().squeeze(),
            'target': target.cpu().numpy().squeeze(),
            'target_state': target[:, :4].cpu().numpy().squeeze(),
            'target_control': target[:, 4:].cpu().numpy().squeeze()
        }
        
        return optimal_controls, predicted_states, mpc_params
    
    def visualize_prediction(self, hist_states, predicted_states, true_states=None, mpc_params=None):
        """Visualize prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Trajectory plot
        ax = axes[0, 0]
        ax.plot(hist_states[:, 0], hist_states[:, 1], 'b-', label='History', linewidth=2)
        ax.plot(predicted_states[:, 0], predicted_states[:, 1], 'r--', label='MPC Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(true_states[:, 0], true_states[:, 1], 'g-', label='Ground Truth', linewidth=2)
        
        if mpc_params is not None:
            target_state = mpc_params['target_state']
            ax.plot(target_state[0], target_state[1], 'r*', markersize=15, label='Target')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Vehicle Trajectory (LSTM-MPC)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Speed profile
        ax = axes[0, 1]
        time_hist = np.arange(len(hist_states)) * 0.05
        time_pred = np.arange(len(predicted_states)) * 0.05
        
        ax.plot(time_hist, hist_states[:, 2] * 3.6, 'b-', label='History', linewidth=2)
        ax.plot(time_pred, predicted_states[:, 2] * 3.6, 'r--', label='MPC Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(time_pred, true_states[:, 2] * 3.6, 'g-', label='Ground Truth', linewidth=2)
        
        if mpc_params is not None:
            ax.axhline(y=mpc_params['target_state'][2] * 3.6, color='r', linestyle=':', label='Target Speed')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Speed Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Yaw angle
        ax = axes[1, 0]
        ax.plot(time_hist, np.rad2deg(hist_states[:, 3]), 'b-', label='History', linewidth=2)
        ax.plot(time_pred, np.rad2deg(predicted_states[:, 3]), 'r--', label='MPC Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(time_pred, np.rad2deg(true_states[:, 3]), 'g-', label='Ground Truth', linewidth=2)
        
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
            ax.set_title('MPC Weights (LSTM Predicted)')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
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
    parser = argparse.ArgumentParser(description='Vehicle LSTM-MPC Training and Prediction')
    
    # Data parameters
    parser.add_argument('--data_folder', default='vehicle_datasets', help='Folder containing datasets')
    parser.add_argument('--train_file', default='vehicle_train_dataset.pkl', help='Training dataset')
    parser.add_argument('--test_file', default='vehicle_test_dataset.pkl', help='Test dataset')
    
    # Model parameters
    parser.add_argument('--save_dir', default='models', help='Model save directory')
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Run mode
    parser.add_argument('--mode', choices=['train', 'test', 'predict'], default='train',
                       help='Mode: train, test, or predict')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"=== Vehicle LSTM-MPC Controller ===")
    print(f"Mode: {args.mode}")
    
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
        
        # Create model
        model = VehicleLSTMMPC(
            input_dim=6,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            state_dim=4,
            control_dim=2
        )
        
        # Create trainer
        trainer = VehicleLSTMMPCTrainer(model)
        
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
        model_path = os.path.join(args.save_dir, 'best_vehicle_lstm_mpc.pth')
        predictor = VehicleLSTMMPCPredictor(model_path)
        
        # Test samples
        for i in range(min(5, len(test_dataset_full['training_sequences']))):
            seq = test_dataset_full['training_sequences'][i]
            
            # Prepare data
            hist_states = seq['hist_states']
            current_state = seq['current_state']
            future_states = seq['future_states']
            
            # Convert controls
            max_accel = 5.0
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
            print(f"  Target: {mpc_params['target']}")
            print(f"  State weights: {mpc_params['state_weights']}")
            print(f"  Control weights: {mpc_params['control_weights']}")
            
            # Visualize
            predictor.visualize_prediction(
                hist_states, 
                predicted_states[:len(future_states)], 
                future_states[:len(predicted_states)],
                mpc_params
            )

if __name__ == "__main__":
    main()