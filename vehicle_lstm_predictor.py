import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

class VehicleDataset(Dataset):
    def __init__(self, sequences, scaler_states=None, scaler_controls=None, is_train=True):
        self.sequences = sequences
        self.is_train = is_train
        
        all_hist_states = np.vstack([seq['hist_states'] for seq in sequences])
        all_hist_controls = np.vstack([seq['hist_controls'] for seq in sequences])
        all_future_states = np.vstack([seq['future_states'] for seq in sequences])
        all_future_controls = np.vstack([seq['future_controls'] for seq in sequences])
        
        if scaler_states is None:
            self.scaler_states = StandardScaler()
            self.scaler_states.fit(np.vstack([all_hist_states, all_future_states]))
        else:
            self.scaler_states = scaler_states
            
        if scaler_controls is None:
            self.scaler_controls = StandardScaler()
            self.scaler_controls.fit(np.vstack([all_hist_controls, all_future_controls]))
        else:
            self.scaler_controls = scaler_controls
        
        print(f"Dataset initialized with {len(sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        hist_states = self.scaler_states.transform(seq['hist_states'])
        hist_controls = self.scaler_controls.transform(seq['hist_controls'])
        future_states = self.scaler_states.transform(seq['future_states'])
        future_controls = self.scaler_controls.transform(seq['future_controls'])
        
        input_features = np.concatenate([hist_states, hist_controls], axis=1)
        
        return {
            'input_features': torch.FloatTensor(input_features),
            'future_states': torch.FloatTensor(future_states),
            'future_controls': torch.FloatTensor(future_controls)
        }

class VehicleLSTM(nn.Module):
    def __init__(self, 
                 input_dim=7,
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=4,
                 predict_steps=100):
        super(VehicleLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_steps = predict_steps
        self.output_dim = output_dim
        
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_state = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.fc_control = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3) 
        )
        
    def forward(self, input_features):
        encoder_out, _ = self.encoder_lstm(input_features)
        context = encoder_out[:, -1, :]
        predicted_states, predicted_controls = self.decode(context)
        return predicted_states, predicted_controls
    
    def decode(self, context):
        # Expand context to all prediction timesteps
        decoder_input = context.unsqueeze(1).repeat(1, self.predict_steps, 1)
        
        h_dec, _ = self.decoder_lstm(decoder_input)
        predicted_states = self.fc_state(h_dec)
        predicted_controls = self.fc_control(h_dec)
        
        return predicted_states, predicted_controls

class TrajectoryLoss(nn.Module):
    def __init__(self, state_weights=None, control_weight=0.0):
        super(TrajectoryLoss, self).__init__()
        
        if state_weights is None:
            self.state_weights = torch.tensor([10.0, 10.0, 1.0, 1.0])
        else:
            self.state_weights = torch.tensor(state_weights)
            
        self.control_weight = control_weight
        self.mse = nn.MSELoss(reduction='none')  # Important: use 'none' for element-wise loss
    
    def forward(self, pred_states, true_states, pred_controls=None, true_controls=None):
        # Calculate element-wise MSE for states
        state_mse = self.mse(pred_states, true_states)  # (batch_size, seq_len, 4)
        
        # Move weights to the same device as the states
        weights = self.state_weights.to(pred_states.device)
        
        # Apply weights to different state dimensions
        # Broadcasting: (batch_size, seq_len, 4) * (4,) -> (batch_size, seq_len, 4)
        weighted_state_loss = state_mse * weights.unsqueeze(0).unsqueeze(0)
        
        # Average over all dimensions
        state_loss = weighted_state_loss.mean()
        total_loss = state_loss
        
        control_loss = torch.tensor(0.0, device=pred_states.device)
        if pred_controls is not None and true_controls is not None:
            control_loss = self.mse(pred_controls, true_controls).mean()
            total_loss = state_loss + self.control_weight * control_loss
            return total_loss, state_loss, control_loss
        
        return total_loss

class VehicleLSTMTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', save_dir='models'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.save_dir = save_dir
        
        # Create model save directory
        os.makedirs(save_dir, exist_ok=True)
        
    def train_model(self, train_loader, val_loader, epochs=100, lr=1e-3, patience=15):
        criterion = TrajectoryLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Model will be saved to: {self.save_dir}")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_state_loss = 0.0
            train_control_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                input_features = batch['input_features'].to(self.device)
                future_states = batch['future_states'].to(self.device)
                future_controls = batch['future_controls'].to(self.device)
                
                pred_states, pred_controls = self.model(input_features)
                
                loss_result = criterion(pred_states, future_states, pred_controls, future_controls)
                if isinstance(loss_result, tuple):
                    loss, state_loss, control_loss = loss_result
                    train_control_loss += control_loss.item()
                else:
                    loss = loss_result
                    state_loss = loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_state_loss += state_loss.item()
            
            train_loss /= len(train_loader)
            train_state_loss /= len(train_loader)
            train_control_loss /= len(train_loader)
            
            val_loss, val_state_loss, val_control_loss = self.evaluate(val_loader, criterion)
            scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.6f} (State: {train_state_loss:.6f}, Control: {train_control_loss:.6f})")
            print(f"  Val Loss: {val_loss:.6f} (State: {val_state_loss:.6f}, Control: {val_control_loss:.6f})")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model to specified directory
                model_path = os.path.join(self.save_dir, 'best_vehicle_lstm.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': {
                        'input_dim': self.model.encoder_lstm.input_size,
                        'hidden_dim': self.model.hidden_dim,
                        'num_layers': self.model.num_layers,
                        'output_dim': self.model.output_dim,
                        'predict_steps': self.model.predict_steps
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
        
    def evaluate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0.0
        val_state_loss = 0.0
        val_control_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_features = batch['input_features'].to(self.device)
                future_states = batch['future_states'].to(self.device)
                future_controls = batch['future_controls'].to(self.device)
                
                pred_states, pred_controls = self.model(input_features)
                
                loss_result = criterion(pred_states, future_states, pred_controls, future_controls)
                if isinstance(loss_result, tuple):
                    loss, state_loss, control_loss = loss_result
                    val_control_loss += control_loss.item()
                else:
                    loss = loss_result
                    state_loss = loss
                
                val_loss += loss.item()
                val_state_loss += state_loss.item()
        
        return val_loss / len(val_loader), val_state_loss / len(val_loader), val_control_loss / len(val_loader)

class VehiclePredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            self.model = VehicleLSTM(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'], 
                num_layers=config['num_layers'],
                output_dim=config['output_dim'],
                predict_steps=config['predict_steps']
            )
        else:
            print("Warning: Old checkpoint format detected. Using default config with num_layers=1")
            self.model = VehicleLSTM(num_layers=1)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    
    def predict_trajectory(self, hist_states, hist_controls, scaler_states, scaler_controls):
        hist_states_norm = scaler_states.transform(hist_states)
        hist_controls_norm = scaler_controls.transform(hist_controls)
        
        input_features = np.concatenate([hist_states_norm, hist_controls_norm], axis=1)
        input_features = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_states, pred_controls = self.model(input_features)
        
        pred_states = pred_states.cpu().numpy().squeeze()
        pred_controls = pred_controls.cpu().numpy().squeeze()
        
        pred_states = scaler_states.inverse_transform(pred_states)
        pred_controls = scaler_controls.inverse_transform(pred_controls)
        
        return pred_states, pred_controls
    
    def visualize_prediction(self, hist_states, pred_states, true_states=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Trajectory
        ax = axes[0, 0]
        ax.plot(hist_states[:, 0], hist_states[:, 1], 'b-', label='History', linewidth=2)
        ax.plot(pred_states[:, 0], pred_states[:, 1], 'r--', label='Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(true_states[:, 0], true_states[:, 1], 'g-', label='Ground Truth', linewidth=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Speed
        ax = axes[0, 1]
        time_hist = np.arange(len(hist_states)) * 0.05
        time_pred = np.arange(len(hist_states), len(hist_states) + len(pred_states)) * 0.05
        
        ax.plot(time_hist, hist_states[:, 3] * 3.6, 'b-', label='History', linewidth=2)
        ax.plot(time_pred, pred_states[:, 3] * 3.6, 'r--', label='Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(time_pred, true_states[:, 3] * 3.6, 'g-', label='Ground Truth', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Speed Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Yaw angle
        ax = axes[1, 0]
        ax.plot(time_hist, np.rad2deg(hist_states[:, 2]), 'b-', label='History', linewidth=2)
        ax.plot(time_pred, np.rad2deg(pred_states[:, 2]), 'r--', label='Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(time_pred, np.rad2deg(true_states[:, 2]), 'g-', label='Ground Truth', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw Angle (degrees)')
        ax.set_title('Yaw Angle Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Errors
        if true_states is not None:
            ax = axes[1, 1]
            position_error = np.sqrt((pred_states[:, 0] - true_states[:, 0])**2 + 
                                   (pred_states[:, 1] - true_states[:, 1])**2)
            speed_error = np.abs(pred_states[:, 3] - true_states[:, 3]) * 3.6
            
            ax.plot(time_pred, position_error, 'r-', label='Position Error (m)', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(time_pred, speed_error, 'b--', label='Speed Error (km/h)', linewidth=2)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position Error (m)', color='r')
            ax2.set_ylabel('Speed Error (km/h)', color='b')
            ax.set_title('Prediction Errors')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def calculate_temporal_errors(pred_states, true_states, time_step=0.05):
    """Calculate average errors at different future time points"""
    time_points = [1, 2, 3, 4, 5]  # seconds
    time_indices = [int(t / time_step) - 1 for t in time_points]  # convert to indices
    
    error_results = {}
    
    for i, (t, idx) in enumerate(zip(time_points, time_indices)):
        if idx >= pred_states.shape[1]:
            continue
            
        # 位置误差
        position_error = np.sqrt(
            (pred_states[:, idx, 0] - true_states[:, idx, 0])**2 + 
            (pred_states[:, idx, 1] - true_states[:, idx, 1])**2
        )
        
        # 速度误差
        speed_error = np.abs(pred_states[:, idx, 3] - true_states[:, idx, 3]) * 3.6
        
        # 朝向角误差
        yaw_error = np.abs(pred_states[:, idx, 2] - true_states[:, idx, 2])
        yaw_error = np.minimum(yaw_error, 2*np.pi - yaw_error)
        yaw_error = np.rad2deg(yaw_error)
        
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

def evaluate_full_dataset(predictor, test_dataset_full, scalers, device='cpu'):
    """Evaluate entire test dataset and compute temporal error statistics"""
    print("Starting full dataset evaluation...")
    print(f"Total test sequences: {len(test_dataset_full['training_sequences'])}")
    
    all_pred_states = []
    all_true_states = []
    
    for i, seq in enumerate(tqdm(test_dataset_full['training_sequences'], desc="Processing sequences")):
        try:
            pred_states, pred_controls = predictor.predict_trajectory(
                seq['hist_states'], seq['hist_controls'],
                scalers['scaler_states'], scalers['scaler_controls']
            )
            
            all_pred_states.append(pred_states)
            all_true_states.append(seq['future_states'])
            
        except Exception as e:
            print(f"Error processing sequence {i}: {e}")
            continue
    
    if not all_pred_states:
        print("No valid predictions generated!")
        return None
    
    all_pred_states = np.array(all_pred_states)
    all_true_states = np.array(all_true_states)
    
    print(f"Successfully processed {len(all_pred_states)} sequences")
    
    temporal_errors = calculate_temporal_errors(all_pred_states, all_true_states)
    
    return temporal_errors

def print_error_statistics(error_results):
    """Print error statistics results"""
    print("\n" + "="*80)
    print("TEMPORAL ERROR ANALYSIS RESULTS")
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

def plot_error_trends(error_results):
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
    ax.set_title('Position Error vs Time')
    ax.grid(True, alpha=0.3)
    
    # Speed error
    ax = axes[1]
    ax.errorbar(time_points, speed_means, yerr=speed_stds, marker='s', capsize=5, capthick=2, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed Error (km/h)')
    ax.set_title('Speed Error vs Time')
    ax.grid(True, alpha=0.3)
    
    # Yaw error
    ax = axes[2]
    ax.errorbar(time_points, yaw_means, yerr=yaw_stds, marker='^', capsize=5, capthick=2, color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw Error (degrees)')
    ax.set_title('Yaw Error vs Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def load_dataset_from_folder(data_folder, dataset_name):
    """Load dataset from specified folder"""
    dataset_path = os.path.join(data_folder, dataset_name)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset loaded: {dataset_path}")
    print(f"  Sequences: {len(dataset['training_sequences'])}")
    if 'config' in dataset:
        print(f"  Configuration: {dataset['config']}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Vehicle LSTM Trajectory Prediction')
    
    # Data folder parameters
    parser.add_argument('--data_folder', default='vehicle_datasets', help='Folder containing all datasets')
    
    # Dataset filename parameters (optional for custom filenames)
    parser.add_argument('--train_file', default='vehicle_train_dataset.pkl', help='Training dataset filename')
    parser.add_argument('--test_file', default='vehicle_test_dataset.pkl', help='Test dataset filename')
    
    # Model save directory
    parser.add_argument('--save_dir', default='models', help='Directory to save models and scalers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='LSTM layers')
    
    # Run mode
    parser.add_argument('--mode', choices=['train', 'test', 'predict', 'evaluate'], default='evaluate', 
                       help='Mode: train, test, predict, or evaluate')
    parser.add_argument('--save_results', action='store_true', help='Save error results to file')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"=== Vehicle LSTM Trajectory Prediction ===")
    print(f"Data folder: {args.data_folder}")
    print(f"Model save directory: {args.save_dir}")
    print(f"Mode: {args.mode}")
    print("-" * 50)
    
    if args.mode == 'train':
        # Load training and validation datasets
        train_dataset_full = load_dataset_from_folder(args.data_folder, args.train_file)
        val_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
        
        train_dataset = VehicleDataset(train_dataset_full['training_sequences'])
        val_dataset = VehicleDataset(val_dataset_full['training_sequences'], 
                                   train_dataset.scaler_states, 
                                   train_dataset.scaler_controls, 
                                   is_train=False)
        
        # Save scalers to model directory
        scalers_path = os.path.join(args.save_dir, 'scalers.pkl')
        with open(scalers_path, 'wb') as f:
            pickle.dump({
                'scaler_states': train_dataset.scaler_states,
                'scaler_controls': train_dataset.scaler_controls
            }, f)
        print(f"Scalers saved to: {scalers_path}")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        model = VehicleLSTM(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        trainer = VehicleLSTMTrainer(model, save_dir=args.save_dir)
        trainer.train_model(train_loader, val_loader, epochs=args.epochs, lr=args.lr)
        
    elif args.mode == 'test':
        # Load test dataset
        test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
        
        # Load scalers
        scalers_path = os.path.join(args.save_dir, 'scalers.pkl')
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)

        sample_start = 700
        sample_end = 1000

        test_dataset = VehicleDataset(test_dataset_full['training_sequences'][sample_start:sample_end],
                                    scalers['scaler_states'],
                                    scalers['scaler_controls'],
                                    is_train=False)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Load model
        model_path = os.path.join(args.save_dir, 'best_vehicle_lstm.pth')
        predictor = VehiclePredictor(model_path)
        
        for i, batch in enumerate(test_loader):
            if i >= 10:
                break
                
            seq = test_dataset_full['training_sequences'][sample_start + i]
            pred_states, pred_controls = predictor.predict_trajectory(
                seq['hist_states'], seq['hist_controls'],
                scalers['scaler_states'], scalers['scaler_controls']
            )
            print(seq['hist_states'], pred_states, seq['future_states'])
            predictor.visualize_prediction(seq['hist_states'], pred_states, seq['future_states'])
    
    elif args.mode == 'evaluate':
        print("Loading test dataset and model...")
        
        # Load test dataset
        test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
        
        # Load scalers
        scalers_path = os.path.join(args.save_dir, 'scalers.pkl')
        if not os.path.exists(scalers_path):
            print(f"Error: Scalers not found at {scalers_path}")
            print("Please train the model first to generate scalers.")
            return
            
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        # Load model
        model_path = os.path.join(args.save_dir, 'best_vehicle_lstm.pth')
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first.")
            return
            
        predictor = VehiclePredictor(model_path)
        
        # Evaluate entire test dataset
        error_results = evaluate_full_dataset(predictor, test_dataset_full, scalers)
        
        if error_results:
            # Print results
            print_error_statistics(error_results)
            
            # Plot trends
            plot_error_trends(error_results)
            
            # Save results (optional)
            if args.save_results:
                results_path = os.path.join(args.save_dir, 'temporal_error_results.pkl')
                with open(results_path, 'wb') as f:
                    pickle.dump(error_results, f)
                print(f"Results saved to: {results_path}")
        else:
            print("Evaluation failed!")
    
    elif args.mode == 'predict':
        print("Interactive prediction mode")
        print("Not implemented yet. Use 'test' mode for visualization.")

if __name__ == "__main__":
    main()