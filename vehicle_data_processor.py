"""
Vehicle Data Processor: Convert CARLA data to LSTM-MPC training format
Modified to save all pkl files to a folder and use relative coordinate system
"""

import pandas as pd
import numpy as np
import os
import pickle
import glob
from pathlib import Path
import argparse
from tqdm import tqdm

class VehicleDataProcessor:
    """Vehicle trajectory data processor with relative coordinate system"""
    
    def __init__(self, sampling_freq=20.0, history_time=3.0, predict_time=5.0):
        """
        Args:
            sampling_freq: Sampling frequency (Hz)
            history_time: History time window (seconds)
            predict_time: Prediction time window (seconds)
        """
        self.sampling_freq = sampling_freq
        self.history_time = history_time
        self.predict_time = predict_time
        
        # Calculate corresponding steps
        self.history_steps = int(history_time * sampling_freq) + 1  # 3s * 20Hz + 1 = 61 steps (包含当前状态)
        self.predict_steps = int(predict_time * sampling_freq)       # 5s * 20Hz = 100 steps
        
        print(f"Data processing config:")
        print(f"  Sampling freq: {sampling_freq} Hz")
        print(f"  History window: {history_time}s + current ({self.history_steps} steps total)")
        print(f"  Prediction window: {predict_time}s ({self.predict_steps} steps)")
        print(f"  Using relative coordinate system (current state as origin)")
    
    def normalize_angle(self, angle):
        """将角度归一化到 [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def transform_to_coordinates(self, states, current_state):
        """
        将状态转换为以current_state为原点的相对坐标系
        Args:
            states: (N, 4) array [x, y, yaw, speed]  
            current_state: (4,) array [x_current, y_current, yaw_current, speed_current]
        Returns:
            relative_states: (N, 4) array in relative coordinate system
        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
            
        relative_states = states.copy()
        
        # 提取当前状态
        x_current, y_current, yaw_current, speed_current = current_state
        
        # 1. 平移：将当前位置设为原点
        relative_states[:, 0] = states[:, 0] - x_current  # x' = x - x_current
        relative_states[:, 1] = states[:, 1] - y_current  # y' = y - y_current
        
        # 2. 旋转：将当前朝向设为0度方向
        cos_yaw = np.cos(-yaw_current)
        sin_yaw = np.sin(-yaw_current)
        
        x_translated = relative_states[:, 0]
        y_translated = relative_states[:, 1]
        
        # 旋转变换
        relative_states[:, 0] = x_translated * cos_yaw - y_translated * sin_yaw
        relative_states[:, 1] = x_translated * sin_yaw + y_translated * cos_yaw
        
        # 3. 角度变换：相对于当前朝向
        relative_states[:, 2] = self.normalize_angle(states[:, 2] - yaw_current)
        
        # 4. 速度保持不变
        relative_states[:, 3] = states[:, 3]
        
        return relative_states
    
    def load_carla_episode(self, episode_path):
        """Load single CARLA episode data"""
        trajectory_file = os.path.join(episode_path, 'trajectory.csv')
        
        if not os.path.exists(trajectory_file):
            print(f"Warning: trajectory file not found {trajectory_file}")
            return None
        
        try:
            df = pd.read_csv(trajectory_file)
            return df
        except Exception as e:
            print(f"Failed to load trajectory file {trajectory_file}: {e}")
            return None
    
    def extract_vehicle_states(self, df):
        """Extract vehicle states from CARLA data"""
        # Vehicle states: [x, y, yaw, speed]
        states = np.column_stack([
            df['x'].values,      # x position
            df['y'].values,      # y position
            np.deg2rad(df['yaw'].values),  # yaw angle (deg to rad)
            df['speed'].values / 3.6       # speed (km/h to m/s)
        ])
        
        return states
    
    def extract_vehicle_controls(self, df):
        """Extract control inputs from CARLA data"""
        # Control inputs: [throttle, brake, steer]
        controls = np.column_stack([
            df['throttle'].values,
            df['brake'].values,
            df['steer'].values
        ])
        
        return controls
    
    def create_training_sequences(self, states, controls):
        """Create training sequences from trajectory data with relative coordinate system"""
        sequences = []
        total_steps = len(states)
        
        # Minimum required trajectory length: history (including current) + future
        min_length = self.history_steps + self.predict_steps  # 61 + 100 = 161
        
        if total_steps < min_length:
            print(f"Trajectory too short: {total_steps} < {min_length}, skipping")
            return sequences
        
        # Extract sequences with sliding window
        for i in range(total_steps - min_length + 1):
            current_idx = i + self.history_steps - 1  # 当前状态的索引
            
            # 原始数据序列
            hist_states_raw = states[i:i+self.history_steps]                    # (61, 4)
            hist_controls_raw = controls[i:i+self.history_steps]                # (61, 3)
            current_state_raw = states[current_idx]                             # (4,)
            current_control_raw = controls[current_idx]                         # (3,)
            future_states_raw = states[i+self.history_steps:i+self.history_steps+self.predict_steps]   # (100, 4)
            future_controls_raw = controls[i+self.history_steps:i+self.history_steps+self.predict_steps] # (100, 3)
            
            # === 坐标系变换：以当前状态为原点 ===
            # 变换历史状态
            hist_states = self.transform_to_coordinates(hist_states_raw, current_state_raw)
            
            # 变换未来状态
            future_states = self.transform_to_coordinates(future_states_raw, current_state_raw)
            
            # 当前状态在相对坐标系中应该是 [0, 0, 0, speed]
            current_state = np.array([0.0, 0.0, 0.0, current_state_raw[3]])
            
            # 控制量不需要变换
            hist_controls = hist_controls_raw
            current_control = current_control_raw  
            future_controls = future_controls_raw
            
            # 验证变换正确性：hist_states的最后一个状态应该接近 [0, 0, 0, speed]
            last_hist_state = hist_states[-1]
            assert abs(last_hist_state[0]) < 1e-10, f"x coordinate not zero: {last_hist_state[0]}"
            assert abs(last_hist_state[1]) < 1e-10, f"y coordinate not zero: {last_hist_state[1]}"  
            assert abs(last_hist_state[2]) < 1e-10, f"yaw angle not zero: {last_hist_state[2]}"
            
            sequence = {
                'hist_states': hist_states,           # Past 3s + current in relative coords: (61, 4)
                'hist_controls': hist_controls,       # Past 3s + current: (61, 3)  
                'current_state': current_state,       # [0, 0, 0, speed]: (4,)
                'current_control': current_control,   # t=0: (3,)
                'future_states': future_states,       # Future 5s in relative coords: (100, 4)
                'future_controls': future_controls,   # Future 5s: (100, 3)
                'sequence_id': i,
                'current_idx': current_idx,           # For debugging
                # 保留原始数据用于调试
                'raw_current_state': current_state_raw,  # 原始当前状态
                'raw_current_position': current_state_raw[:3]  # [x, y, yaw] 用于逆变换
            }
            
            sequences.append(sequence)
        
        return sequences
    
    def process_single_episode(self, episode_path):
        """Process single episode"""
        df = self.load_carla_episode(episode_path)
        if df is None:
            return []
        
        # Data preprocessing
        # 1. Remove stationary or abnormal data points
        df = df[df['speed'] > 0.1]  # Remove nearly stationary points
        df = df.reset_index(drop=True)
        
        min_required_length = self.history_steps + self.predict_steps  # 61 + 100 = 161
        if len(df) < min_required_length:
            print(f"Insufficient episode data points: {len(df)} < {min_required_length}")
            return []
        
        # 2. Extract states and controls
        states = self.extract_vehicle_states(df)
        controls = self.extract_vehicle_controls(df)
        
        # 3. Create training sequences with relative coordinate transformation
        sequences = self.create_training_sequences(states, controls)
        
        return sequences
    
    def process_carla_data(self, data_root, output_folder='vehicle_datasets', max_episodes=None):
        """Process CARLA dataset and save to folder"""
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder: {output_folder}")
        
        # Define output path for complete dataset
        output_path = os.path.join(output_folder, 'vehicle_lstm_dataset.pkl')
        
        # Find all episode directories
        episode_dirs = []
        for session_dir in glob.glob(os.path.join(data_root, '*')):
            if os.path.isdir(session_dir):
                for episode_dir in glob.glob(os.path.join(session_dir, 'episode_*')):
                    if os.path.isdir(episode_dir):
                        episode_dirs.append(episode_dir)
        
        if not episode_dirs:
            print(f"No episode data found in {data_root}")
            return
        
        print(f"Found {len(episode_dirs)} episodes")
        
        # Limit number of episodes to process
        if max_episodes and max_episodes < len(episode_dirs):
            episode_dirs = episode_dirs[:max_episodes]
            print(f"Processing first {max_episodes} episodes")
        
        # Process all episodes
        all_sequences = []
        episode_info = []
        
        for episode_dir in tqdm(episode_dirs, desc="Processing Episodes"):
            episode_name = os.path.basename(episode_dir)
            session_name = os.path.basename(os.path.dirname(episode_dir))
            
            try:
                sequences = self.process_single_episode(episode_dir)
                
                if sequences:
                    all_sequences.extend(sequences)
                    
                    episode_info.append({
                        'episode_name': episode_name,
                        'session_name': session_name,
                        'episode_path': episode_dir,
                        'num_sequences': len(sequences)
                    })
                
            except Exception as e:
                print(f"Failed to process episode {episode_dir}: {e}")
                continue
        
        print(f"\nData processing completed:")
        print(f"  Successfully processed episodes: {len(episode_info)}")
        print(f"  Total training sequences: {len(all_sequences)}")
        
        # Build dataset
        dataset = {
            'training_sequences': all_sequences,
            'episode_info': episode_info,
            'config': {
                'sampling_freq': self.sampling_freq,
                'history_time': self.history_time,
                'predict_time': self.predict_time,
                'history_steps': self.history_steps,  # includes current state
                'predict_steps': self.predict_steps,
                'state_dim': 4,    # [x, y, yaw, speed]
                'control_dim': 3,  # [throttle, brake, steer]
                'min_sequence_length': self.history_steps + self.predict_steps,
                'coordinate_system': 'relative',  # 新增：标记使用相对坐标系
                'relative_origin': 'current_state'  # 新增：以当前状态为原点
            }
        }
        
        # Save dataset to folder
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved: {output_path}")
        
        # Print statistics
        self.print_dataset_statistics(dataset)
        
        return dataset, output_folder  # Return both dataset and folder path
    
    def print_dataset_statistics(self, dataset):
        """Print dataset statistics"""
        sequences = dataset['training_sequences']
        episodes = dataset['episode_info']
        config = dataset['config']
        
        print(f"\n=== Dataset Statistics (Relative Coordinate System) ===")
        print(f"Configuration:")
        print(f"  Sampling freq: {config['sampling_freq']} Hz")
        print(f"  History window: {config['history_time']}s + current ({config['history_steps']} steps total)")
        print(f"  Prediction window: {config['predict_time']}s ({config['predict_steps']} steps)")
        print(f"  Minimum sequence length: {config['min_sequence_length']} steps")
        print(f"  State dimension: {config['state_dim']}")
        print(f"  Control dimension: {config['control_dim']}")
        print(f"  Coordinate system: {config.get('coordinate_system', 'absolute')} (origin: {config.get('relative_origin', 'N/A')})")
        
        print(f"\nData statistics:")
        print(f"  Number of episodes: {len(episodes)}")
        print(f"  Number of training sequences: {len(sequences)}")
        print(f"  Average sequences per episode: {np.mean([ep['num_sequences'] for ep in episodes]):.1f}")
        
        # Analyze state distribution in relative coordinate system
        if sequences:
            # 当前状态分布（应该都是[0,0,0,speed]）
            all_current_states = np.array([seq['current_state'] for seq in sequences])
            print(f"\nCurrent state verification (should be [0,0,0,speed]):")
            print(f"  X: min={np.min(all_current_states[:, 0]):.6f}, max={np.max(all_current_states[:, 0]):.6f}")
            print(f"  Y: min={np.min(all_current_states[:, 1]):.6f}, max={np.max(all_current_states[:, 1]):.6f}")
            print(f"  Yaw: min={np.min(all_current_states[:, 2]):.6f}, max={np.max(all_current_states[:, 2]):.6f}")
            print(f"  Speed: avg={np.mean(all_current_states[:, 3])*3.6:.1f} km/h, max={np.max(all_current_states[:, 3])*3.6:.1f} km/h")
            
            # 历史状态分布（相对坐标系）
            all_hist_states = np.concatenate([seq['hist_states'] for seq in sequences[:100]], axis=0)  # 只取前100个样本避免内存问题
            print(f"\nHistorical states distribution (relative coordinates, sample from first 100 sequences):")
            print(f"  X range: [{np.min(all_hist_states[:, 0]):.1f}, {np.max(all_hist_states[:, 0]):.1f}] m")
            print(f"  Y range: [{np.min(all_hist_states[:, 1]):.1f}, {np.max(all_hist_states[:, 1]):.1f}] m")
            print(f"  Yaw range: [{np.min(all_hist_states[:, 2]):.2f}, {np.max(all_hist_states[:, 2]):.2f}] rad")
            
            # 未来状态分布（相对坐标系）
            all_future_states = np.concatenate([seq['future_states'] for seq in sequences[:100]], axis=0)
            print(f"\nFuture states distribution (relative coordinates, sample from first 100 sequences):")
            print(f"  X range: [{np.min(all_future_states[:, 0]):.1f}, {np.max(all_future_states[:, 0]):.1f}] m")
            print(f"  Y range: [{np.min(all_future_states[:, 1]):.1f}, {np.max(all_future_states[:, 1]):.1f}] m")
            print(f"  Yaw range: [{np.min(all_future_states[:, 2]):.2f}, {np.max(all_future_states[:, 2]):.2f}] rad")
            
            # 原始当前位置分布（用于了解数据覆盖范围）
            all_raw_positions = np.array([seq['raw_current_position'] for seq in sequences])
            print(f"\nOriginal position distribution (for reference):")
            print(f"  X coordinate range: [{np.min(all_raw_positions[:, 0]):.1f}, {np.max(all_raw_positions[:, 0]):.1f}] m")
            print(f"  Y coordinate range: [{np.min(all_raw_positions[:, 1]):.1f}, {np.max(all_raw_positions[:, 1]):.1f}] m")
            print(f"  Yaw angle range: [{np.min(all_raw_positions[:, 2]):.2f}, {np.max(all_raw_positions[:, 2]):.2f}] rad")
            
        # Verify sequence structure
        if sequences:
            seq = sequences[0]
            print(f"\nSequence structure verification:")
            print(f"  hist_states shape: {seq['hist_states'].shape}")
            print(f"  hist_controls shape: {seq['hist_controls'].shape}")
            print(f"  current_state shape: {seq['current_state'].shape}")
            print(f"  current_control shape: {seq['current_control'].shape}")
            print(f"  future_states shape: {seq['future_states'].shape}")
            print(f"  future_controls shape: {seq['future_controls'].shape}")
            
            # 验证当前状态是否为[0,0,0,speed]
            current_state = seq['current_state']
            is_origin = np.allclose(current_state[:3], [0, 0, 0], atol=1e-10)
            print(f"  Current state is at origin [0,0,0]: {is_origin}")
            print(f"  Current state: [{current_state[0]:.6f}, {current_state[1]:.6f}, {current_state[2]:.6f}, {current_state[3]:.3f}]")
            
            # 验证hist_states最后一个元素是否也是[0,0,0,speed]
            last_hist_state = seq['hist_states'][-1]
            hist_matches_current = np.allclose(last_hist_state, current_state, atol=1e-10)
            print(f"  Last history state matches current state: {hist_matches_current}")
    
    def create_train_test_split(self, dataset_path, output_folder, train_ratio=0.8):
        """Create train/test dataset split and save to folder"""
        
        # Load dataset
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        sequences = dataset['training_sequences']
        episodes = dataset['episode_info']
        
        # Define output paths in the same folder
        train_output = os.path.join(output_folder, 'vehicle_train_dataset.pkl')
        test_output = os.path.join(output_folder, 'vehicle_test_dataset.pkl')
        
        # Split by episode to ensure same episode data doesn't appear in both train and test sets
        np.random.seed(42)
        episode_indices = np.random.permutation(len(episodes))
        train_episode_count = int(len(episodes) * train_ratio)
        
        train_episode_indices = set(episode_indices[:train_episode_count])
        test_episode_indices = set(episode_indices[train_episode_count:])
        
        # Allocate sequences
        train_sequences = []
        test_sequences = []
        train_episodes = []
        test_episodes = []
        
        sequence_idx = 0
        for ep_idx, episode in enumerate(episodes):
            episode_seq_count = episode['num_sequences']
            
            if ep_idx in train_episode_indices:
                train_sequences.extend(sequences[sequence_idx:sequence_idx + episode_seq_count])
                train_episodes.append(episode)
            else:
                test_sequences.extend(sequences[sequence_idx:sequence_idx + episode_seq_count])
                test_episodes.append(episode)
            
            sequence_idx += episode_seq_count
        
        # Build training set
        train_dataset = {
            'training_sequences': train_sequences,
            'episode_info': train_episodes,
            'config': dataset['config']
        }
        
        # Build test set
        test_dataset = {
            'training_sequences': test_sequences,
            'episode_info': test_episodes,
            'config': dataset['config']
        }
        
        # Save datasets to folder
        with open(train_output, 'wb') as f:
            pickle.dump(train_dataset, f)
        
        with open(test_output, 'wb') as f:
            pickle.dump(test_dataset, f)
        
        print(f"\nDataset split completed:")
        print(f"Training set: {len(train_sequences)} sequences, {len(train_episodes)} episodes -> {train_output}")
        print(f"Test set: {len(test_sequences)} sequences, {len(test_episodes)} episodes -> {test_output}")
        
        return train_dataset, test_dataset

def main():
    """Data processing main function"""
    parser = argparse.ArgumentParser(description='CARLA vehicle data processing with relative coordinate system')
    parser.add_argument('--data_root', default='collected_data', help='CARLA data root directory')
    parser.add_argument('--output_folder', default='vehicle_datasets', help='Output folder for all pkl files')
    parser.add_argument('--max_episodes', type=int, default=None, help='Maximum episodes to process')
    parser.add_argument('--sampling_freq', type=float, default=20.0, help='Sampling frequency (Hz)')
    parser.add_argument('--history_time', type=float, default=3.0, help='History time window (seconds)')
    parser.add_argument('--predict_time', type=float, default=5.0, help='Prediction time window (seconds)')
    parser.add_argument('--create_split', action='store_true', default=True, help='Create train/test split')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    
    args = parser.parse_args()
    
    print("=== CARLA Vehicle Data Processing Tool (Relative Coordinate System) ===")
    print(f"Data root directory: {args.data_root}")
    print(f"Output folder: {args.output_folder}")
    print(f"Using relative coordinate system: current state as origin [0,0,0]")
    
    # Create data processor
    processor = VehicleDataProcessor(
        sampling_freq=args.sampling_freq,
        history_time=args.history_time,
        predict_time=args.predict_time
    )
    
    # Process data and save to folder
    dataset, output_folder = processor.process_carla_data(
        data_root=args.data_root,
        output_folder=args.output_folder,
        max_episodes=args.max_episodes
    )
    
    # Create train/test split
    if args.create_split and dataset:
        dataset_path = os.path.join(output_folder, 'vehicle_lstm_dataset.pkl')
        processor.create_train_test_split(
            dataset_path=dataset_path,
            output_folder=output_folder,
            train_ratio=args.train_ratio
        )
    
    print(f"\nData processing completed!")
    print(f"All files saved to folder: {output_folder}/")
    print(f"  - vehicle_lstm_dataset.pkl (complete dataset)")
    print(f"  - vehicle_train_dataset.pkl (training set)")
    print(f"  - vehicle_test_dataset.pkl (test set)")
    print(f"\nKey changes:")
    print(f"  - Current state normalized to [0, 0, 0, speed]")
    print(f"  - All historical and future states in relative coordinate system") 
    print(f"  - Preserved original position data for potential inverse transformation")

if __name__ == "__main__":
    main()