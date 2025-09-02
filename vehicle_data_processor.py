"""
Vehicle Data Processor: Convert CARLA data to LSTM-MPC training format
Supports 3s history -> 5s prediction time window configuration
Fixed time window alignment issue
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
    """Vehicle trajectory data processor"""
    
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
        """Create training sequences from trajectory data"""
        sequences = []
        total_steps = len(states)
        
        # Minimum required trajectory length: history (including current) + future
        min_length = self.history_steps + self.predict_steps  # 61 + 100 = 161
        
        if total_steps < min_length:
            print(f"Trajectory too short: {total_steps} < {min_length}, skipping")
            return sequences
        
        # Extract sequences with sliding window
        for i in range(total_steps - min_length + 1):
            # Time alignment:
            # - Historical states: [i, i+history_steps)     -> past 3s + current (61 steps, t=-3s to t=0s)
            # - Current state: i+history_steps-1            -> t=0 (last step of history)
            # - Future states: [i+history_steps, i+history_steps+predict_steps) -> future 5s (100 steps, t=0.05s to t=5s)
            
            current_idx = i + self.history_steps - 1  # 当前状态的索引
            
            # Historical sequences (input) - past 3s + current moment (包含当前状态)
            hist_states = states[i:i+self.history_steps]                    # (61, 4)
            hist_controls = controls[i:i+self.history_steps]                # (61, 3)
            
            # Current state (MPC initial state) - t=0 (历史状态的最后一个)
            current_state = states[current_idx]                             # (4,)
            current_control = controls[current_idx]                         # (3,)
            
            # Future ground truth trajectory (target) - future 5 seconds
            future_states = states[i+self.history_steps:i+self.history_steps+self.predict_steps]   # (100, 4)
            future_controls = controls[i+self.history_steps:i+self.history_steps+self.predict_steps] # (100, 3)
            
            sequence = {
                'hist_states': hist_states,           # Past 3s + current: (61, 4)
                'hist_controls': hist_controls,       # Past 3s + current: (61, 3)  
                'current_state': current_state,       # t=0: (4,) - same as hist_states[-1]
                'current_control': current_control,   # t=0: (3,) - same as hist_controls[-1]
                'future_states': future_states,       # Future 5s: (100, 4)
                'future_controls': future_controls,   # Future 5s: (100, 3)
                'sequence_id': i,
                'current_idx': current_idx            # For debugging
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
        
        # 3. Create training sequences
        sequences = self.create_training_sequences(states, controls)
        
        return sequences
    
    def process_carla_data(self, data_root, output_path=None, max_episodes=None):
        """Process CARLA dataset"""
        
        if output_path is None:
            output_path = 'vehicle_lstm_dataset.pkl'
        
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
                'min_sequence_length': self.history_steps + self.predict_steps
            }
        }
        
        # Save dataset
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved: {output_path}")
        
        # Print statistics
        self.print_dataset_statistics(dataset)
        
        return dataset
    
    def print_dataset_statistics(self, dataset):
        """Print dataset statistics"""
        sequences = dataset['training_sequences']
        episodes = dataset['episode_info']
        config = dataset['config']
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Configuration:")
        print(f"  Sampling freq: {config['sampling_freq']} Hz")
        print(f"  History window: {config['history_time']}s + current ({config['history_steps']} steps total)")
        print(f"  Prediction window: {config['predict_time']}s ({config['predict_steps']} steps)")
        print(f"  Minimum sequence length: {config['min_sequence_length']} steps")
        print(f"  State dimension: {config['state_dim']}")
        print(f"  Control dimension: {config['control_dim']}")
        
        print(f"\nData statistics:")
        print(f"  Number of episodes: {len(episodes)}")
        print(f"  Number of training sequences: {len(sequences)}")
        print(f"  Average sequences per episode: {np.mean([ep['num_sequences'] for ep in episodes]):.1f}")
        
        # Analyze state distribution
        if sequences:
            all_states = np.array([seq['current_state'] for seq in sequences])
            all_speeds = all_states[:, 3] * 3.6  # Convert to km/h
            
            print(f"\nState distribution:")
            print(f"  X coordinate range: [{np.min(all_states[:, 0]):.1f}, {np.max(all_states[:, 0]):.1f}]")
            print(f"  Y coordinate range: [{np.min(all_states[:, 1]):.1f}, {np.max(all_states[:, 1]):.1f}]")
            print(f"  Yaw angle range: [{np.min(all_states[:, 2]):.2f}, {np.max(all_states[:, 2]):.2f}] rad")
            print(f"  Speed stats: avg {np.mean(all_speeds):.1f} km/h, max {np.max(all_speeds):.1f} km/h")
            
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
            
            # Verify that current_state is the same as the last element of hist_states
            current_matches_history = np.allclose(seq['current_state'], seq['hist_states'][-1])
            print(f"  Current state matches last history state: {current_matches_history}")
    
    def create_train_test_split(self, dataset_path, train_ratio=0.8, 
                            train_output='vehicle_train_dataset.pkl',
                            test_output='vehicle_test_dataset.pkl'):
        """Create train/test dataset split"""
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        sequences = dataset['training_sequences']
        episodes = dataset['episode_info']
        
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
        
        # Save datasets
        with open(train_output, 'wb') as f:
            pickle.dump(train_dataset, f)
        
        with open(test_output, 'wb') as f:
            pickle.dump(test_dataset, f)
        
        print(f"\nDataset split completed:")
        print(f"Training set: {len(train_sequences)} sequences, {len(train_episodes)} episodes -> {train_output}")
        print(f"Test set: {len(test_sequences)} sequences, {len(test_episodes)} episodes -> {test_output}")
        
        # === 新增：详细显示episode分配情况 ===
        print(f"\n=== Episode Assignment Details ===")
        print(f"Split ratio: {train_ratio:.1%} train / {1-train_ratio:.1%} test")
        print(f"Random seed: 42")
        
        # 按session分组显示
        from collections import defaultdict
        train_by_session = defaultdict(list)
        test_by_session = defaultdict(list)
        
        # 收集训练集episode信息
        for episode in train_episodes:
            session_name = episode['session_name']
            episode_name = episode['episode_name']
            num_seq = episode['num_sequences']
            train_by_session[session_name].append((episode_name, num_seq))
        
        # 收集测试集episode信息  
        for episode in test_episodes:
            session_name = episode['session_name']
            episode_name = episode['episode_name']
            num_seq = episode['num_sequences']
            test_by_session[session_name].append((episode_name, num_seq))
        
        # 显示训练集
        print(f"\nTRAINING SET ({len(train_episodes)} episodes):")
        train_total_sequences = 0
        for session_name in sorted(train_by_session.keys()):
            episodes_info = train_by_session[session_name]
            session_sequences = sum(num_seq for _, num_seq in episodes_info)
            train_total_sequences += session_sequences
            
            print(f"  Session: {session_name} ({len(episodes_info)} episodes, {session_sequences} sequences)")
            for episode_name, num_seq in sorted(episodes_info):
                print(f"    - {episode_name}: {num_seq} sequences")
        
        print(f"  Training total: {train_total_sequences} sequences")
        
        # 显示测试集
        print(f"\nTEST SET ({len(test_episodes)} episodes):")
        test_total_sequences = 0
        for session_name in sorted(test_by_session.keys()):
            episodes_info = test_by_session[session_name]
            session_sequences = sum(num_seq for _, num_seq in episodes_info)
            test_total_sequences += session_sequences
            
            print(f"  Session: {session_name} ({len(episodes_info)} episodes, {session_sequences} sequences)")
            for episode_name, num_seq in sorted(episodes_info):
                print(f"    - {episode_name}: {num_seq} sequences")
        
        print(f"  Test total: {test_total_sequences} sequences")
        
        # 显示统计摘要
        print(f"\nSPLIT SUMMARY:")
        print(f"  Total episodes: {len(episodes)} -> Train: {len(train_episodes)}, Test: {len(test_episodes)}")
        print(f"  Total sequences: {len(sequences)} -> Train: {len(train_sequences)}, Test: {len(test_sequences)}")
        print(f"  Actual split ratio: {len(train_episodes)/len(episodes):.1%} train / {len(test_episodes)/len(episodes):.1%} test")
        print(f"  Sequence ratio: {len(train_sequences)/len(sequences):.1%} train / {len(test_sequences)/len(sequences):.1%} test")
        
        # 验证数据完整性
        assert len(train_sequences) + len(test_sequences) == len(sequences), "Sequence count mismatch!"
        assert len(train_episodes) + len(test_episodes) == len(episodes), "Episode count mismatch!"
        print(f"  Data integrity verified")
        
        return train_dataset, test_dataset

def main():
    """Data processing main function"""
    parser = argparse.ArgumentParser(description='CARLA vehicle data processing')
    parser.add_argument('--data_root', default='collected_data', help='CARLA data root directory')
    parser.add_argument('--output', default='vehicle_lstm_dataset.pkl', help='Output dataset path')
    parser.add_argument('--max_episodes', type=int, default=None, help='Maximum episodes to process')
    parser.add_argument('--sampling_freq', type=float, default=20.0, help='Sampling frequency (Hz)')
    parser.add_argument('--history_time', type=float, default=3.0, help='History time window (seconds)')
    parser.add_argument('--predict_time', type=float, default=5.0, help='Prediction time window (seconds)')
    parser.add_argument('--create_split', action='store_true', default=True, help='Create train/test split')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    
    args = parser.parse_args()
    
    print("=== CARLA Vehicle Data Processing Tool ===")
    print(f"Data root directory: {args.data_root}")
    print(f"Output path: {args.output}")
    
    # Create data processor
    processor = VehicleDataProcessor(
        sampling_freq=args.sampling_freq,
        history_time=args.history_time,
        predict_time=args.predict_time
    )
    
    # Process data
    dataset = processor.process_carla_data(
        data_root=args.data_root,
        output_path=args.output,
        max_episodes=args.max_episodes
    )
    
    # Create train/test split
    if args.create_split and dataset:
        processor.create_train_test_split(
            dataset_path=args.output,
            train_ratio=args.train_ratio
        )
    
    print("\nData processing completed!")

if __name__ == "__main__":
    main()