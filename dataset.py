import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from tokenizer import ARCTokenizer
from token_converter import TokenTo3DConverter


class ARCDataset:
    """Dataset class for ARC challenges with data augmentation"""
    
    def __init__(self, challenges_path: str, solutions_path: str = None):
        self.challenges_path = challenges_path
        self.solutions_path = solutions_path
        
        # Load challenges
        with open(challenges_path, 'r') as f:
            self.challenges = json.load(f)
        
        # Load solutions if provided
        self.solutions = None
        if solutions_path:
            with open(solutions_path, 'r') as f:
                self.solutions = json.load(f)
    
    def get_challenge_data(self, challenge_id: str) -> Dict:
        """Get data for a specific challenge"""
        challenge = self.challenges[challenge_id]
        
        # Get training examples
        train_examples = challenge.get('train', [])
        
        # Get test examples
        test_examples = challenge.get('test', [])
        
        # Get solution if available
        solution = None
        if self.solutions and challenge_id in self.solutions:
            solution = self.solutions[challenge_id][0]  # First solution
        
        return {
            'train_examples': train_examples,
            'test_examples': test_examples,
            'solution': solution,
            'challenge_id': challenge_id
        }
    
    def get_all_challenges(self) -> List[str]:
        """Get list of all challenge IDs"""
        return list(self.challenges.keys())
    
    def create_augmented_samples(self, challenge_id: str) -> List[Dict]:
        """Create augmented training samples from a challenge"""
        data = self.get_challenge_data(challenge_id)
        
        # For training data, we can create augmented samples using the original train examples
        # and the test examples (which don't have outputs, so we can't use them for training)
        samples = []
        
        # Use original training examples (these have both input and output)
        if len(data['train_examples']) >= 2:
            # Create sample with first 2 training examples
            train_examples = data['train_examples'][:2]
            test_input = data['test_examples'][0]['input'] if data['test_examples'] else []
            test_output = data['solution']
            
            samples.append({
                'train_examples': train_examples,
                'test_input': test_input,
                'test_output': test_output,
                'challenge_id': challenge_id,
                'sample_id': f"{challenge_id}_orig"
            })
            
            # If we have more training examples, create additional samples
            if len(data['train_examples']) >= 4:
                # Use examples 2 and 3 as training
                train_examples = data['train_examples'][2:4]
                samples.append({
                    'train_examples': train_examples,
                    'test_input': test_input,
                    'test_output': test_output,
                    'challenge_id': challenge_id,
                    'sample_id': f"{challenge_id}_aug_0"
                })
            
            # If we have even more, use examples 1 and 3
            if len(data['train_examples']) >= 4:
                train_examples = [data['train_examples'][1], data['train_examples'][3]]
                samples.append({
                    'train_examples': train_examples,
                    'test_input': test_input,
                    'test_output': test_output,
                    'challenge_id': challenge_id,
                    'sample_id': f"{challenge_id}_aug_1"
                })
        
        return samples

class ARCTorchDataset(Dataset):
    """PyTorch Dataset for ARC challenges"""
    
    def __init__(self, arc_dataset: ARCDataset, tokenizer: ARCTokenizer, 
                 token_converter = None):  # Optional converter
        self.arc_dataset = arc_dataset
        self.tokenizer = tokenizer
        self.token_converter = token_converter  # Optional converter for 3D vectors
        
        # Create all samples with augmentation
        self.samples = []
        for challenge_id in arc_dataset.get_all_challenges():
            samples = arc_dataset.create_augmented_samples(challenge_id)
            self.samples.extend(samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Create input sequence
        input_seq = self.tokenizer.create_input_sequence(
            sample['train_examples'], 
            sample['test_input']
        )
        
        # Create target sequence
        if sample['test_output']:
            target_seq = self.tokenizer.create_target_sequence(sample['test_output'])
        else:
            # Create dummy target for test data
            target_seq = [self.tokenizer.SOS_TOKEN, self.tokenizer.EOS_TOKEN]
        
        # Pad sequences
        input_seq = self.tokenizer.pad_sequence(input_seq, 5400)   # 5 * 30x30 + bunch of extra tokens + possible target 30x30= 6*30x30
        target_seq = self.tokenizer.pad_sequence(target_seq, 1000) # max 30x30 + punch of extra tokens
        
        # Calculate dimensions
        input_dims = []
        output_dims = []
        
        for example in sample['train_examples']:
            input_dims.append((len(example['input']), len(example['input'][0]) if example['input'] else 0))
            output_dims.append((len(example['output']), len(example['output'][0]) if example['output'] else 0))
        
        test_input_dims = (len(sample['test_input']), len(sample['test_input'][0]) if sample['test_input'] else 0)
        test_output_dims = (len(sample['test_output']), len(sample['test_output'][0]) if sample['test_output'] else 0)
        
        # Convert to 3D vectors if converter is provided
        if self.token_converter is not None:
            input_3d = self.token_converter.tokens_to_3d(
                input_seq,
                input_dims,
                output_dims,
                test_input_dims,
                test_output_dims=test_output_dims,
                is_target=False
            )
            target_3d = self.token_converter.tokens_to_3d(
                target_seq,
                input_dims,
                output_dims,
                test_input_dims,
                test_output_dims=test_output_dims,
                is_target=True
            )
            return {
                'input': input_3d,  # Shape: [seq_len, 3] - [value, x, y]
                'target': target_3d,  # Shape: [seq_len, 3] - [value, x, y]
                'input_tokens': torch.tensor(input_seq, dtype=torch.int8),  # Keep original tokens too (int8 for memory efficiency)
                'target_tokens': torch.tensor(target_seq, dtype=torch.int8),  # Keep original tokens too (int8 for memory efficiency)
                'sample_id': sample['sample_id'],
                'challenge_id': sample['challenge_id'],
                'input_dims': input_dims,
                'output_dims': output_dims,
                'test_input_dims': test_input_dims,
                'test_output_dims': test_output_dims
            }
        else:
            # Return original token format
            return {
                'input': torch.tensor(input_seq, dtype=torch.int8),  # int8 for memory efficiency
                'target': torch.tensor(target_seq, dtype=torch.int8),  # int8 for memory efficiency
                'sample_id': sample['sample_id'],
                'challenge_id': sample['challenge_id'],
                'input_dims': input_dims,
                'output_dims': output_dims,
                'test_input_dims': test_input_dims,
                'test_output_dims': test_output_dims
            }


