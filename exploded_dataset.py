# Refactored ARCExplodedDataset - works directly with 3D vectors
from torch.utils.data import Dataset
from tokenizer import ARCTokenizer
from dataset import ARCTorchDataset
from tqdm import tqdm
import torch

class ARCExplodedDataset(Dataset):
    """
    Explodes ARCTorchDataset into trainable samples.
    
    Takes each sample from ARCTorchDataset and creates multiple training samples:
    - Sample 0: input → predict target[0]
    - Sample 1: input + target[0] → predict target[1]
    - Sample 2: input + target[0:2] → predict target[2]
    - etc.
    
    Expects both input and target to be in 3D vector format [value, x, y].
    When adding target tokens:
    1. Loop through input sequence and replace first PAD token with target token
    2. If no PAD token found, append to end and remove first token
    """
    
    def __init__(self, torch_dataset: ARCTorchDataset, tokenizer: ARCTokenizer, sequence_length: int = 5400):
        self.torch_dataset = torch_dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        
        # Create all exploded samples
        self.exploded_samples = []
        
        print(f"Exploding {len(torch_dataset)} base samples...")
        for base_idx in tqdm(range(len(torch_dataset))):
            base_sample = torch_dataset[base_idx]
            
            # Get input and target as 3D tensors [seq_len, 3]
            input_3d = base_sample['input']  # Shape: [max_length, 3]
            target_3d = base_sample['target']  # Shape: [max_length, 3]
            
            # Find actual length of input (before padding)
            # PAD token has value = PAD_TOKEN (10), x = -1, y = -1
            input_actual_len = 0
            for i in range(input_3d.shape[0]):
                if input_3d[i, 0].item() == self.tokenizer.PAD_TOKEN:
                    break
            input_actual_len = i-1
            
            target_actual_len = 0
            for i in range(target_3d.shape[0]):
                if target_3d[i, 0].item() == self.tokenizer.PAD_TOKEN:
                    break
            target_actual_len = i-1
            
            #print(input_actual_len, target_actual_len)
            
            target_vectors = target_3d[0:target_actual_len]
            
            
            # Optimized version - remove unnecessary cloning and use input_actual_len directly
            # Replace the target_vectors collection and loop in cell 9 with this:

            # In the target_vectors collection (around line 43-50):
            # Change: target_vectors.append(target_3d[i].clone())
            # To:     target_vectors.append(target_3d[i])  # No clone needed

            # In the loop (around line 58-67):
            # Replace the entire loop with this optimized version:

            # Start with full input sequence (we'll modify it in place)
            current_seq = input_3d.clone()
            for i, target_vector in enumerate(target_vectors):
                # Calculate position where we should place this target token
                # Start from input_actual_len and add i (position in target sequence)
                target_pos = input_actual_len + i
                
                if i>0:
                    # first target vector is not added to the input sequence
                    if target_pos < self.sequence_length:
                        # Check if position has a PAD token
                        if current_seq[target_pos, 0].item() == self.tokenizer.PAD_TOKEN:
                            # Replace PAD token with target vector
                            current_seq[target_pos] = target_vectors[i-1]
                        else:
                            #print("Sequence is full - append and remove from beginning", target_pos, current_seq.shape)
                            # Sequence is full - append and remove from beginning
                            current_seq = torch.cat([current_seq[1:], target_vectors[i-1].unsqueeze(0)], dim=0)
                    else:
                        current_seq = torch.cat([current_seq[1:], target_vectors[i-1].unsqueeze(0)], dim=0)

                # Store exploded sample
                exploded_sample = {
                    'input_3d': current_seq.clone(),
                    'target_vector': target_vector.clone(),  # Clone here since we store it separately
                    'target_position': i,
                    'base_sample_idx': base_idx,
                    'base_sample_id': base_sample.get('sample_id', f'sample_{base_idx}'),
                    'challenge_id': base_sample.get('challenge_id', ''),
                    'input_dims': base_sample.get('input_dims', []),
                    'output_dims': base_sample.get('output_dims', []),
                    'test_input_dims': base_sample.get('test_input_dims', (0, 0)),
                    'test_output_dims': base_sample.get('test_output_dims', (0, 0)),
                }

                self.exploded_samples.append(exploded_sample)
        
        print(f"Created {len(self.exploded_samples)} exploded samples from {len(torch_dataset)} base samples")
    
    def __len__(self):
        return len(self.exploded_samples)
    
    def __getitem__(self, idx):
        sample = self.exploded_samples[idx]
        
        input_3d = sample['input_3d']  # Shape: [max_length, 3]
        target_vector = sample['target_vector']  # Shape: [3]
        
        # Create attention mask (1 for non-padding, 0 for padding)
        attention_mask = (input_3d[:, 0] != self.tokenizer.PAD_TOKEN).long()
        
        return {
            'input_3d': input_3d,  # [max_length, 3] - full 3D vectors
            'target_vector': target_vector,  # [3] - target as 3D vector
            'target_value': target_vector[0].item(),  # Just the value token for convenience
            'attention_mask': attention_mask,  # [max_length]
            'target_position': sample['target_position'],
            'base_sample_idx': sample['base_sample_idx'],
            'base_sample_id': sample['base_sample_id'],
            'challenge_id': sample['challenge_id'],
            'input_dims': sample['input_dims'],
            'output_dims': sample['output_dims'],
            'test_input_dims': sample['test_input_dims'],
            'test_output_dims': sample['test_output_dims'],
        }

