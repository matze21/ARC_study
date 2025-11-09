## Token to 3D Vector Converter
from typing import List, Tuple, Optional
import torch
from tokenizer import ARCTokenizer

class TokenTo3DConverter:
    """Converts token sequences to 3D vectors [value, x, y] with coordinate information"""
    
    def __init__(self, tokenizer: ARCTokenizer):
        self.tokenizer = tokenizer
    
    def tokens_to_3d(self, 
                     tokens: List[int],
                     input_dims: List[Tuple[int, int]],
                     output_dims: List[Tuple[int, int]],
                     test_input_dims: Tuple[int, int],
                     test_output_dims: Optional[Tuple[int, int]] = None,
                     is_target: bool = False) -> torch.Tensor:
        """
        Convert token sequence to 3D vectors [value, x, y]
        
        Args:
            tokens: List of token IDs
            input_dims: List of (height, width) for training input grids
            output_dims: List of (height, width) for training output grids
            test_input_dims: (height, width) for test input grid
            is_target: If True, this is a target sequence (starts with OUTPUT_TOKEN)
        
        Returns:
            Tensor of shape [seq_len, 3] where each row is [value, x, y]
            Special tokens have x=-1, y=-1
        """
        result = []
        
        # Track current grid context
        current_grid_type = None  # 'train_input', 'train_output', 'test_input'
        current_grid_idx = 0
        current_row = 0
        current_col = 0
        current_grid_dims = None
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Handle special tokens that change context
            if token == self.tokenizer.SOS_TOKEN:
                result.append([token, -1, -1])
                i += 1
                continue
            elif token == self.tokenizer.EOS_TOKEN:
                result.append([token, -1, -1])
                i += 1
                continue
            elif token == self.tokenizer.PAD_TOKEN:
                result.append([token, -1, -1])
                i += 1
                continue
            elif token == self.tokenizer.TRAIN_TOKEN:
                current_grid_type = None
                current_grid_idx = 0
                result.append([token, -1, -1])
                i += 1
                continue
            elif token == self.tokenizer.TEST_TOKEN:
                current_grid_type = None
                result.append([token, -1, -1])
                i += 1
                continue
            elif token == self.tokenizer.INPUT_TOKEN:
                # Determine which input grid we're in
                if is_target:
                    # In target sequence, INPUT_TOKEN shouldn't appear
                    result.append([token, -1, -1])
                    i += 1
                    continue
                
                if current_grid_type is None:
                    # First INPUT after TRAIN - this is training input
                    if current_grid_idx < len(input_dims):
                        current_grid_dims = input_dims[current_grid_idx]
                        current_grid_type = 'train_input'
                elif current_grid_type == 'train_output':
                    # INPUT after OUTPUT in training - next training example
                    current_grid_idx += 1
                    if current_grid_idx < len(input_dims):
                        current_grid_dims = input_dims[current_grid_idx]
                        current_grid_type = 'train_input'
                elif current_grid_type is None:
                    # INPUT after TEST - this is test input
                    current_grid_dims = test_input_dims
                    current_grid_type = 'test_input'
                
                current_row = 0
                current_col = 0
                result.append([token, -1, -1])
                i += 1
                continue
            elif token == self.tokenizer.OUTPUT_TOKEN:
                # Determine which output grid we're in
                if current_grid_type == 'train_input':
                    # OUTPUT after INPUT in training
                    if current_grid_idx < len(output_dims):
                        current_grid_dims = output_dims[current_grid_idx]
                        current_grid_type = 'train_output'
                elif current_grid_type is None:
                    # OUTPUT at start (for target sequence) or after TEST
                    if is_target:
                        # For target sequence, use test_output_dims if available
                        if test_output_dims is not None:
                            current_grid_dims = test_output_dims
                        elif len(output_dims) > 0:
                            current_grid_dims = output_dims[0]  # Fallback to first output dims
                        else:
                            current_grid_dims = (1, 1)  # Default fallback
                    elif len(output_dims) > 0:
                        current_grid_dims = output_dims[0]
                    current_grid_type = 'train_output'
                
                current_row = 0
                current_col = 0
                result.append([token, -1, -1])
                i += 1
                continue
            elif token == self.tokenizer.NEWLINE_TOKEN:
                # Move to next row
                if current_grid_dims is not None:
                    current_row += 1
                    current_col = 0
                result.append([token, -1, -1])
                i += 1
                continue
            elif token < 10:  # Value token (0-9)
                # This is a grid value - add coordinates
                if current_grid_dims is not None:
                    h, w = current_grid_dims
                    # Clamp to valid ranges
                    row = min(current_row, h - 1)
                    col = min(current_col, w - 1)
                    result.append([token, col, row])  # x=col, y=row
                    
                    # Move to next column
                    current_col += 1
                else:
                    # No grid context, treat as special
                    result.append([token, -1, -1])
                i += 1
            else:
                # Unknown token, treat as special
                result.append([token, -1, -1])
                i += 1
        
        return torch.tensor(result, dtype=torch.long)


