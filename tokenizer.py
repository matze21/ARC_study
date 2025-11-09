from typing import List, Dict, Tuple

class ARCTokenizer:
    """Tokenizer for ARC challenges with special tokens for structure"""
    
    def __init__(self):
        # Value tokens (0-9)
        self.value_tokens = list(range(10))
        
        # Special tokens
        self.PAD_TOKEN = 10
        self.SOS_TOKEN = 11  # Start of sequence
        self.EOS_TOKEN = 12  # End of sequence
        self.TRAIN_TOKEN = 13  # Start of training example
        self.TEST_TOKEN = 14  # Start of test example
        self.INPUT_TOKEN = 15  # Start of input grid
        self.OUTPUT_TOKEN = 16  # Start of output grid
        self.NEWLINE_TOKEN = 17  # Grid separator (], [)
        
        self.vocab_size = 18
        
        # Token mappings
        self.token_to_id = {
            'PAD': self.PAD_TOKEN,
            'SOS': self.SOS_TOKEN,
            'EOS': self.EOS_TOKEN,
            'TRAIN': self.TRAIN_TOKEN,
            'TEST': self.TEST_TOKEN,
            'INPUT': self.INPUT_TOKEN,
            'OUTPUT': self.OUTPUT_TOKEN,
            'NEWLINE': self.NEWLINE_TOKEN
        }
    
    def grid_to_tokens(self, grid: List[List[int]]) -> List[int]:
        """Convert 2D grid to token sequence"""
        if not grid or not grid[0]:
            return []
        
        tokens = []
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                tokens.append(value)  # Just the value, position will be encoded separately
            if i < len(grid) - 1:  # Add newline between rows (except last)
                tokens.append(self.NEWLINE_TOKEN)
        
        return tokens
    
    def tokens_to_grid(self, tokens: List[int], target_shape: Tuple[int, int]) -> List[List[int]]:
        """Convert token sequence back to 2D grid"""
        h, w = target_shape
        grid = [[0 for _ in range(w)] for _ in range(h)]
        
        # Filter out special tokens and newlines
        values = [t for t in tokens if t < 10]  # Only keep value tokens (0-9)
        
        idx = 0
        for i in range(h):
            for j in range(w):
                if idx < len(values):
                    grid[i][j] = values[idx]
                    idx += 1
        
        return grid
    
    def create_input_sequence(self, train_examples: List[Dict], test_input: List[List[int]]) -> List[int]:
        """Create input sequence from training examples and test input"""
        sequence = [self.SOS_TOKEN]
        
        # Add training examples (exactly 2)
        for i, example in enumerate(train_examples[:2]):
            sequence.append(self.TRAIN_TOKEN)
            
            # Add input
            sequence.append(self.INPUT_TOKEN)
            input_tokens = self.grid_to_tokens(example['input'])
            sequence.extend(input_tokens)
            
            # Add output
            sequence.append(self.OUTPUT_TOKEN)
            output_tokens = self.grid_to_tokens(example['output'])
            sequence.extend(output_tokens)
        
        # Add test input
        sequence.append(self.TEST_TOKEN)
        sequence.append(self.INPUT_TOKEN)
        test_tokens = self.grid_to_tokens(test_input)
        sequence.extend(test_tokens)
        
        return sequence
    
    def create_target_sequence(self, target_grid: List[List[int]]) -> List[int]:
        """Create target sequence for training"""
        sequence = [self.SOS_TOKEN]
        sequence.append(self.OUTPUT_TOKEN)
        target_tokens = self.grid_to_tokens(target_grid)
        sequence.extend(target_tokens)
        sequence.append(self.EOS_TOKEN)
        return sequence
    
    def pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """Pad sequence to max_length"""
        if len(sequence) > max_length:
            return sequence[:max_length]
        return sequence + [self.PAD_TOKEN] * (max_length - len(sequence))


