import re

def extract_tokens(text):
    """Extract tokens from text"""
    tokens = re.findall(r'\[([A-Z_]+)\]', text)
    return tokens

def process_transformer_output(transformer_output):
    """Process transformer output and extract REMI tokens"""
    # Remove emotion and confidence
    cleaned = re.sub(r'Emotion:\s*\w+\s*Confidence:\s*[\d.]+', '', transformer_output)
    
    # Remove token brackets but keep the content
    cleaned = re.sub(r'\[([A-Z_]+)\]', r'\1', cleaned)
    
    # Extract REMI-like tokens (Bar_, TimeSig_, Position_, Tempo_, Program_, Pitch_, Velocity_, Duration_)
    remi_pattern = r'(Bar_\w+|TimeSig_\d+/\d+|Position_\d+|Tempo_[\d.]+|Program_\d+|Pitch_\d+|Velocity_\d+|Duration_[\d.]+(?:\.[\d]+)?)'
    remi_tokens = re.findall(remi_pattern, cleaned)
    
    return ' '.join(remi_tokens)