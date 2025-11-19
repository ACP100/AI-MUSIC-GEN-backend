import re

def extract_tokens(text):
    """Extract tokens from text"""
    tokens = re.findall(r'\[([A-Z_]+)\]', text)
    return tokens

def process_transformer_output(transformer_output):
    """Process transformer output and extract REMI tokens with better formatting"""
    # Remove emotion and confidence
    cleaned = re.sub(r'Emotion:\s*\w+\s*Confidence:\s*[\d.]+', '', transformer_output)
    
    # Extract all potential tokens - more comprehensive approach
    tokens = cleaned.split()
    
    # Filter for valid REMI-like tokens
    remi_pattern = r'^(Bar_\w+|TimeSig_\d+/\d+|Position_\d+|Tempo_[\d.]+|Program_\d+|Pitch_\d+|Velocity_\d+|Duration_[\d.]+(?:\.\d+)?|INSTRUMENT_\w+|GENRE_\w+|KEY_\w+)$'
    remi_tokens = [token for token in tokens if re.match(remi_pattern, token)]
    
    # If we don't have enough tokens, create some basic ones
    if len(remi_tokens) < 5:
        print("⚠️ Few REMI tokens found, adding basic structure")
        basic_tokens = [
            "Bar_Start", "TimeSig_4/4", "Tempo_120.0", 
            "Position_0", "Program_0", "Pitch_60", "Velocity_80", "Duration_1.0",
            "Position_480", "Program_0", "Pitch_64", "Velocity_80", "Duration_1.0"
        ]
        remi_tokens.extend(basic_tokens)
    
    return ' '.join(remi_tokens)