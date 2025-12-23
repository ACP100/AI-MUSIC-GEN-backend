# import re

# def extract_tokens(text):
#     """Extract tokens from text"""
#     tokens = re.findall(r'\[([A-Z_]+)\]', text)
#     return tokens

# def process_transformer_output(transformer_output):
#     """Process transformer output and extract REMI tokens with better formatting"""
#     # Remove emotion and confidence
#     cleaned = re.sub(r'Emotion:\s*\w+\s*Confidence:\s*[\d.]+', '', transformer_output)
    
#     # Extract all potential tokens - more comprehensive approach
#     tokens = cleaned.split()
    
#     # Filter for valid REMI-like tokens
#     remi_pattern = r'^(Bar_\w+|TimeSig_\d+/\d+|Position_\d+|Tempo_[\d.]+|Program_\d+|Pitch_\d+|Velocity_\d+|Duration_[\d.]+(?:\.\d+)?|INSTRUMENT_\w+|GENRE_\w+|KEY_\w+)$'
#     remi_tokens = [token for token in tokens if re.match(remi_pattern, token)]
    
#     # If we don't have enough tokens, create some basic ones
#     if len(remi_tokens) < 5:
#         print("⚠️ Few REMI tokens found, adding basic structure")
#         basic_tokens = [
#             "Bar_Start", "TimeSig_4/4", "Tempo_120.0", 
#             "Position_0", "Program_0", "Pitch_60", "Velocity_80", "Duration_1.0",
#             "Position_480", "Program_0", "Pitch_64", "Velocity_80", "Duration_1.0"
#         ]
#         remi_tokens.extend(basic_tokens)
    
#     return ' '.join(remi_tokens)




import re

def extract_tokens(text):
    """Extract tokens from text"""
    tokens = re.findall(r'\[([A-Z_]+)\]', text)
    return tokens

def clean_transformer_output(transformer_output):
    """
    Clean transformer output by:
    1. Removing everything before and including <SEP>
    2. Extracting only valid REMI tokens
    """
    # Find the <SEP> token and get everything after it
    if '<SEP>' in transformer_output:
        # Split at <SEP> and take everything after it
        parts = transformer_output.split('<SEP>', 1)
        if len(parts) > 1:
            cleaned = parts[1].strip()
        else:
            cleaned = transformer_output
    else:
        cleaned = transformer_output
    
    # Remove any remaining control tokens that aren't valid REMI tokens
    # Valid REMI tokens: Bar_, TimeSig_, Position_, Tempo_, Program_, Pitch_, Velocity_, Duration_
    # Also remove emotion and confidence lines
    lines = []
    for line in cleaned.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Skip emotion/confidence lines
        if line.lower().startswith('emotion_') or line.lower().startswith('confidence_'):
            continue
        
        # Skip genre/instrument/tempo control tokens in brackets
        if re.search(r'^\[[A-Z_]+_[A-Z0-9_]+\]$', line):
            continue
        
        lines.append(line)
    
    return ' '.join(lines)

def process_transformer_output(transformer_output):
    """Process transformer output and extract clean REMI tokens"""
    # First clean the output
    cleaned = clean_transformer_output(transformer_output)
    
    # Split into tokens
    tokens = cleaned.split()
    
    # Filter for valid REMI tokens with proper patterns
    valid_tokens = []
    
    for token in tokens:
        token = token.strip()
        
        # Check if it's a valid REMI token
        if re.match(r'^(Bar_(?:Start|End|None))$', token):
            valid_tokens.append(token)
        elif re.match(r'^TimeSig_\d+/\d+$', token):
            valid_tokens.append(token)
        elif re.match(r'^Position_\d+$', token):
            valid_tokens.append(token)
        elif re.match(r'^Tempo_[\d.]+$', token):
            valid_tokens.append(token)
        elif re.match(r'^Program_\d+$', token):
            valid_tokens.append(token)
        elif re.match(r'^Pitch_\d+$', token):
            valid_tokens.append(token)
        elif re.match(r'^Velocity_\d+$', token):
            valid_tokens.append(token)
        elif re.match(r'^Duration_[\d.]+$', token):
            valid_tokens.append(token)
        elif token in ['Bar_Start', 'Bar_End', 'Bar_None']:
            valid_tokens.append(token)
    
    # If we don't have enough tokens, add some basic structure
    if len(valid_tokens) < 10:
        print(f"⚠️ Only {len(valid_tokens)} valid REMI tokens found, adding basic structure")
        basic_tokens = [
            "Bar_Start", "TimeSig_4/4", "Position_0", "Tempo_120.0",
            "Program_0", "Pitch_60", "Velocity_80", "Duration_1.0",
            "Position_480", "Program_0", "Pitch_64", "Velocity_80", "Duration_1.0",
            "Bar_End"
        ]
        valid_tokens.extend(basic_tokens)
    
    print(f"✅ Extracted {len(valid_tokens)} REMI tokens")
    if valid_tokens:
        print(f"First 10 tokens: {valid_tokens[:10]}")
    
    return ' '.join(valid_tokens)

def tokens_to_remi_format(tokens):
    """Convert tokens to properly formatted REMI text"""
    return ' '.join(tokens)