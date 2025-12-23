# music_inference.py
"""
Complete inference module for music generation.
Can be used: 
1. Standalone: python music_inference.py "Emotion_joy Confidence_0.57 [GENRE_ELECTRONIC] ..."
2. Imported: from music_inference import run_inference, MusicGenerator
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL DEFINITION
# -----------------------------

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask=None):
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + self.attn_dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class MusicTransformer(nn.Module):
    """Main transformer model for music generation"""
    def __init__(
        self,
        num_emb,
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        max_seq_len=1024,
        dropout=0.1,
        pad_id=None,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_emb, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_id = pad_id

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(self, input_seq):
        B, L = input_seq.shape
        device = input_seq.device

        if L > self.pos_emb.num_embeddings:
            raise ValueError(f"Sequence length L={L} exceeds max_seq_len={self.pos_emb.num_embeddings}")

        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

        positions = torch.arange(L, device=device)
        x = self.token_emb(input_seq) + self.pos_emb(positions).unsqueeze(0)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, causal_mask=causal_mask)

        return self.fc_out(x)


# -----------------------------
# SAMPLING HELPERS
# -----------------------------

def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Apply top-k filtering to logits"""
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    vals, _ = torch.topk(logits, top_k, dim=-1)
    cutoff = vals[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


def encode_prompt_tokens(prompt_tokens, token_to_id):
    """
    Convert prompt tokens to IDs with special tokens
    Returns: <BOS> + prompt_tokens + <SEP>
    """
    seq = ["<BOS>", *prompt_tokens]
    if "<SEP>" not in seq:
        seq.append("<SEP>")
    unk = token_to_id["<UNK>"]
    return [token_to_id.get(t, unk) for t in seq]


@torch.no_grad()
def generate(
    model,
    prompt_tokens,
    token_to_id,
    id_to_token,
    device,
    max_new_tokens=2000,
    temperature=0.9,
    top_k=50,
    eos_token="<EOS>",
    forbid_tokens=("<PAD>",),
    progress_callback=None,
):
    """
    Generate tokens from prompt using the model
    
    Args:
        model: Loaded MusicTransformer model
        prompt_tokens: List of control tokens (e.g., ["Emotion_joy", "[GENRE_ELECTRONIC]"])
        token_to_id: Vocabulary mapping
        id_to_token: Reverse vocabulary mapping
        device: torch device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        eos_token: End-of-sequence token
        forbid_tokens: Tokens to forbid during generation
        progress_callback: Optional callback for progress updates
    
    Returns:
        List of generated tokens
    """
    model.eval()
    max_seq_len = model.pos_emb.num_embeddings

    eos_id = token_to_id[eos_token]
    forbid_ids = [token_to_id[t] for t in forbid_tokens if t in token_to_id]

    # Start with encoded prompt
    full_ids = encode_prompt_tokens(prompt_tokens, token_to_id)  # includes <BOS> ... <SEP>

    for step in range(max_new_tokens):
        # Only the last max_seq_len tokens are fed to the model
        ctx_ids = full_ids[-max_seq_len:]
        x = torch.tensor([ctx_ids], dtype=torch.long, device=device)

        logits = model(x)                 # (1, T, V)
        next_logits = logits[:, -1, :]    # (1, V)

        if temperature is not None and temperature > 0:
            next_logits = next_logits / temperature

        if forbid_ids:
            next_logits[:, forbid_ids] = float("-inf")

        next_logits = top_k_filtering(next_logits, top_k)

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        # Append to history
        full_ids.append(next_id)

        # Report progress
        if progress_callback:
            progress_callback(step + 1, max_new_tokens, len(full_ids))
        elif (step + 1) % 100 == 0:
            print(f"Generated {step+1}/{max_new_tokens} tokens... (total: {len(full_ids)})")

        # Stop if EOS token is generated
        if next_id == eos_id:
            break

    # Decode all tokens
    out_tokens = [id_to_token.get(i, "<UNK>") for i in full_ids]
    return out_tokens


# -----------------------------
# MAIN INFERENCE CLASS
# -----------------------------

class MusicGenerator:
    """Main class for loading model and generating music"""
    
    def __init__(self, model_path=None, vocab_path=None):
        """
        Initialize the music generator
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            vocab_path: Path to vocabulary (.json file)
        """
        self.device = device
        self.model = None
        self.token_to_id = None
        self.id_to_token = None
        
        # Default paths
        self.default_model_paths = [
            "music_transformer.pt",
            "F:/AI MUSIC GEN backend/music_transformer.pt"
        ]
        
        self.default_vocab_paths = [
            "vocab.json",
            "F:/AI MUSIC GEN backend/vocab.json"
        ]
        
        self.model_path = model_path
        self.vocab_path = vocab_path
        
    def load_model(self):
        """Load model and vocabulary"""
        print(f"Loading model on device: {self.device}")
        
        # Find vocabulary file
        vocab_path = self.vocab_path
        if vocab_path is None:
            for path in self.default_vocab_paths:
                if os.path.exists(path):
                    vocab_path = path
                    break
        
        if not vocab_path or not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found. Tried: {self.default_vocab_paths}")
        
        print(f"Loading vocabulary from: {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        self.token_to_id = {t: int(i) for t, i in token_to_id.items()}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        
        # Find model file
        model_path = self.model_path
        if model_path is None:
            for path in self.default_model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found. Tried: {self.default_model_paths}")
        
        print(f"Loading model from: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device)
        pad_id = self.token_to_id.get("<PAD>", None)
        
        # Create model with correct architecture
        self.model = MusicTransformer(
            num_emb=len(self.token_to_id),
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            max_seq_len=1024,
            dropout=0.0,   # inference mode
            pad_id=pad_id,
        ).to(self.device)
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
        print("✅ Model loaded successfully")
        return self
    
    def generate(
        self,
        prompt_tokens,
        max_new_tokens=2000,
        temperature=0.9,
        top_k=50,
        output_file=None,
        show_progress=True
    ):
        """
        Generate music from prompt tokens
        
        Args:
            prompt_tokens: List of control tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            output_file: Optional file to save output
            show_progress: Whether to print progress
            
        Returns:
            Generated tokens as list and string
        """
        if self.model is None or self.token_to_id is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Generating with prompt: {prompt_tokens[:5]}..." if len(prompt_tokens) > 5 else f"Generating with prompt: {prompt_tokens}")
        
        # Progress callback
        def progress_callback(current, total, full_len):
            if show_progress and current % 100 == 0:
                print(f"Generated {current}/{total} tokens... (total: {full_len})")
        
        # Generate tokens
        out_tokens = generate(
            model=self.model,
            prompt_tokens=prompt_tokens,
            token_to_id=self.token_to_id,
            id_to_token=self.id_to_token,
            device=self.device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            forbid_tokens=("<PAD>",),
            progress_callback=progress_callback if show_progress else None,
        )
        
        # Convert to string
        output_str = " ".join(out_tokens)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_str)
            print(f"✅ Output saved to: {output_file}")
        
        print(f"✅ Generation complete. Total tokens: {len(out_tokens)}")
        
        # Show REMI part (after <SEP>)
        if "<SEP>" in output_str:
            parts = output_str.split("<SEP>", 1)
            if len(parts) > 1:
                remi_part = parts[1].strip()
                if "<EOS>" in remi_part:
                    remi_part = remi_part.split("<EOS>")[0]
                print(f"\n🎵 Generated music tokens (first 300 chars):")
                print(remi_part[:300] + "..." if len(remi_part) > 300 else remi_part)
        
        return out_tokens, output_str


# -----------------------------
# CONVENIENCE FUNCTIONS
# -----------------------------

def run_inference(
    prompt_tokens,
    output_file="generated_music.txt",
    max_new_tokens=2000,
    model_path=None,
    vocab_path=None
):
    """
    One-line function to run inference
    
    Args:
        prompt_tokens: List of control tokens or space-separated string
        output_file: Path to save output
        max_new_tokens: Maximum tokens to generate
        model_path: Optional custom model path
        vocab_path: Optional custom vocab path
        
    Returns:
        Generated tokens and string
    """
    # Convert string to list if needed
    if isinstance(prompt_tokens, str):
        prompt_tokens = prompt_tokens.split()
    
    # Create generator and run
    generator = MusicGenerator(model_path=model_path, vocab_path=vocab_path)
    generator.load_model()
    
    return generator.generate(
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        output_file=output_file
    )


# -----------------------------
# COMMAND LINE INTERFACE
# -----------------------------

def main():
    """Command line entry point"""
    if len(sys.argv) < 2:
        print("""
Music Generation Inference Tool
===============================
Usage:
  python music_inference.py "<prompt_tokens>" [output_file] [max_tokens]
  
Examples:
  python music_inference.py "Emotion_joy Confidence_0.57 [GENRE_ELECTRONIC]"
  python music_inference.py "Emotion_sad [GENRE_CLASSICAL]" output.txt 1000
  
Default output: generated_music.txt
Default max tokens: 2000
        """)
        sys.exit(1)
    
    # Parse arguments
    prompt_str = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "generated_music.txt"
    max_new_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    
    print(f"Generating music...")
    print(f"Prompt: {prompt_str[:100]}..." if len(prompt_str) > 100 else f"Prompt: {prompt_str}")
    print(f"Output: {output_file}")
    print(f"Max tokens: {max_new_tokens}")
    print("-" * 50)
    
    # Run inference
    try:
        tokens, text = run_inference(
            prompt_tokens=prompt_str,
            output_file=output_file,
            max_new_tokens=max_new_tokens
        )
        print(f"\n✅ Success! Generated {len(tokens)} tokens.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()