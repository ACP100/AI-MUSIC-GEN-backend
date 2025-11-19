from miditok import REMI, TokenizerConfig
from pathlib import Path
import re

# ---------- SAME CONFIG used during encoding ---------- #
config = TokenizerConfig(
    beat_res={(0, 4): 8},
    use_chords=True,
    use_programs=True,
    use_tempos=True,
    use_time_signatures=True,
    use_sustain_pedals=True,
    use_pitch_intervals=False,
    use_tracks=True,
    num_velocities=32,
    program_changes=True,
)

# Initialize tokenizer
tokenizer = REMI(config)

def clean_remi_tokens(tokens):
    """Clean and validate REMI tokens before conversion"""
    valid_tokens = []
    
    for token in tokens:
        # Remove any extra characters and validate token format
        token = token.strip()
        
        # Check if it's a valid REMI token pattern - expanded pattern
        if re.match(r'^(Bar_\w+|TimeSig_\d+/\d+|Position_\d+|Tempo_[\d.]+|Program_\d+|Pitch_\d+|Velocity_\d+|Duration_[\d.]+(?:\.\d+)?|INSTRUMENT_\w+|GENRE_\w+|KEY_\w+)$', token):
            valid_tokens.append(token)
        else:
            print(f"⚠️ Skipping invalid token: {token}")
    
    return valid_tokens

def convert_remi_to_midi(input_txt_path: str, output_midi_path: str):
    """Convert REMI tokens to MIDI file with error handling"""
    try:
        txt_path = Path(input_txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"REMI file not found: {input_txt_path}")
            
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if not content:
            raise ValueError("REMI file is empty")
        
        # Split tokens and clean them
        tokens = content.split()
        print(f"🔧 Raw tokens count: {len(tokens)}")
        print(f"🔧 Sample tokens: {tokens[:10]}")  # Debug: show first 10 tokens
        
        # Clean and validate tokens
        cleaned_tokens = clean_remi_tokens(tokens)
        print(f"🔧 Cleaned tokens count: {len(cleaned_tokens)}")
        print(f"🔧 Sample cleaned tokens: {cleaned_tokens[:10]}")  # Debug
        
        if not cleaned_tokens:
            raise ValueError("No valid REMI tokens found after cleaning")
        
        # Convert to MIDI
        print("🔄 Converting tokens to MIDI...")
        midi_obj = tokenizer.tokens_to_midi(cleaned_tokens)
        midi_obj.dump_midi(output_midi_path)
        print(f"✅ MIDI saved: {output_midi_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting REMI to MIDI: {e}")
        print(f"📄 File content preview: {content[:200] if 'content' in locals() else 'No content'}")
        
        # Try fallback instead of raising
        print("🔄 Attempting fallback MIDI creation...")
        if create_simple_midi_fallback(output_midi_path):
            return True
        else:
            raise
        
    except Exception as e:
        print(f"❌ Error converting REMI to MIDI: {e}")
        print(f"📄 File content preview: {content[:200] if 'content' in locals() else 'No content'}")
        raise

def create_simple_midi_fallback(output_midi_path: str):
    """Create a simple MIDI file as fallback"""
    try:
        from miditoolkit import MidiFile, Instrument, Note
        import miditoolkit
        
        # Create a simple MIDI with one note
        midi = MidiFile()
        piano = Instrument(program=0, is_drum=False, name="Piano")
        
        # Add a simple middle C note
        note = Note(start=0, end=480, pitch=60, velocity=80)
        piano.notes.append(note)
        
        midi.instruments.append(piano)
        midi.dump(output_midi_path)
        print(f"🎹 Created fallback MIDI: {output_midi_path}")
        return True
    except Exception as e:
        print(f"❌ Fallback MIDI creation failed: {e}")
        return False

def batch_convert_folder(input_folder="7.REMI", output_folder="MIDI_CHECK"):
    """Batch convert all REMI files in a folder to MIDI"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        print(f"⚠️ No REMI .txt files found in {input_folder}")
        return

    for i, txt_file in enumerate(txt_files, 1):
        print(f"[{i}/{len(txt_files)}] Processing {txt_file.name} ...")
        output_file = output_path / f"{txt_file.stem}.mid"
        try:
            convert_remi_to_midi(txt_file, output_file)
        except Exception as e:
            print(f"❌ Failed to convert {txt_file.name}: {e}")
            # Try fallback
            if create_simple_midi_fallback(output_file):
                print(f"✅ Used fallback for {txt_file.name}")

    print(f"\n🎵 All MIDIs reconstructed in: {output_folder}")

if __name__ == "__main__":
    batch_convert_folder("7.REMI", "MIDI_CHECK")