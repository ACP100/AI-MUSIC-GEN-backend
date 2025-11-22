

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

tokenizer = REMI(config)


# ---------------- CLEANING (milder than your old version) ---------------- #
def clean_tokens(tokens):
    """
    Clean REMI tokens but DON'T REMOVE valid ones.
    Only remove obvious garbage characters.
    """
    cleaned = []
    for t in tokens:
        t = t.strip()

        # Remove brackets like: [GENRE_JAZZ]
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1]

        # Remove accidental commas or newlines
        t = t.replace(",", "").replace("\n", "").strip()

        if t:
            cleaned.append(t)

    return cleaned


# -------------------------------------------------------------------------- #
# ------------------------- MAIN CONVERSION LOGIC -------------------------- #
# -------------------------------------------------------------------------- #
def convert_remi_to_midi(input_txt_path: str, output_midi_path: str):
    try:
        txt_path = Path(input_txt_path)

        if not txt_path.exists():
            raise FileNotFoundError(f"❌ REMI token file not found: {input_txt_path}")

        # Load tokens
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        tokens = raw_content.split()
        print(f"🔧 Raw token count: {len(tokens)}")

        # Clean tokens, but do NOT validate with regex aggressively
        tokens = clean_tokens(tokens)
        print(f"🔧 Cleaned token count: {len(tokens)}")
        print(f"🔧 First few tokens: {tokens[:10]}")

        if len(tokens) < 5:
            raise ValueError("❌ Not enough REMI tokens to build MIDI")

        # Convert to MIDI using miditok
        print("🎵 Converting REMI → MIDI...")
        midi_obj = tokenizer.tokens_to_midi(tokens)
        midi_obj.dump_midi(output_midi_path)

        print(f"✅ MIDI saved: {output_midi_path}")
        return True

    except Exception as e:
        print("\n❌ ERROR in convert_remi_to_midi:")
        print(e)
        print("📄 File content preview:")
        print(raw_content[:300] if 'raw_content' in locals() else "No content loaded")
        raise e



# -------------------------------------------------------------------------- #
# ----------------------- BATCH CONVERSION FOR TESTS ----------------------- #
# -------------------------------------------------------------------------- #
def batch_convert_folder(input_folder="7.REMI", output_folder="MIDI_CHECK"):
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

    print(f"\n🎵 All MIDIs reconstructed in: {output_folder}")


if __name__ == "__main__":
    batch_convert_folder("7.REMI1", "MIDI_CHECK1")
