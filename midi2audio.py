import os
import subprocess
import wave
import struct


def convert_midi_to_audio(midi_path, audio_output_path, use_fallback=True):
    """
    Convert MIDI to audio using FluidSynth.
    If FluidSynth or SoundFont is missing, optionally fallback to generating 1 second of silence.
    """

    # --- 1. Locate SoundFont ----------------------------------------------------
    soundfont_candidates = [
        # "./FluidR3_GM.sf2",
        # "./MuseScore_General.sf2",
        # "./MuseScore_General.sf3",
        # "./Arachno.sf2",
        # "./GeneralUserGS.sf3",
        "./GeneralUser-GS.sf2"

    ]

    soundfont_path = next((p for p in soundfont_candidates if os.path.exists(p)), None)

    if soundfont_path is None:
        print("❌ No SoundFont (.sf2) found on your system.")
        if use_fallback:
            print("⚠️ Using fallback silent audio.")
            create_mock_audio(audio_output_path)
            return False
        else:
            raise FileNotFoundError("No SoundFont found for FluidSynth")

    # --- 2. Build correct FluidSynth command (ORDER MATTERS) --------------------
    command = [
        "fluidsynth",
        "-F", audio_output_path,      # output WAV
        "-ni",                        # no interactive shell
        soundfont_path,               # soundfont
        midi_path,                    # input MIDI
        "-r", "44100"                 # sample rate
    ]

    print("🎵 Running FluidSynth:")
    print(" ".join(command))

    # --- 3. Execute FluidSynth --------------------------------------------------
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ FluidSynth Error:")
        print(result.stderr)

        if use_fallback:
            print("⚠️ Falling back to 1-second silent audio.")
            create_mock_audio(audio_output_path)
            return False
        else:
            raise RuntimeError("FluidSynth failed: " + result.stderr)

    # --- 4. Verify output -------------------------------------------------------
    if not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) < 2000:
        print("❌ FluidSynth produced an empty or invalid file.")
        if use_fallback:
            print("⚠️ Using fallback silent audio.")
            create_mock_audio(audio_output_path)
            return False
        else:
            raise RuntimeError("FluidSynth output file too small")

    print("✅ Audio conversion complete:", audio_output_path)
    return True



# -----------------------------------------------------------------------------
# FALLBACK: Create 1-second silent WAV (only used if conversion fails)
# -----------------------------------------------------------------------------

def create_mock_audio(output_path):
    """
    Create a silent 1-second WAV audio file.
    Used only when FluidSynth fails.
    """
    with wave.open(output_path, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.setnframes(44100)

        silence_frame = struct.pack("<h", 0)
        for _ in range(44100):
            wav.writeframes(silence_frame)

    print(f"🟡 Created mock audio (1 sec): {output_path}")
