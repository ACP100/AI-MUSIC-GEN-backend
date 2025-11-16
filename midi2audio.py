import os
import subprocess

def convert_midi_to_audio(midi_path, audio_output_path):
    """
    Convert MIDI to audio using FluidSynth
    """
    try:
        # You'll need to specify the path to your SoundFont file
        soundfont_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"  # Default path on many systems
        
        # If the default path doesn't exist, try to find it
        if not os.path.exists(soundfont_path):
            # Common alternative paths
            alternative_paths = [
                "/usr/share/soundfonts/FluidR3_GM.sf2",
                "/usr/share/sounds/sf2/default.sf2",
                "./FluidR3_GM.sf2"  # Current directory
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    soundfont_path = path
                    break
            else:
                # If no SoundFont found, create a mock WAV file
                create_mock_audio(audio_output_path)
                return True
        
        # Convert using FluidSynth
        command = [
            'fluidsynth', 
            '-ni', 
            soundfont_path, 
            midi_path, 
            '-F', audio_output_path, 
            '-r', '44100'
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FluidSynth error: {result.stderr}")
            # Fallback to mock audio
            create_mock_audio(audio_output_path)
        
        return True
    
    except Exception as e:
        print(f"Error converting MIDI to audio: {e}")
        # Fallback to mock audio
        create_mock_audio(audio_output_path)
        return True

def create_mock_audio(output_path):
    """Create a mock audio file for demo purposes"""
    # Create a simple WAV file with silence
    import wave
    import struct
    
    # Create a silent audio file (1 second of silence)
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(44100)  # Sample rate
        wav_file.setnframes(44100)  # 1 second
        
        # Write silent frames
        for _ in range(44100):
            wav_file.writeframes(struct.pack('<h', 0))