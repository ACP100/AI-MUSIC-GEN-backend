from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import subprocess
import sys
from utils.token_processor import extract_tokens, process_transformer_output, clean_transformer_output
from lyrics_emotion import detect_emotion_from_text, load_emotion_models
from remi2midi import convert_remi_to_midi
from midi2audio import convert_midi_to_audio
from music_infrence import run_inference, MusicGenerator

app = Flask(__name__)
CORS(app)

# Create necessary directories
os.makedirs('temp_files', exist_ok=True)
os.makedirs('output/audio', exist_ok=True)
os.makedirs('output/midi', exist_ok=True)

# Load emotion models once at startup
classifiers, tokenizers, weights = load_emotion_models()

class MusicGenerator:
    def __init__(self):
        self.workflow_steps = [
            self.save_lyrics,
            self.extract_input_tokens,
            self.extract_emotion,
            self.combine_emotion_tokens,
            self.feed_transformer,
            self.process_transformer_output,
            self.convert_to_midi,
            self.convert_to_audio
        ]
    
    def save_lyrics(self, data, session_id):
        """Step 1: Save lyrics to file"""
        lyrics = data['lyrics']
        with open(f'temp_files/lyrics_{session_id}.txt', 'w', encoding='utf-8') as f:
            f.write(lyrics)
        return {"step": "lyrics_saved"}
    
    def extract_input_tokens(self, data, session_id):
        """Step 2: Extract tokens from input"""
        tokens = {
            'GENRE': data['genre'],
            'INSTRUMENTS': data['instruments'],
            'TEMPO': data['tempo'],
            'KEY': data['key']
        }
        
        # Save tokens to file
        with open(f'temp_files/tokens_{session_id}.txt', 'w', encoding='utf-8') as f:
            for key, value in tokens.items():
                if isinstance(value, list):
                    for item in value:
                        f.write(f"[{key}_{item.upper()}]\n")
                else:
                    f.write(f"[{key}_{value.upper()}]\n")
        
        return {"step": "tokens_extracted"}
    
    def extract_emotion(self, data, session_id):
        """Step 3: Extract emotion from lyrics using your emotion detection"""
        lyrics_path = f'temp_files/lyrics_{session_id}.txt'
        
        # Read lyrics
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            lyrics = f.read()
        
        # Use your emotion detection function
        emotion, confidence = detect_emotion_from_text(
            lyrics, classifiers, tokenizers, weights
        )
        
        # Save emotion
        with open(f'temp_files/emotion_{session_id}.txt', 'w', encoding='utf-8') as f:
            f.write(f"Emotion_{emotion} Confidence_{confidence:.2f}")
        
        return {
            "step": "emotion_extracted",
            "emotion": emotion,
            "confidence": confidence
        }
    
    def combine_emotion_tokens(self, data, session_id):
        """Step 4: Combine emotion and tokens"""
        # Read emotion
        with open(f'temp_files/emotion_{session_id}.txt', 'r', encoding='utf-8') as f:
            emotion_line = f.read().strip()
        
        # Read tokens
        with open(f'temp_files/tokens_{session_id}.txt', 'r', encoding='utf-8') as f:
            tokens_lines = f.readlines()
        
        # Combine - create prompt for transformer
        # Format: Emotion_joy Confidence_0.57 [GENRE_ELECTRONIC] [INSTRUMENT_SYNTH_LEAD] ...
        combined_content = emotion_line + ' ' + ''.join(tokens_lines).replace('\n', ' ')
        
        with open(f'temp_files/emotion_tokens_{session_id}.txt', 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        return {"step": "emotion_tokens_combined"}
    

    def feed_transformer(self, data, session_id):
        with open(f'temp_files/emotion_tokens_{session_id}.txt', 'r', encoding='utf-8') as f:
            emotion_tokens = f.read().strip()
    
    # Parse tokens
        prompt_tokens = []
        parts = emotion_tokens.split()
    
        for part in parts:
            part = part.strip()
            if  part:
            # Clean up
                if part.startswith('"') and part.endswith('"'):
                    part = part[1:-1]
                elif part.startswith("'") and part.endswith("'"):
                    part = part[1:-1]
            
            # Only add valid tokens
                if any(x in part for x in ['Emotion_', 'Confidence_', '[', ']']):
                    prompt_tokens.append(part)
    
        print(f"🎵 Running inference with {len(prompt_tokens)} prompt tokens")
        print(f"Sample tokens: {prompt_tokens[:5]}")
    
        output_file = f'temp_files/transformer_output_{session_id}.txt'
    
        try:
        # Use the single-file inference module I provided
            from music_infrence import run_inference
        
            print("📊 Starting transformer inference...")
            out_tokens, _ = run_inference(
            prompt_tokens=prompt_tokens,
            output_file=output_file,
            max_new_tokens=2000
        )
        
            print(f"✅ Inference completed successfully - {len(out_tokens)} tokens")
        
        except Exception as e:
            print(f"❌ Error in inference: {e}")
            import traceback
            traceback.print_exc()
            return self.fallback_transformer_output(data, session_id)
    
        return {"step": "transformer_fed"}

    def fallback_transformer_output(self, data, session_id):
        """Fallback transformer output for demo/testing"""
        emotion = data.get('emotion', 'joy')
        confidence = data.get('confidence', 0.57)
        
        # Use the actual format from your transformer output
        demo_output = f"""<BOS> Emotion_{emotion} Confidence_{confidence:.2f} [GENRE_{data['genre'].upper()}] [KEY_{data['key'].upper()}] [INSTRUMENT_PIANO] [TEMPO_{data['tempo'].upper()}] <SEP> Bar_Start TimeSig_4/4 Position_0 Tempo_120.0 Bar_None TimeSig_4/4 Position_0 Program_0 Pitch_60 Velocity_80 Duration_1.0 Position_480 Program_0 Pitch_64 Velocity_75 Duration_1.0 Position_960 Program_0 Pitch_67 Velocity_70 Duration_1.0 Bar_End"""
        
        # Save transformer output
        output_file = f'temp_files/transformer_output_{session_id}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(demo_output)
        
        print(f"⚠️ Using fallback transformer output for {session_id}")
        return {"step": "transformer_fed"}
    
    def process_transformer_output(self, data, session_id):
        """Step 6: Process transformer output and convert to clean REMI"""
        input_file = f'temp_files/transformer_output_{session_id}.txt'
        
        with open(input_file, 'r', encoding='utf-8') as f:
            transformer_output = f.read()
        
        print(f"Original transformer output length: {len(transformer_output)} chars")
        
        # Clean and extract REMI tokens
        remi_tokens = process_transformer_output(transformer_output)
        
        print(f"Cleaned REMI tokens length: {len(remi_tokens)} chars")
        print(f"Sample REMI tokens: {remi_tokens[:200]}...")
        
        # Save REMI tokens
        remi_file = f'temp_files/remi_{session_id}.txt'
        with open(remi_file, 'w', encoding='utf-8') as f:
            f.write(remi_tokens)
        
        return {"step": "remi_converted"}
    
    def convert_to_midi(self, data, session_id):
        """Step 7: Convert REMI to MIDI"""
        remi_path = f'temp_files/remi_{session_id}.txt'
        midi_path = f'output/midi/output_{session_id}.mid'
        
        try:
            print(f"Converting REMI to MIDI: {remi_path} -> {midi_path}")
            
            # Check if remi file exists and has content
            if not os.path.exists(remi_path):
                raise FileNotFoundError(f"REMI file not found: {remi_path}")
            
            with open(remi_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    raise ValueError("REMI file is empty")
            
            convert_remi_to_midi(remi_path, midi_path)
            
            # Check if MIDI was created
            if os.path.exists(midi_path):
                print(f"✅ MIDI created successfully: {midi_path}")
                return {"step": "midi_created"}
            else:
                raise Exception("MIDI file was not created")
                
        except Exception as e:
            print(f"❌ Error converting to MIDI: {e}")
            # Try to create a simple MIDI as fallback
            try:
                self.create_fallback_midi(session_id)
                return {"step": "midi_created_fallback"}
            except Exception as e2:
                return {"step": "midi_conversion_failed", "error": str(e)}
    
    def create_fallback_midi(self, session_id):
        """Create a simple fallback MIDI"""
        from miditoolkit import MidiFile, Instrument, Note
        
        midi_path = f'output/midi/output_{session_id}.mid'
        midi = MidiFile()
        
        # Create piano track
        piano = Instrument(program=0, is_drum=False, name="Piano")
        
        # Add a simple C major chord
        notes = [
            Note(start=0, end=480, pitch=60, velocity=80),  # C
            Note(start=0, end=480, pitch=64, velocity=80),  # E
            Note(start=0, end=480, pitch=67, velocity=80),  # G
            Note(start=480, end=960, pitch=62, velocity=75),  # D
            Note(start=480, end=960, pitch=65, velocity=75),  # F
            Note(start=480, end=960, pitch=69, velocity=75),  # A
        ]
        
        piano.notes.extend(notes)
        midi.instruments.append(piano)
        midi.dump(midi_path)
        print(f"🟡 Created fallback MIDI: {midi_path}")
    
    def convert_to_audio(self, data, session_id):
        """Step 8: Convert MIDI to audio"""
        midi_path = f'output/midi/output_{session_id}.mid'
        audio_path = f'output/audio/output_{session_id}.wav'
        
        try:
            print(f"Converting MIDI to audio: {midi_path} -> {audio_path}")
            
            if not os.path.exists(midi_path):
                raise FileNotFoundError(f"MIDI file not found: {midi_path}")
            
            convert_midi_to_audio(midi_path, audio_path)
            
            if os.path.exists(audio_path):
                print(f"✅ Audio created successfully: {audio_path}")
                return {"step": "audio_created"}
            else:
                raise Exception("Audio file was not created")
                
        except Exception as e:
            print(f"❌ Error converting to audio: {e}")
            return {"step": "audio_conversion_failed", "error": str(e)}

    def cleanup_temp_files(self, session_id):
        """Remove temporary session files"""
        filenames = [
            f'temp_files/lyrics_{session_id}.txt',
            f'temp_files/tokens_{session_id}.txt',
            f'temp_files/emotion_{session_id}.txt',
            f'temp_files/emotion_tokens_{session_id}.txt',
            f'temp_files/transformer_output_{session_id}.txt',
            f'temp_files/remi_{session_id}.txt',
            f'temp_files/prompt_{session_id}.txt',
        ]
        for fp in filenames:
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception as e:
                print(f"Warning: failed to remove {fp}: {e}")


@app.route('/generate-music', methods=['POST'])
def generate_music():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['lyrics', 'genre', 'instruments', 'tempo', 'key']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Initialize music generator
        generator = MusicGenerator()
        
        # Decide whether to auto-clean temp files (default: True). Frontend may pass {"cleanup": False}.
        auto_cleanup = data.get('cleanup', True)
        
        # Execute workflow steps
        steps = []
        emotion = "sadness"
        confidence = 1.0

        try:
            for step in generator.workflow_steps:
                try:
                    result = step(data, session_id)
                    step_name = result.get('step', 'unknown')
                    steps.append(step_name)
                    
                    # Check for errors
                    if 'error' in result:
                        return jsonify({
                            'error': f'Step failed: {step.__name__}',
                            'details': result.get('error', 'Unknown error')
                        }), 500
                    
                    # Store emotion and confidence from emotion extraction step
                    if step_name == "emotion_extracted":
                        emotion = result.get('emotion', 'sadness')
                        confidence = result.get('confidence', 1.0)
                        
                    data.update(result)  # Merge results for next steps
                except Exception as e:
                    return jsonify({
                        'error': f'Step failed: {step.__name__}',
                        'details': str(e)
                    }), 500
            
            # Return final result in the specified format
            final_result = {
                'session_id': session_id,
                'status': 'completed',
                'steps': steps,
                'emotion': emotion,
                'confidence': confidence,
                'downloads': {
                    'midi': f'/download/midi/{session_id}',
                    'audio': f'/download/audio/{session_id}'
                },
                'playback': {
                    'midi': f'/play/midi/{session_id}',
                    'audio': f'/play/audio/{session_id}'
                }
            }
            
            return jsonify(final_result)
        finally:
            # Auto-clean temporary session files unless disabled by the client
            try:
                if auto_cleanup:
                    generator.cleanup_temp_files(session_id)
                    print(f"Temp files for session {session_id} cleaned up.")
                else:
                    print(f"Temp files for session {session_id} preserved (cleanup disabled).")
            except Exception as e:
                print(f"Error during cleanup for session {session_id}: {e}")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/midi/<session_id>', methods=['GET'])
def download_midi(session_id):
    midi_path = f'output/midi/output_{session_id}.mid'
    if os.path.exists(midi_path):
        return send_file(midi_path, as_attachment=True, download_name=f'music_{session_id}.mid')
    else:
        return jsonify({'error': 'MIDI file not found'}), 404

@app.route('/download/audio/<session_id>', methods=['GET'])
def download_audio(session_id):
    audio_path = f'output/audio/output_{session_id}.wav'
    if os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=True, download_name=f'music_{session_id}.wav')
    else:
        return jsonify({'error': 'Audio file not found'}), 404

# NEW: Add endpoints for direct audio playback (not as attachment)
@app.route('/play/audio/<session_id>', methods=['GET'])
def play_audio(session_id):
    audio_path = f'output/audio/output_{session_id}.wav'
    if os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=False, mimetype='audio/wav')
    else:
        return jsonify({'error': 'Audio file not found'}), 404

@app.route('/play/midi/<session_id>', methods=['GET'])
def play_midi(session_id):
    midi_path = f'output/midi/output_{session_id}.mid'
    if os.path.exists(midi_path):
        return send_file(midi_path, as_attachment=False, mimetype='audio/midi')
    else:
        return jsonify({'error': 'MIDI file not found'}), 404
    
# For testing purpose only 
def test_remi_conversion():
    """Test the REMI conversion with your actual file"""
    try:
        print("🚀 Starting manual REMI conversion test...")
        
        # Create a test session ID
        session_id = "test_session_123"
        
        # Make sure your remi.txt file is in the same directory as app.py
        remi_source_file = "remi2tst/generated_music_fixed.txt"
        
        if not os.path.exists(remi_source_file):
            print(f"❌ REMI file not found: {remi_source_file}")
            print("Please make sure 'remi.txt' is in the same folder as app.py")
            return False
        
        # Copy your remi.txt to the expected temp location
        import shutil
        temp_remi_path = f"temp_files/remi_{session_id}.txt"
        shutil.copy(remi_source_file, temp_remi_path)
        print(f"✅ Copied REMI file to: {temp_remi_path}")
        
        # Test the conversion steps that your backend uses
        from remi2midi import convert_remi_to_midi
        from midi2audio import convert_midi_to_audio
        
        midi_path = f"output/midi/output_{session_id}.mid"
        audio_path = f"output/audio/output_{session_id}.wav"
        
        print("\n🎵 Testing REMI to MIDI conversion...")
        success = convert_remi_to_midi(temp_remi_path, midi_path)
        
        if success and os.path.exists(midi_path):
            midi_size = os.path.getsize(midi_path)
            print(f"✅ MIDI conversion successful!")
            print(f"📁 MIDI file: {midi_path}")
            print(f"📊 File size: {midi_size} bytes")
        else:
            print("❌ MIDI conversion failed")
            return False
        
        print("\n🔊 Testing MIDI to Audio conversion...")
        convert_midi_to_audio(midi_path, audio_path)
        
        if os.path.exists(audio_path):
            audio_size = os.path.getsize(audio_path)
            print(f"✅ Audio conversion successful!")
            print(f"📁 Audio file: {audio_path}")
            print(f"📊 File size: {audio_size} bytes")
            
            # Test playback endpoints
            print(f"\n🎮 Test playback URLs:")
            print(f"   MIDI: http://localhost:5000/play/midi/{session_id}")
            print(f"   Audio: http://localhost:5000/play/audio/{session_id}")
            print(f"   Download MIDI: http://localhost:5000/download/midi/{session_id}")
            print(f"   Download Audio: http://localhost:5000/download/audio/{session_id}")
        else:
            print("❌ Audio conversion failed")
            return False
        
        print("\n🎉 All conversions successful! Your REMI file works with the backend!")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
# test_remi_conversion()


if __name__ == '__main__':
    app.run(debug=True, port=5000)
