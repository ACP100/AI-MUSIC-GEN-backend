from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from utils.token_processor import extract_tokens, process_transformer_output
from lyrics_emotion import detect_emotion_from_text, load_emotion_models
from remi2midi import convert_remi_to_midi
from midi2audio import convert_midi_to_audio

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
        return {"status": "success", "step": "lyrics_saved"}
    
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
        
        return {"status": "success", "step": "tokens_extracted", "tokens": tokens}
    
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
            f.write(f"Emotion: {emotion} Confidence: {confidence:.2f}")
        
        return {
            "status": "success", 
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
        
        # Combine
        combined_content = emotion_line + '\n' + ''.join(tokens_lines)
        
        with open(f'temp_files/emotion_tokens_{session_id}.txt', 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        return {"status": "success", "step": "emotion_tokens_combined"}
    
    def feed_transformer(self, data, session_id):
        """Step 5: Feed to transformer model (Mock for now)"""
        # Read combined emotion and tokens
        with open(f'temp_files/emotion_tokens_{session_id}.txt', 'r', encoding='utf-8') as f:
            emotion_tokens = f.read()
        
        # This is where you'll integrate with your actual transformer model
        # For demo, using the example output you provided but incorporating actual emotion
        emotion = data.get('emotion', 'sadness')  # From previous step
        confidence = data.get('confidence', 1.00)
        
        demo_output = f"Emotion: {emotion} Confidence: {confidence:.2f} [GENRE_JAZZ] [KEY_C_MAJOR] [INSTRUMENT_BRASS_MISC] [INSTRUMENT_BRASS_SECTION] [INSTRUMENT_PERC_TONAL] [TEMPO_FAST] Bar_None TimeSig_4/4 Position_0 Tempo_134.84 Program_58 Pitch_40 Velocity_95 Duration_0.3.8 Program_11 Pitch_67 Velocity_107 Duration_1.0.8 Position_4 Program_58 Pitch_40 Velocity_91 Duration_0.3.8 Position_8 Pitch_40 Velocity_91 Duration_0.3.8 Program_11 Pitch_72 Velocity_95 Duration_0.5.8"
        
        # Save transformer output
        with open(f'temp_files/transformer_output_{session_id}.txt', 'w', encoding='utf-8') as f:
            f.write(demo_output)
        
        return {"status": "success", "step": "transformer_fed", "output_sample": demo_output[:100] + "..."}
    
    def process_transformer_output(self, data, session_id):
        """Step 6: Process transformer output and convert to REMI"""
        with open(f'temp_files/transformer_output_{session_id}.txt', 'r', encoding='utf-8') as f:
            transformer_output = f.read()
        
        # Process and extract REMI tokens
        remi_tokens = process_transformer_output(transformer_output)
        
        # Save REMI tokens
        with open(f'temp_files/remi_{session_id}.txt', 'w', encoding='utf-8') as f:
            f.write(remi_tokens)
        
        return {"status": "success", "step": "remi_converted", "remi_sample": remi_tokens[:100] + "..."}
    
    def convert_to_midi(self, data, session_id):
        """Step 7: Convert REMI to MIDI using your r1_remi2midi.py"""
        remi_path = f'temp_files/remi_{session_id}.txt'
        midi_path = f'output/midi/output_{session_id}.mid'
        
        try:
            convert_remi_to_midi(remi_path, midi_path)
            return {"status": "success", "step": "midi_created", "midi_path": midi_path}
        except Exception as e:
            return {"status": "error", "step": "midi_conversion_failed", "error": str(e)}
    
    def convert_to_audio(self, data, session_id):
        """Step 8: Convert MIDI to audio"""
        midi_path = f'output/midi/output_{session_id}.mid'
        audio_path = f'output/audio/output_{session_id}.wav'
        
        try:
            convert_midi_to_audio(midi_path, audio_path)
            return {"status": "success", "step": "audio_created", "audio_path": audio_path}
        except Exception as e:
            return {"status": "error", "step": "audio_conversion_failed", "error": str(e)}

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
        
        # Execute workflow steps
        results = []
        current_data = data
        
        for step in generator.workflow_steps:
            try:
                result = step(current_data, session_id)
                results.append(result)
                if result['status'] == 'error':
                    return jsonify({
                        'error': f'Step failed: {step.__name__}',
                        'details': result.get('error', 'Unknown error'),
                        'completed_steps': results
                    }), 500
                current_data.update(result)  # Merge results for next steps
            except Exception as e:
                return jsonify({
                    'error': f'Step failed: {step.__name__}',
                    'details': str(e),
                    'completed_steps': results
                }), 500
        
        # Return final result with download links
        final_result = {
            'session_id': session_id,
            'status': 'completed',
            'steps': [r.get('step', 'unknown') for r in results],
            'emotion': results[2].get('emotion', 'unknown'),
            'confidence': results[2].get('confidence', 0),
            'downloads': {
                'midi': f'/download/midi/{session_id}',
                'audio': f'/download/audio/{session_id}'
            }
        }
        
        return jsonify(final_result)
    
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)