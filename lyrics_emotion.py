from transformers import pipeline, AutoTokenizer
import os, re

# ------------------ Emotion Mapping ------------------ #
emotion_map = {
    "joy": ["joy", "amusement", "excitement", "approval", "pride", "relief", "admiration"],
    "sadness": ["sadness", "grief", "disappointment", "remorse", "embarrassment", "loneliness"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "disgust": ["disgust", "disapproval", "embarrassment"],
    "neutral": ["neutral", "calm", "love", "caring", "desire", "gratitude"]
}

def map_to_universal_emotion(label):
    label = label.lower()
    for uni, group in emotion_map.items():
        if label in group:
            return uni
    return "neutral"

# ------------------ Text Preprocessing ------------------ #
def preprocess_lyrics(lyrics):
    lyrics = lyrics.lower()
    lyrics = re.sub(r'[^\w\s]', '', lyrics)
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    return lyrics

def chunk_by_tokens(text, tokenizer, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i + max_tokens])

# ------------------ Emotion Detection ------------------ #
def load_emotion_models():
    """Load emotion models and return classifiers, tokenizers, and weights"""
    model_names = {
        "hartmann": "j-hartmann/emotion-english-distilroberta-base",
        "cardiffnlp": "cardiffnlp/twitter-roberta-base-emotion",
        "goemotions": "joeddav/distilbert-base-uncased-go-emotions-student"
    }

    weights = {
        "cardiffnlp": 0.4,
        "hartmann": 0.4,
        "goemotions": 0.2
    }

    classifiers = {}
    tokenizers = {}
    
    print("Loading emotion models...")
    for key, name in model_names.items():
        try:
            tokenizers[key] = AutoTokenizer.from_pretrained(name)
            classifiers[key] = pipeline("text-classification", model=name, top_k=None)
            print(f"✅ Loaded {key} model")
        except Exception as e:
            print(f"❌ Failed to load {key} model: {e}")
            # Fallback to mock for demo
            if not classifiers:
                classifiers[key] = None
                tokenizers[key] = None
    
    return classifiers, tokenizers, weights

def detect_emotion_from_text(lyrics, classifiers, tokenizers, weights):
    """Detect emotion from lyrics text"""
    lyrics = preprocess_lyrics(lyrics)
    all_scores = {}

    for clf_name, classifier in classifiers.items():
        if classifier is None:
            continue
            
        tokenizer = tokenizers[clf_name]
        if tokenizer is None:
            continue
            
        try:
            chunks = list(chunk_by_tokens(lyrics, tokenizer, max_tokens=510))
            results = classifier(chunks, truncation=True, max_length=512)

            # Aggregate per model
            model_scores = {}
            for res_list in results:
                if isinstance(res_list, dict):
                    res_list = [res_list]
                for r in res_list:
                    label = map_to_universal_emotion(r["label"])
                    model_scores[label] = model_scores.get(label, 0) + r["score"]

            # Find the top emotion per model
            if model_scores:
                top_label = max(model_scores, key=model_scores.get)
                normalized_score = model_scores[top_label] / sum(model_scores.values())
                # Apply weight and add to ensemble
                all_scores[top_label] = all_scores.get(top_label, 0) + normalized_score * weights[clf_name]
        except Exception as e:
            print(f"Error in {clf_name} classifier: {e}")
            continue

    # If no models worked, return default
    if not all_scores:
        return "neutral", 0.5

    # Determine final emotion and scale confidence
    final_emotion = max(all_scores, key=all_scores.get)
    total_score = sum(all_scores.values())
    final_confidence = all_scores[final_emotion] / total_score if total_score > 0 else 0

    return final_emotion, final_confidence

# ------------------ Main (for standalone use) ------------------ #
def main():
    lyrics_folder = "4A.lyrics"
    output_folder = "5A.lyrics_emotion"
    os.makedirs(output_folder, exist_ok=True)

    classifiers, tokenizers, weights = load_emotion_models()

    for filename in os.listdir(lyrics_folder):
        if filename.lower().endswith(".txt"):
            path = os.path.join(lyrics_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                lyrics = f.read()

            emotion, score = detect_emotion_from_text(lyrics, classifiers, tokenizers, weights)

            print(f"Processed {filename} => Final Emotion: {emotion} (Confidence: {score:.2f})")

            out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_emotion.txt")
            with open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write(f"Emotion: {emotion}\nConfidence: {score:.2f}")

if __name__ == "__main__":
    main()