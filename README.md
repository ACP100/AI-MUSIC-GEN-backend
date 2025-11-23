# AI-MUSIC-GEN Backend

A backend service for generating music from text lyrics using **emotion analysis**, **token conditioning**, **REMI symbolic generation**, and **MIDI-to-audio synthesis with FluidSynth**.

This API converts natural language lyrics into expressive audio and MIDI, and provides endpoints to **download** and **stream** the generated files.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Architecture Diagram](#architecture-diagram)
* [Directory Structure](#directory-structure)
* [Installation](#installation)

  * [System Requirements](#system-requirements)
  * [Python Dependencies](#python-dependencies)
  * [Installing FluidSynth](#installing-fluidsynth)
* [Running the Server](#running-the-server)
* [API Documentation](#api-documentation)

  * [POST /generate-music](#post-generate-music)
  * [Download & Playback Endpoints](#download--playback-endpoints)
* [File Purpose Reference](#file-purpose-reference)
* [Troubleshooting](#troubleshooting)

---

# Project Overview

The backend converts user inputs into a complete musical output:

1. Reads lyrics and musical preferences (genre, tempo, key, instruments)
2. Detects emotional tone using an ensemble of transformer models
3. Converts conditioning + emotion into REMI symbolic tokens
4. Converts REMI → MIDI using miditok
5. Converts MIDI → WAV using FluidSynth and GeneralUser-GS.sf2 soundfont
6. Returns downloadable and streamable audio links

This backend is designed to be integrated with a frontend for AI-powered music generation.

---

# Features

### Emotion Detection

Uses weighted ensemble of the following:

* j-hartmann DistilRoBERTa Emotion Model
* CardiffNLP Twitter-RoBERTa Emotion
* GoEmotions distilled model

### Symbolic Music Generation

* Creates REMI tokens
* Converts tokens to MIDI using miditok’s REMI tokenizer
* Auto-cleans or falls back on simple MIDI if token errors occur

### Audio Synthesis

* Converts MIDI → WAV with FluidSynth
* Includes fallback mode that generates a silent WAV if FluidSynth fails

### Modern Flask Backend

* Direct MIDI download
* Direct WAV playback
* Automatic temp file cleanup
* Fully CORS-enabled

---

# Architecture Diagram

```
User Input
  - Lyrics
  - Genre
  - Tempo
  - Instruments
  - Key
        |
        v
Emotion Detection (3-model ensemble)
        |
        v
Token Generation (conditioning + emotion)
        |
        v
Transformer (mock output for now)
        |
        v
REMI Token Extraction (token_processor.py)
        |
        v
REMI → MIDI (remi2midi.py using miditok)
        |
        v
MIDI → WAV (FluidSynth + GeneralUser-GS.sf2)
        |
        v
Response JSON + Download URLs
```

---

# Directory Structure

```
AI-MUSIC-GEN-backend/
│
├── app.py                   # Main Flask server
├── midi2audio.py            # MIDI → WAV synthesis
├── remi2midi.py             # REMI → MIDI converter
├── lyrics_emotion.py        # Emotion detection module
├── token_processor.py       # Token cleaning and extraction
├── GeneralUser-GS.sf2       # SoundFont for audio rendering
└── notes.md                 # Project notes
```

---

# Installation

## System Requirements

* Python 3.9+
* 8 GB RAM recommended
* Fluidsynth installed system-wide
* A valid `.sf2` soundfont

## Python Dependencies

Create a virtual environment:

```
python -m venv venv
```

Activate:

```
venv/Scripts/activate     # Windows
source venv/bin/activate  # Linux/Mac
```

Install libraries:

```
pip install flask flask-cors transformers torch miditok miditoolkit numpy
```

Or if you have a `requirements.txt`:

```
pip install -r requirements.txt
```

---

## Installing FluidSynth

### Windows (Chocolatey)

```
choco install fluidsynth
```

### Ubuntu / Debian

```
sudo apt update
sudo apt install fluidsynth
```

### MacOS (Homebrew)

```
brew install fluidsynth
```

### Verifying the installation:

```
fluidsynth --version
```

---

# Running the Server

From the project directory:

```
python app.py
```

Server runs at:

```
http://localhost:5000
```

---

# API Documentation

## POST `/generate-music`

### Request Body

```json
{
  "lyrics": "I walk alone in the rain...",
  "genre": "jazz",
  "instruments": ["piano", "saxophone"],
  "tempo": "slow",
  "key": "c_minor"
}
```

### Successful Response

```json
{
    "session_id": "4d2c8a...",
    "status": "completed",
    "emotion": "sadness",
    "confidence": 0.87,
    "steps": [
        "lyrics_saved",
        "tokens_extracted",
        "emotion_extracted",
        "emotion_tokens_combined",
        "transformer_fed",
        "remi_converted",
        "midi_created",
        "audio_created"
    ],
    "downloads": {
        "midi": "/download/midi/4d2c8a",
        "audio": "/download/audio/4d2c8a"
    },
    "playback": {
        "midi": "/play/midi/4d2c8a",
        "audio": "/play/audio/4d2c8a"
    }
}
```

---

## Download & Playback Endpoints

### Download

* `/download/midi/<session_id>`
* `/download/audio/<session_id>`

### Stream / Play

* `/play/midi/<session_id>`
* `/play/audio/<session_id>`

---

# File Purpose Reference

### app.py

Main workflow engine initiating all 8 steps:

* lyrics → tokens → emotion → REMI → MIDI → WAV

### midi2audio.py

Uses FluidSynth. Has robust error handling and fallback WAV generation.

### remi2midi.py

Uses miditok’s REMI tokenizer with strict token cleaning.

### lyrics_emotion.py

Ensemble emotion detection with tokenized chunking and standardization.

### token_processor.py

Extracts valid REMI tokens and builds minimal fallback structure if needed.

---

# Troubleshooting

### FluidSynth not installed

WAV file is silent or missing.
Install FluidSynth system-wide.

### SoundFont not found

midi2audio.py checks:

```
./GeneralUser-GS.sf2
/usr/share/sounds/sf2/FluidR3_GM.sf2
/usr/share/soundfonts/FluidR3_GM.sf2
```

Place your `.sf2` in project root.

### Emotion detection slow

The three transformer models are heavy.
Enable GPU for better performance.

### REMI error: `'1.0'`

Your transformer output is malformed.
token_processor.py auto-corrects most issues.

---
