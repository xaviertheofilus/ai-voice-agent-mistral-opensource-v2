## AI Voice Agent - Intelligent Voice Assistant System

https://youtu.be/hj2m4afAlfQ

## Project Description

AI Voice Agent is an intelligent voice assistant application that enables interaction through voice and text in both Indonesian and English. The system is equipped with speech-to-text (STT), text-to-speech (TTS), template matching, and RAG (Retrieval Augmented Generation) capabilities to provide accurate and natural responses.

<img width="1600" height="832" alt="image" src="https://github.com/user-attachments/assets/71ffddb1-3087-4ce8-814e-539c11125aa0" />
<img width="1897" height="589" alt="image" src="https://github.com/user-attachments/assets/414a00a0-afbd-4d7c-88e3-3f852fc2ab60" />

## Key Features

- 🎤 **Real-time Voice Input**: High-accuracy live audio transcription
- 🔊 **Natural Voice Output**: Human-like voice responses with natural intonation
- 🤖 **Multi-language Support**: Indonesian and English language support
- 📚 **Knowledge Base**: PDF document understanding through RAG
- 📋 **Template Matching**: Fast responses for common questions
- 💬 **Modern Web Interface**: Responsive UI with real-time audio visualization
- 📊 **Data Export**: Conversation export in JSON format

## Technology Stack

### Backend
- **FastAPI** - Modern, fast web framework
- **Faster-Whisper** - Optimized STT engine
- **Edge-TTS** - TTS engine with natural neural voices
- **LangChain** - RAG framework for document processing
- **FAISS** - Vector store for semantic search
- **Ollama** - Local language models (Mistral, Llama, etc.)

### Frontend
- **HTML5 & CSS3** - Responsive user interface
- **JavaScript ES6+** - Web application logic
- **Web Audio API** - Audio visualization and processing
- **WebSocket** - Real-time communication with backend

## Installation & Setup

### Prerequisites
- Python 3.8+
- Pip
- Virtualenv (recommended)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-voice-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download STT model (optional)**
   ```bash
   # Models are automatically downloaded on first run
   # For manual download:
   python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:8000`

## Project Structure

```
ai-voice-agent/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── processors/           # Processing modules
│   ├── __init__.py
│   ├── stt_processor.py  # Speech-to-text processing
│   ├── tts_processor.py  # Text-to-speech processing
│   ├── rag_processor.py  # RAG processing
│   └── template_matcher.py # Template matching
├── static/               # Frontend files
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/                 # Data and documents
│   └── templates/        # CSV template files
├── models/               # ML models (auto-created)
├── conversations/        # Conversation exports
└── temp/                 # Temporary files
```

## Configuration

### Language Models
The system supports various models through Ollama:
- Mistral 7B (default)
- Llama 3.2 1B
- Gemma 2B

To change models, edit parameters in `rag_processor.py`:
```python
def __init__(self, model_name="mistral:7b", max_context=2048, data_folder="data"):
```

### Voice Settings
Voice configuration can be adjusted in `tts_processor.py`:
```python
def __init__(self, use_neural=True, voice_quality="high"):
```

## Usage Guide

### Voice Interaction
1. Click the "Start Recording" button
2. Speak clearly in Indonesian or English
3. The system will transcribe and respond to your query
4. Responses will be provided in both text and audio formats

### Text Interaction
1. Type your question in the text input field
2. Press Enter or click the send button
3. Receive both text and audio responses

### Document Upload
1. Upload PDF files to expand the knowledge base
2. Add CSV templates for common Q&A pairs
3. Drag and drop or click to select files

### Conversation Management
1. Download conversations as JSON files
2. Clear chat history when needed
3. Monitor system status through the status panel

## Performance Optimization

For better performance:
- Use a base or larger Whisper model for better accuracy
- Ensure sufficient RAM (recommended 8GB+)
- Use SSD storage for faster model loading
- Close other resource-intensive applications

## Troubleshooting

### Common Issues
1. **Audio not working**: Check browser microphone permissions
2. **Models not loading**: Ensure stable internet connection for initial download
3. **Slow responses**: Check system resources and close other applications

### Getting Help
1. Check the system status panel for component status
2. Review browser console for error messages
3. Ensure all dependencies are properly installed

## License

This project is open source and available under the MIT License.

