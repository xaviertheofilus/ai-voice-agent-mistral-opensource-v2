# [file name]: app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn
import asyncio
import base64
import os
import logging
import json
import time
from datetime import datetime
from typing import Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor

# Import processors
try:
    from processors.stt_processor import OptimizedSTTProcessor
    from processors.tts_processor import OptimizedTTSProcessor
    from processors.rag_processor import OptimizedRAGProcessor
    from processors.template_matcher import OptimizedTemplateMatcher
    PROCESSORS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importing processors: {e}")
    PROCESSORS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Voice Agent - Enhanced", version="2.1.0")

# Initialize processors
processors = {
    'stt': None,
    'tts': None,
    'rag': None,
    'template': None,
    'initialized': False
}

active_sessions: Dict[str, Dict] = {}
conversation_history: Dict[str, Dict] = {}
executor = ThreadPoolExecutor(max_workers=4)

async def initialize_processors():
    try:
        logger.info("Initializing processors...")
        
        if not PROCESSORS_AVAILABLE:
            logger.error("Processors are not available due to import errors")
            return False
        
        # Initialize STT with optimized settings
        try:
            processors['stt'] = OptimizedSTTProcessor(
                model_size="base",  # Better accuracy
                device="cpu",
                compute_type="int8"
            )
            logger.info("STT processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing STT processor: {e}")
            processors['stt'] = None
        
        # Initialize TTS with optimized settings
        try:
            processors['tts'] = OptimizedTTSProcessor(
                use_neural=True,
                voice_quality="high"  # Higher quality
            )
            logger.info("TTS processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TTS processor: {e}")
            processors['tts'] = None
        
        # Initialize RAG with efficient model
        try:
            processors['rag'] = OptimizedRAGProcessor(
                model_name="mistral:7b",
                max_context=1024
            )
            logger.info("RAG processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG processor: {e}")
            processors['rag'] = None
        
        # Initialize template matcher
        try:
            processors['template'] = OptimizedTemplateMatcher()
            logger.info("Template matcher initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing template matcher: {e}")
            processors['template'] = None
        
        # Check if essential processors are initialized
        if processors['stt'] and processors['tts']:
            processors['initialized'] = True
            logger.info("All essential processors initialized successfully!")
            return True
        else:
            logger.error("Essential processors (STT/TTS) failed to initialize")
            processors['initialized'] = False
            return False
            
    except Exception as e:
        logger.error(f"Error initializing processors: {str(e)}")
        processors['initialized'] = False
        return False

@app.on_event("startup")
async def startup_event():
    os.makedirs("static", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/templates", exist_ok=True)
    os.makedirs("conversations", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Initialize processors in background to not block startup
    asyncio.create_task(initialize_processors())

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: Frontend files not found</h1>", status_code=404)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = None
    
    try:
        await websocket.accept()
        client_id = str(id(websocket))
        
        active_sessions[client_id] = {
            'websocket': websocket,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        
        conversation_history[client_id] = {
            'transcripts': [],
            'responses': [],
            'timestamps': [],
            'template_matches': []
        }
        
        logger.info(f"Client {client_id} connected")
        
        await websocket.send_json({
            "type": "status",
            "message": "Connected successfully",
            "client_id": client_id
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                asyncio.create_task(process_audio_message(websocket, client_id, data))
            elif data.get("type") == "text":
                asyncio.create_task(process_text_message(websocket, client_id, data))
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if client_id:
            active_sessions.pop(client_id, None)

async def process_audio_message(websocket: WebSocket, client_id: str, data: dict):
    try:
        if not processors['initialized']:
            await websocket.send_json({
                "type": "error",
                "message": "System still initializing, please wait..."
            })
            return
            
        audio_data = base64.b64decode(data["data"])
        
        await websocket.send_json({"type": "processing", "stage": "transcribing"})
        
        # Process transcription in a separate thread
        transcript = await asyncio.get_event_loop().run_in_executor(
            executor, 
            processors['stt'].transcribe_audio, 
            audio_data
        )
        
        if not transcript or transcript.strip() == "":
            await websocket.send_json({
                "type": "error",
                "message": "Could not transcribe audio. Please try again."
            })
            return
            
        conversation_history[client_id]['transcripts'].append(transcript)
        conversation_history[client_id]['timestamps'].append(datetime.now().isoformat())
        
        await websocket.send_json({
            "type": "transcript",
            "text": transcript
        })
        
        await websocket.send_json({"type": "processing", "stage": "generating"})
        
        # Try template matching first for faster response (if available)
        template_response = None
        if processors['template']:
            template_response = processors['template'].match_template(transcript)
        
        if template_response:
            response = template_response
            conversation_history[client_id]['template_matches'].append({
                'query': transcript,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        elif processors['rag']:
            # Use RAG for more complex queries
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                processors['rag'].generate_response,
                transcript
            )
        else:
            response = "Maaf, sistem AI belum siap. Silakan coba beberapa saat lagi."
        
        conversation_history[client_id]['responses'].append(response)
        
        await websocket.send_json({
            "type": "response",
            "text": response
        })
        
        await websocket.send_json({"type": "processing", "stage": "synthesizing"})
        
        # Detect language for TTS
        language = detect_language(transcript)
        
        # Generate audio response if TTS is available
        if processors['tts']:
            audio_response = await asyncio.get_event_loop().run_in_executor(
                executor,
                processors['tts'].synthesize_speech,
                response,
                language
            )
            
            if audio_response:
                audio_b64 = base64.b64encode(audio_response).decode("utf-8")
                await websocket.send_json({
                    "type": "audio_response",
                    "data": audio_b64
                })
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": f"Processing error: {str(e)}"
        })

async def process_text_message(websocket: WebSocket, client_id: str, data: dict):
    try:
        text = data.get("text", "").strip()
        if not text:
            return
            
        conversation_history[client_id]['transcripts'].append(text)
        conversation_history[client_id]['timestamps'].append(datetime.now().isoformat())
        
        # Try template matching first for faster response (if available)
        template_response = None
        if processors['template']:
            template_response = processors['template'].match_template(text)
        
        if template_response:
            response = template_response
        elif processors['rag']:
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                processors['rag'].generate_response,
                text
            )
        else:
            response = "Maaf, sistem AI belum siap. Silakan coba beberapa saat lagi."
        
        conversation_history[client_id]['responses'].append(response)
        
        await websocket.send_json({
            "type": "response",
            "text": response
        })
        
        # Also generate audio response for text messages
        language = detect_language(text)
        
        if processors['tts']:
            audio_response = await asyncio.get_event_loop().run_in_executor(
                executor,
                processors['tts'].synthesize_speech,
                response,
                language
            )
            
            if audio_response:
                audio_b64 = base64.b64encode(audio_response).decode("utf-8")
                await websocket.send_json({
                    "type": "audio_response",
                    "data": audio_b64
                })
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": f"Error: {str(e)}"
        })

def detect_language(text: str) -> str:
    if not text:
        return "id"
    
    # Simple language detection based on common words
    english_words = ['the', 'and', 'is', 'are', 'what', 'where', 'when', 'why', 'how', 'hello', 'hi']
    text_lower = text.lower()
    
    english_count = sum(1 for word in english_words if word in text_lower)
    word_count = len(text_lower.split())
    
    # If more than 30% of detected words are English, use English
    if word_count > 0 and (english_count / word_count) > 0.3:
        return "en"
    
    return "id"

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")
        
        file_path = f"data/{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        if processors['rag']:
            success = await asyncio.get_event_loop().run_in_executor(
                executor,
                processors['rag'].add_document,
                file_path
            )
            
            if success:
                return {"message": "PDF uploaded and processed", "filename": file.filename}
            else:
                raise HTTPException(status_code=500, detail="Failed to process PDF")
        else:
            raise HTTPException(status_code=503, detail="RAG processor not available")
            
    except Exception as e:
        logger.error(f"PDF upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-template")
async def upload_template(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")
        
        file_path = f"data/templates/{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        if processors['template']:
            processors['template'].reload_templates()
            return {"message": "Template uploaded", "filename": file.filename}
        else:
            raise HTTPException(status_code=503, detail="Template processor not available")
        
    except Exception as e:
        logger.error(f"Template upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-conversation/{client_id}")
async def download_conversation(client_id: str):
    try:
        if client_id not in conversation_history:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        conversation_data = conversation_history[client_id]
        
        structured_data = {
            "conversation_id": client_id,
            "created_at": conversation_data.get('created_at', datetime.now().isoformat()),
            "total_exchanges": len(conversation_data['transcripts']),
            "exchanges": []
        }
        
        for i in range(len(conversation_data['transcripts'])):
            exchange = {
                "id": i + 1,
                "timestamp": conversation_data['timestamps'][i] if i < len(conversation_data['timestamps']) else datetime.now().isoformat(),
                "user_input": conversation_data['transcripts'][i],
                "assistant_response": conversation_data['responses'][i] if i < len(conversation_data['responses']) else "No response"
            }
            structured_data["exchanges"].append(exchange)
        
        if 'template_matches' in conversation_data and conversation_data['template_matches']:
            structured_data["template_matches"] = conversation_data['template_matches']
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"conversation_{client_id}_{timestamp}.json"
        file_path = f"conversations/{filename}"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/json'
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        status = {
            "status": "healthy" if processors['initialized'] else "initializing",
            "processors": {
                "stt": processors['stt'] is not None,
                "tts": processors['tts'] is not None,
                "rag": processors['rag'] is not None,
                "template": processors['template'] is not None
            },
            "active_sessions": len(active_sessions),
            "conversation_history": len(conversation_history),
            "timestamp": datetime.now().isoformat()
        }
        
        if processors['rag']:
            rag_status = {
                "documents_loaded": processors['rag'].has_documents() if hasattr(processors['rag'], 'has_documents') else False,
                "vector_store_ready": processors['rag'].is_ready() if hasattr(processors['rag'], 'is_ready') else False
            }
            status["rag_status"] = rag_status
        
        if processors['template']:
            template_status = {
                "templates_loaded": processors['template'].has_templates() if hasattr(processors['template'], 'has_templates') else False,
                "template_count": processors['template'].get_template_count() if hasattr(processors['template'], 'get_template_count') else 0
            }
            status["template_status"] = template_status
        
        return status
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/restart-processors")
async def restart_processors():
    try:
        global processors
        processors = {
            'stt': None,
            'tts': None,
            'rag': None,
            'template': None,
            'initialized': False
        }
        
        success = await initialize_processors()
        return {"status": "success" if success else "error", "message": "Processors restarted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=False
    )