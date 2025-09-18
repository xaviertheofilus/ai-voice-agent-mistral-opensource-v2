"""
Processors package for AI Voice Agent
Contains optimized processors for STT, TTS, RAG, and Template Matching
"""

try:
    from .stt_processor import OptimizedSTTProcessor
    from .tts_processor import OptimizedTTSProcessor
    from .rag_processor import OptimizedRAGProcessor
    from .template_matcher import OptimizedTemplateMatcher
except ImportError as e:
    print(f"Warning: Could not import processors: {e}")

__all__ = [
    'OptimizedSTTProcessor',
    'OptimizedTTSProcessor', 
    'OptimizedRAGProcessor',
    'OptimizedTemplateMatcher'
]

__version__ = '2.0.0'