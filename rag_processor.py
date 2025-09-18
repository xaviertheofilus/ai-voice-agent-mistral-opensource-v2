import os
import logging
from typing import List, Optional, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain_community.llms import GPT4All
    GPT4ALL_AVAILABLE = True
except ImportError:
    GPT4ALL_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizedRAGProcessor:
    def __init__(self, model_name="mistral:7b", max_context=2048, data_folder="data"):
        self.model_name = model_name
        self.max_context = max_context
        self.data_folder = data_folder
        
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.text_splitter = None
        
        self._documents_loaded = False
        self._vector_store_ready = False
        self._llm_ready = False
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self._initialize()
    
    def _initialize(self):
        try:
            logger.info("Initializing RAG processor...")
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self._initialize_llm()
            self._load_documents()
            
            logger.info("RAG processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG processor: {str(e)}")
    
    def _initialize_llm(self):
        # Prioritize smaller models for better performance
        llm_options = [
            ("ollama", "mistral:7b"),  # Efficient and good for Indonesian/English
            ("ollama", "llama3.2:1b"),  # Very lightweight
            ("ollama", "gemma:2b"),     # Google's efficient model
            ("gpt4all", "mistral-7b-openorca.Q4_0.gguf"),
            ("gpt4all", "orca-mini-3b-gguf2-q4_0.gguf")
        ]
        
        for llm_type, model_name in llm_options:
            try:
                if llm_type == "ollama" and OLLAMA_AVAILABLE:
                    self.llm = Ollama(
                        model=model_name,
                        timeout=30,
                        temperature=0.1,
                        top_p=0.9,
                        num_ctx=self.max_context,
                        num_predict=256,
                        stop=['Human:', 'Assistant:', '\n\n']
                    )
                    
                    # Test the model with a simple prompt
                    test_response = self.llm("Hello")
                    if test_response:
                        logger.info(f"Successfully initialized Ollama with {model_name}")
                        self._llm_ready = True
                        return
                        
                elif llm_type == "gpt4all" and GPT4ALL_AVAILABLE:
                    self.llm = GPT4All(
                        model=model_name,
                        max_tokens=256,
                        temp=0.1,
                        top_p=0.9,
                        echo=False,
                        n_threads=4
                    )
                    
                    test_response = self.llm("Hello")
                    if test_response:
                        logger.info(f"Successfully initialized GPT4All with {model_name}")
                        self._llm_ready = True
                        return
                        
            except Exception as e:
                logger.warning(f"Failed to initialize {llm_type} with {model_name}: {e}")
                continue
        
        logger.error("No suitable LLM could be initialized")
        self._llm_ready = False
    
    def _load_documents(self):
        try:
            if not os.path.exists(self.data_folder):
                os.makedirs(self.data_folder, exist_ok=True)
                logger.info(f"Created data folder: {self.data_folder}")
                return
            
            documents = []
            supported_files = []
            
            for filename in os.listdir(self.data_folder):
                if filename.startswith('templates'):
                    continue
                    
                file_path = os.path.join(self.data_folder, filename)
                if os.path.isfile(file_path):
                    if filename.lower().endswith(('.pdf', '.txt')):
                        supported_files.append((filename, file_path))
            
            if not supported_files:
                logger.info("No documents found to load")
                return
            
            for filename, file_path in supported_files:
                try:
                    logger.info(f"Loading document: {filename}")
                    
                    if filename.lower().endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif filename.lower().endswith('.txt'):
                        loader = TextLoader(file_path, encoding='utf-8')
                    else:
                        continue
                    
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} pages from {filename}")
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
                    continue
            
            if documents:
                self._create_vector_store(documents)
                self._documents_loaded = True
                logger.info(f"Successfully loaded {len(documents)} document pages")
            else:
                logger.warning("No documents were successfully loaded")
                
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
    
    def _create_vector_store(self, documents):
        try:
            logger.info("Creating vector store...")
            
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} text chunks")
            
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            self._setup_qa_chain()
            
            self._vector_store_ready = True
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            self._vector_store_ready = False
    
    def _setup_qa_chain(self):
        if not self.llm or not self.vectorstore:
            logger.warning("Cannot setup QA chain: LLM or vector store not ready")
            return
        
        try:
            prompt_template = """Gunakan konteks berikut untuk menjawab pertanyaan dengan akurat dan ringkas.
Jika jawabannya tidak ada dalam konteks, katakan bahwa Anda tidak memiliki informasi tersebut.
Jawab dalam bahasa yang sama dengan pertanyaan.

Konteks: {context}

Pertanyaan: {question}

Jawaban singkat dan akurat:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 3,
                        "fetch_k": 10,
                        "score_threshold": 0.5
                    }
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=False
            )
            
            logger.info("QA chain setup successfully")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
    
    def generate_response(self, query: str) -> str:
        try:
            if not self._llm_ready:
                return "Maaf, sistem AI belum siap. Silakan coba beberapa saat lagi."
            
            if not self._documents_loaded or not self._vector_store_ready:
                return self._generate_general_response(query)
            
            if self.qa_chain:
                try:
                    result = self.qa_chain({"query": query})
                    response = result.get("result", "").strip()
                    
                    if len(response) < 10 or "tidak memiliki informasi" in response.lower():
                        return self._generate_general_response(query)
                    
                    return response
                    
                except Exception as e:
                    logger.warning(f"RAG query failed: {e}")
                    return self._generate_general_response(query)
            else:
                return self._generate_general_response(query)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi."
    
    def _generate_general_response(self, query: str) -> str:
        try:
            if not self.llm:
                return "Maaf, sistem tidak tersedia saat ini."
            
            general_prompt = f"""Anda adalah asisten AI yang membantu dalam bahasa Indonesia.
Jawablah pertanyaan berikut dengan ramah, informatif, dan ringkas.
Jika tidak tahu jawabannya, katakan dengan jujur.

Pertanyaan: {query}

Jawaban dalam bahasa Indonesia:"""
            
            response = self.llm(general_prompt)
            
            if response:
                response = response.strip()
                prefixes_to_remove = ["Jawaban:", "Response:", "A:", "Q:"]
                for prefix in prefixes_to_remove:
                    if response.startswith(prefix):
                        response = response[len(prefix):].strip()
                
                return response if response else "Maaf, saya tidak dapat memahami pertanyaan Anda."
            
            return "Maaf, saya tidak dapat memproses pertanyaan Anda saat ini."
            
        except Exception as e:
            logger.error(f"Error in general response: {str(e)}")
            return "Maaf, terjadi kesalahan teknis. Silakan coba lagi."
    
    def add_document(self, file_path: str) -> bool:
        try:
            logger.info(f"Adding document: {file_path}")
            
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.error(f"Unsupported file type: {file_path}")
                return False
            
            documents = loader.load()
            
            if not documents:
                logger.error("No content found in document")
                return False
            
            splits = self.text_splitter.split_documents(documents)
            
            if self.vectorstore:
                new_vectorstore = FAISS.from_documents(splits, self.embeddings)
                self.vectorstore.merge_from(new_vectorstore)
            else:
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            
            self._setup_qa_chain()
            
            self._documents_loaded = True
            self._vector_store_ready = True
            
            logger.info(f"Successfully added document with {len(splits)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False
    
    def has_documents(self) -> bool:
        return self._documents_loaded
    
    def is_ready(self) -> bool:
        return self._llm_ready and (self._vector_store_ready or not self._documents_loaded)
    
    def get_status(self) -> Dict:
        return {
            "llm_ready": self._llm_ready,
            "documents_loaded": self._documents_loaded,
            "vector_store_ready": self._vector_store_ready,
            "model_name": self.model_name,
            "max_context": self.max_context,
            "is_ready": self.is_ready()
        }
    
    def get_document_count(self) -> int:
        if self.vectorstore:
            return self.vectorstore.index.ntotal
        return 0