class VoiceAssistant {
    constructor() {
        this.socket = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.isConnected = false;
        this.clientId = Date.now();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.systemStatus = {};
        
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.animationId = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initWebSocket();
        this.checkSystemStatus();
        this.initAudioVisualization();
        
        setTimeout(() => {
            this.showWelcomeMessage();
        }, 1000);
    }
    
    setupEventListeners() {
        document.getElementById('recordButton').addEventListener('click', () => this.toggleRecording());
        
        const textInput = document.getElementById('textInput');
        const sendBtn = document.getElementById('sendTextBtn');
        
        sendBtn.addEventListener('click', () => this.sendTextMessage());
        textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendTextMessage();
            }
        });
        
        this.setupFileUpload('pdfUploadArea', 'pdfUpload', 'pdf');
        this.setupFileUpload('csvUploadArea', 'csvUpload', 'csv');
        
        document.getElementById('downloadBtn').addEventListener('click', () => this.downloadConversation());
        document.getElementById('refreshBtn').addEventListener('click', () => this.refreshSystem());
        document.getElementById('helpBtn').addEventListener('click', () => this.showHelpModal());
        document.getElementById('clearChatBtn').addEventListener('click', () => this.clearChat());
        document.getElementById('exportChatBtn').addEventListener('click', () => this.downloadConversation());
        
        document.getElementById('helpModalClose').addEventListener('click', () => this.hideHelpModal());
        
        document.getElementById('helpModal').addEventListener('click', (e) => {
            if (e.target.id === 'helpModal') {
                this.hideHelpModal();
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'Enter':
                        e.preventDefault();
                        this.toggleRecording();
                        break;
                    case 'd':
                        e.preventDefault();
                        this.downloadConversation();
                        break;
                }
            }
        });
    }
    
    setupFileUpload(areaId, inputId, type) {
        const area = document.getElementById(areaId);
        const input = document.getElementById(inputId);
        
        area.addEventListener('click', () => input.click());
        
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('drag-over');
        });
        
        area.addEventListener('dragleave', () => {
            area.classList.remove('drag-over');
        });
        
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if ((type === 'pdf' && file.type === 'application/pdf') ||
                    (type === 'csv' && file.name.toLowerCase().endsWith('.csv'))) {
                    this.handleFileUpload(file, type);
                } else {
                    this.showNotification(`Please select a valid ${type.toUpperCase()} file`, 'error');
                }
            }
        });
        
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleFileUpload(file, type);
            }
        });
    }
    
    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.socket = new WebSocket(wsUrl);
            
            this.socket.onopen = () => {
                console.log('Connected to server');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');
                this.addMessage('system', 'Terhubung ke server. Sistem siap digunakan!');
            };
            
            this.socket.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('error');
            };
            
            this.socket.onclose = () => {
                console.log('WebSocket connection closed');
                this.isConnected = false;
                this.updateConnectionStatus('disconnected');
                
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    this.addMessage('system', `Mencoba menghubungkan kembali (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
                    setTimeout(() => this.initWebSocket(), 2000);
                } else {
                    this.addMessage('system', 'Koneksi terputus. Silakan refresh halaman.');
                }
            };
            
        } catch (e) {
            console.error('Error creating WebSocket:', e);
            this.updateConnectionStatus('error');
        }
    }
    
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'status':
                    this.addMessage('system', data.message);
                    this.clientId = data.client_id || this.clientId;
                    break;
                    
                case 'transcript':
                    this.addMessage('user', data.text);
                    break;
                    
                case 'response':
                    this.removeProcessingMessage();
                    this.addMessage('assistant', data.text);
                    break;
                    
                case 'audio_response':
                    this.playAudioResponse(data.data);
                    break;
                    
                case 'processing':
                    this.showProcessingMessage(data.stage);
                    break;
                    
                case 'error':
                    this.removeProcessingMessage();
                    this.addMessage('system', `Error: ${data.message}`, 'error');
                    break;
                    
                default:
                    console.log('Unknown message type:', data.type);
            }
        } catch (e) {
            console.error('Error processing message:', e);
        }
    }
    
    async toggleRecording() {
        if (!this.isRecording) {
            await this.startRecording();
        } else {
            this.stopRecording();
        }
    }
    
    async startRecording() {
        try {
            if (!this.isConnected) {
                this.showNotification('Tidak terhubung ke server', 'error');
                return;
            }
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                } 
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };
            
            this.mediaRecorder.start(1000);
            this.isRecording = true;
            
            this.updateRecordingUI(true);
            this.startAudioVisualization(stream);
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.showNotification('Tidak dapat mengakses mikrofon. Periksa izin browser.', 'error');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            if (this.mediaRecorder.stream) {
                this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
            
            this.updateRecordingUI(false);
            this.stopAudioVisualization();
        }
    }
    
    async processRecording() {
        try {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            const arrayBuffer = await audioBlob.arrayBuffer();
            const base64Audio = btoa(
                new Uint8Array(arrayBuffer).reduce(
                    (data, byte) => data + String.fromCharCode(byte), ''
                )
            );
            
            this.showProcessingMessage('transcribing');
            
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                this.socket.send(JSON.stringify({
                    type: 'audio',
                    data: base64Audio
                }));
            } else {
                this.addMessage('system', 'Tidak terhubung ke server', 'error');
                this.removeProcessingMessage();
            }
        } catch (e) {
            console.error('Error processing audio:', e);
            this.addMessage('system', 'Error memproses audio', 'error');
            this.removeProcessingMessage();
        }
    }
    
    sendTextMessage() {
        const textInput = document.getElementById('textInput');
        const text = textInput.value.trim();
        
        if (!text) return;
        
        if (!this.isConnected) {
            this.showNotification('Tidak terhubung ke server', 'error');
            return;
        }
        
        textInput.value = '';
        this.addMessage('user', text);
        this.showProcessingMessage('generating');
        
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({
                type: 'text',
                text: text
            }));
        }
    }
    
    async handleFileUpload(file, type) {
        const formData = new FormData();
        formData.append('file', file);
        
        const endpoint = type === 'pdf' ? '/upload-pdf' : '/upload-template';
        const statusEl = document.getElementById('uploadStatus');
        
        try {
            statusEl.textContent = `Mengupload ${file.name}...`;
            statusEl.className = 'upload-status uploading';
            
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                statusEl.textContent = `✓ ${result.filename || file.name} berhasil diupload`;
                statusEl.className = 'upload-status success';
                
                this.addMessage('system', `File ${result.filename || file.name} berhasil diupload dan diproses.`);
                setTimeout(() => this.checkSystemStatus(), 1000);
                
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Upload gagal');
            }
        } catch (error) {
            console.error('Upload error:', error);
            statusEl.textContent = `✗ Error: ${error.message}`;
            statusEl.className = 'upload-status error';
        }
        
        setTimeout(() => {
            statusEl.textContent = '';
            statusEl.className = 'upload-status';
        }, 5000);
    }
    
    async downloadConversation() {
        try {
            const response = await fetch(`/download-conversation/${this.clientId}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `conversation_${this.clientId}_${new Date().toISOString().slice(0, 10)}.json`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showNotification('Percakapan berhasil didownload!', 'success');
            } else {
                const error = await response.text();
                throw new Error(error);
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showNotification('Error downloading conversation', 'error');
        }
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                const status = await response.json();
                this.updateSystemStatus(status);
            }
        } catch (error) {
            console.error('Error checking system status:', error);
        }
    }
    
    updateSystemStatus(status) {
        this.systemStatus = status;
        
        const systemStatusEl = document.getElementById('systemStatus');
        const isHealthy = status.status === 'healthy';
        
        systemStatusEl.innerHTML = `<i class="fas fa-${isHealthy ? 'check-circle' : 'exclamation-triangle'}"></i><span>${isHealthy ? 'Siap' : 'Loading'}</span>`;
        systemStatusEl.className = `status-indicator ${isHealthy ? 'connected' : 'loading'}`;
        
        document.getElementById('sttStatus').textContent = status.processors?.stt ? '✓ Ready' : '✗ Not Ready';
        document.getElementById('ttsStatus').textContent = status.processors?.tts ? '✓ Ready' : '✗ Not Ready';
        document.getElementById('docStatus').textContent = status.rag_status?.documents_loaded ? '✓ Loaded' : '○ Empty';
        document.getElementById('templateStatus').textContent = status.template_status?.templates_loaded ? `✓ ${status.template_status.template_count} loaded` : '○ Empty';
    }
    
    updateConnectionStatus(status) {
        const statusEl = document.getElementById('connectionStatus');
        const statusTexts = {
            'connected': 'Terhubung',
            'disconnected': 'Terputus',
            'error': 'Error',
            'connecting': 'Menghubungkan...'
        };
        
        statusEl.innerHTML = `<i class="fas fa-circle"></i><span>${statusTexts[status]}</span>`;
        statusEl.className = `status-indicator ${status}`;
    }
    
    updateRecordingUI(isRecording) {
        const button = document.getElementById('recordButton');
        const text = document.getElementById('recordText');
        const status = document.getElementById('recordStatus');
        
        if (isRecording) {
            button.classList.add('recording');
            text.textContent = 'Stop Rekam';
            status.textContent = 'Sedang merekam...';
        } else {
            button.classList.remove('recording');
            text.textContent = 'Mulai Rekam';
            status.textContent = 'Tekan untuk berbicara';
        }
    }
    
    addMessage(sender, text, type = '') {
        const messagesContainer = document.getElementById('chatMessages');
        
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message ${type}`;
        
        const iconMap = {
            'user': 'fas fa-user',
            'assistant': 'fas fa-robot',
            'system': 'fas fa-info-circle'
        };
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="${iconMap[sender] || 'fas fa-circle'}"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${this.formatMessage(text)}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        if (sender === 'system' && !type) {
            setTimeout(() => {
                if (messageDiv.parentNode) {
                    messageDiv.remove();
                }
            }, 10000);
        }
    }
    
    formatMessage(text) {
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }
    
    showProcessingMessage(stage) {
        this.removeProcessingMessage();
        
        const stageTexts = {
            'transcribing': 'Mentranskripsi audio...',
            'generating': 'Menghasilkan respons...',
            'synthesizing': 'Mensintesis suara...'
        };
        
        const messagesContainer = document.getElementById('chatMessages');
        const processingDiv = document.createElement('div');
        processingDiv.id = 'processingMessage';
        processingDiv.className = 'message processing-message';
        
        processingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-cog fa-spin"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${stageTexts[stage] || 'Memproses...'}</div>
            </div>
        `;
        
        messagesContainer.appendChild(processingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    removeProcessingMessage() {
        const processingMessage = document.getElementById('processingMessage');
        if (processingMessage) {
            processingMessage.remove();
        }
    }
    
    playAudioResponse(base64Audio) {
        try {
            const audio = new Audio('data:audio/wav;base64,' + base64Audio);
            audio.volume = 0.8;
            audio.play().catch(e => {
                console.error("Error playing audio:", e);
                this.showNotification('Error playing audio response', 'error');
            });
        } catch (e) {
            console.error("Error with audio response:", e);
        }
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => notification.classList.add('show'), 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    showWelcomeMessage() {
        this.addMessage('system', 'Selamat datang! Sistem telah siap. Anda dapat mulai berbicara atau mengetik pertanyaan.');
    }
    
    clearChat() {
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">
                    <i class="fas fa-robot"></i>
                </div>
                <h4>Chat telah dibersihkan</h4>
                <p>Mulai percakapan baru dengan berbicara atau mengetik.</p>
            </div>
        `;
    }
    
    refreshSystem() {
        this.checkSystemStatus();
        this.showNotification('System status refreshed', 'success');
    }
    
    showHelpModal() {
        document.getElementById('helpModal').classList.remove('hidden');
    }
    
    hideHelpModal() {
        document.getElementById('helpModal').classList.add('hidden');
    }
    
    initAudioVisualization() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.log('Audio visualization not supported');
        }
    }
    
    startAudioVisualization(stream) {
        if (!this.audioContext) return;
        
        try {
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            
            this.microphone.connect(this.analyser);
            
            this.visualize();
        } catch (e) {
            console.error('Error starting audio visualization:', e);
        }
    }
    
    visualize() {
        if (!this.analyser) return;
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const bars = document.querySelectorAll('#audioBars .bar');
        
        const draw = () => {
            if (!this.isRecording) return;
            
            this.analyser.getByteFrequencyData(dataArray);
            
            for (let i = 0; i < bars.length; i++) {
                const value = dataArray[i * Math.floor(bufferLength / bars.length)] || 0;
                const height = (value / 255) * 100;
                bars[i].style.height = Math.max(height, 5) + '%';
            }
            
            this.animationId = requestAnimationFrame(draw);
        };
        
        draw();
    }
    
    stopAudioVisualization() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        const bars = document.querySelectorAll('#audioBars .bar');
        bars.forEach(bar => {
            bar.style.height = '5%';
        });
        
        if (this.microphone) {
            this.microphone.disconnect();
            this.microphone = null;
        }
        
        if (this.analyser) {
            this.analyser = null;
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.voiceAssistant = new VoiceAssistant();
});

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        if (window.voiceAssistant) {
            window.voiceAssistant.checkSystemStatus();
        }
    }
});