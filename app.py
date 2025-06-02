#!/usr/bin/env python3
"""
Advanced Flask Web Interface for Diffusion Model Training
Dynamic, animated UI for training Wan2.1 and FLUX models
"""

import sys
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import zipfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from dataset_auto import prepare_dataset
    from train_unified import train_model
    from config_manager import load_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diffusion-training-secret-key'
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
training_sessions = {}
active_jobs = {}

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path('./outputs').mkdir(exist_ok=True)
Path('./data').mkdir(exist_ok=True)


class TrainingSession:
    """Manages a training session"""

    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.status = "preparing"
        self.progress = 0
        self.logs = []
        self.start_time = datetime.now()
        self.end_time = None
        self.error = None
        self.model_path = None

    def add_log(self, message: str, level: str = "info"):
        """Add log message"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)

        # Emit to websocket
        socketio.emit('training_log', {
            'session_id': self.session_id,
            'log': log_entry
        })

    def update_progress(self, progress: int, status: str = None):
        """Update training progress"""
        self.progress = progress
        if status:
            self.status = status

        # Emit to websocket
        socketio.emit('training_progress', {
            'session_id': self.session_id,
            'progress': self.progress,
            'status': self.status
        })

    def complete(self, model_path: str = None, error: str = None):
        """Mark training as complete"""
        self.end_time = datetime.now()
        if error:
            self.status = "failed"
            self.error = error
        else:
            self.status = "completed"
            self.model_path = model_path

        # Emit completion
        socketio.emit('training_complete', {
            'session_id': self.session_id,
            'status': self.status,
            'error': self.error,
            'model_path': self.model_path
        })


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/train')
def train_page():
    """Training configuration page"""
    return render_template('train.html')


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        upload_id = str(uuid.uuid4())
        upload_dir = Path(app.config['UPLOAD_FOLDER']) / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files = []

        for file in files:
            if file.filename == '':
                continue

            filename = secure_filename(file.filename)
            file_path = upload_dir / filename
            file.save(file_path)
            uploaded_files.append(filename)

        return jsonify({
            'upload_id': upload_id,
            'files': uploaded_files,
            'count': len(uploaded_files)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def start_training():
    """Start training process"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['model_type', 'upload_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Create session
        session_id = str(uuid.uuid4())
        config = {
            'model_type': data['model_type'],
            'upload_id': data['upload_id'],
            'training_params': data.get('training_params', {}),
            'dataset_params': data.get('dataset_params', {}),
            'output_dir': f'./outputs/{session_id}'
        }

        session = TrainingSession(session_id, config)
        training_sessions[session_id] = session

        # Start training in background
        thread = threading.Thread(target=run_training, args=(session,))
        thread.daemon = True
        thread.start()

        return jsonify({
            'session_id': session_id,
            'status': 'started'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<session_id>')
def get_status(session_id):
    """Get training status"""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = training_sessions[session_id]

    return jsonify({
        'session_id': session_id,
        'status': session.status,
        'progress': session.progress,
        'start_time': session.start_time.isoformat(),
        'end_time': session.end_time.isoformat() if session.end_time else None,
        'error': session.error,
        'model_path': session.model_path,
        'logs': session.logs[-10:]  # Last 10 logs
    })


@app.route('/api/logs/<session_id>')
def get_logs(session_id):
    """Get full training logs"""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = training_sessions[session_id]
    return jsonify({'logs': session.logs})


@app.route('/api/download/<session_id>')
def download_model(session_id):
    """Download trained model"""
    if session_id not in training_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = training_sessions[session_id]

    if session.status != 'completed' or not session.model_path:
        return jsonify({'error': 'Model not ready'}), 400

    # Create zip file
    zip_path = f'./outputs/{session_id}_model.zip'

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        model_dir = Path(session.model_path)
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(model_dir.parent))

    return send_file(zip_path, as_attachment=True, download_name=f'{session.config["model_type"]}_model.zip')


@app.route('/monitor')
def monitor():
    """Training monitoring page"""
    return render_template('monitor.html', sessions=training_sessions)


@app.route('/api/sessions')
def get_sessions():
    """Get all training sessions"""
    sessions_data = []
    for session_id, session in training_sessions.items():
        sessions_data.append({
            'session_id': session_id,
            'model_type': session.config['model_type'],
            'status': session.status,
            'progress': session.progress,
            'start_time': session.start_time.isoformat(),
            'end_time': session.end_time.isoformat() if session.end_time else None
        })

    return jsonify({'sessions': sessions_data})


def run_training(session: TrainingSession):
    """Run training process in background"""
    try:
        session.add_log("Starting training process...")
        session.update_progress(5, "preparing")

        # Prepare dataset
        upload_dir = Path(app.config['UPLOAD_FOLDER']) / session.config['upload_id']
        data_dir = f'./data/{session.session_id}'

        session.add_log("Preparing dataset with automatic captioning...")
        session.update_progress(10, "dataset_preparation")

        # Dataset preparation
        dataset_params = session.config.get('dataset_params', {})
        stats = prepare_dataset(
            input_dir=str(upload_dir),
            output_dir=data_dir,
            model_type=session.config['model_type'],
            captioning_model=dataset_params.get('captioning_model', 'blip2'),
            min_resolution=dataset_params.get('min_resolution', 512),
            max_resolution=dataset_params.get('max_resolution', 2048),
            quality_threshold=dataset_params.get('quality_threshold', 0.7),
            batch_size=8
        )

        session.add_log(f"Dataset prepared: {stats['processed']} images processed")
        session.update_progress(30, "loading_model")

        # Load training config
        config = load_config('config/unified_config.yaml')

        # Apply user parameters
        training_params = session.config.get('training_params', {})
        lora_params = session.config.get('lora_params', {})

        config['dataset']['train_data_dir'] = data_dir
        config['training']['output_dir'] = session.config['output_dir']

        # Apply training parameters
        if 'batch_size' in training_params:
            config['training']['train_batch_size'] = training_params['batch_size']
        if 'learning_rate' in training_params:
            config['training']['learning_rate'] = training_params['learning_rate']
        if 'max_steps' in training_params:
            config['training']['max_train_steps'] = training_params['max_steps']
        if 'mixed_precision' in training_params:
            config['training']['mixed_precision'] = training_params['mixed_precision']

        # Apply LoRA parameters
        if lora_params:
            config['model']['use_lora'] = True
            if 'lora_type' in lora_params:
                config['model']['lora_type'] = lora_params['lora_type']
            if 'lora_rank' in lora_params:
                config['model']['lora_rank'] = lora_params['lora_rank']
            if 'lora_alpha' in lora_params:
                config['model']['lora_alpha'] = lora_params['lora_alpha']
            if 'lora_dropout' in lora_params:
                config['model']['lora_dropout'] = lora_params['lora_dropout']

            # Handle QLoRA
            if lora_params.get('lora_type') == 'qlora':
                config['model']['use_qlora'] = True
            elif lora_params.get('lora_type') == 'dora':
                config['model']['use_dora'] = True

        session.add_log("Starting model training...")
        session.update_progress(40, "training")

        # Start training
        final_step = train_model(
            config=config,
            model_type=session.config['model_type'],
            variable_size=True
        )

        session.add_log(f"Training completed at step {final_step}")
        session.complete(model_path=config['training']['output_dir'])

    except Exception as e:
        session.add_log(f"Training failed: {str(e)}", "error")
        session.complete(error=str(e))


@socketio.on('connect')
def handle_connect():
    """Handle websocket connection"""
    emit('connected', {'status': 'Connected to training server'})


@socketio.on('subscribe_session')
def handle_subscribe(data):
    """Subscribe to session updates"""
    session_id = data.get('session_id')
    if session_id in training_sessions:
        emit('subscribed', {'session_id': session_id})


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    Path('./templates').mkdir(exist_ok=True)
    Path('./static').mkdir(exist_ok=True)

    print("🚀 Starting Diffusion Training Web Interface...")
    print("📱 Access the interface at: http://localhost:5000")

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
