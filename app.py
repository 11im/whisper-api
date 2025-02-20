# app.py
from flask import Flask, request, jsonify
import whisper
import torch
import os
import time
import logging
import sys
from datetime import datetime
from werkzeug.utils import secure_filename
import codecs

# stdout을 UTF-8로 재설정
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# 로깅 설정
class CustomFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def format(self, record):
        try:
            record.msg = record.msg.encode('cp949', errors='ignore').decode('cp949')
        except:
            pass
        return super().format(record)

# 로거 설정
logger = logging.getLogger('whisper_server')
logger.setLevel(logging.INFO)

# 콘솔 핸들러
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(CustomFormatter())
logger.addHandler(console_handler)

app = Flask(__name__)

# Whisper 모델 로드
logger.info("GPU 사용 가능 여부 확인 중...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"사용할 디바이스: {DEVICE}")

logger.info("Whisper Large 모델 로딩 중...")
start_time = time.time()
model = whisper.load_model("turbo", device=DEVICE)
load_time = time.time() - start_time
logger.info(f"모델 로딩 완료! (소요 시간: {load_time:.2f}초)")

# 파일 업로드 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB 제한

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"업로드 폴더 생성: {UPLOAD_FOLDER}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # 요청 시작 시간
    start_time = time.time()
    logger.info("새로운 음성 인식 요청 수신")

    # 파일 확인
    if 'file' not in request.files:
        logger.error("파일이 요청에 포함되지 않음")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("파일이 선택되지 않음")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.error(f"허용되지 않는 파일 형식: {file.filename}")
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # 파일 정보 출력
        logger.info("=== 파일 정보 ===")
        logger.info(f"원본 파일명: {file.filename}")
        logger.info(f"Content Type: {file.content_type}")
        
        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"저장할 파일 경로: {os.path.abspath(filepath)}")
        
        # 업로드 폴더 확인
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            logger.info(f"업로드 폴더가 없어 생성합니다: {app.config['UPLOAD_FOLDER']}")
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # 파일 저장
        logger.info("파일 저장 시도...")
        file.save(filepath)
        
        # 파일 저장 확인
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            logger.info(f"파일 저장 완료: {filename} (크기: {file_size/1024:.2f}KB)")
        else:
            raise FileNotFoundError(f"파일이 저장되지 않았습니다: {filepath}")
        
        # 음성 인식 수행
        logger.info("음성 인식 처리 시작...")
        transcribe_start = time.time()
        result = model.transcribe(filepath)
        transcribe_time = time.time() - transcribe_start
        logger.info(f"음성 인식 완료! (처리 시간: {transcribe_time:.2f}초)")
        
        # 파일 삭제
        os.remove(filepath)
        logger.info("임시 파일 삭제 완료")
        
        # 전체 처리 시간 계산
        total_time = time.time() - start_time
        logger.info(f"전체 요청 처리 완료 (총 소요 시간: {total_time:.2f}초)")
        
        return jsonify({
            'text': result['text'],
            'segments': result['segments'],
            'language': result['language'],
            'processing_time': {
                'transcribe': transcribe_time,
                'total': total_time
            }
        })
    
    except Exception as e:
        import traceback
        logger.error(f"처리 중 오류 발생: {str(e)}")
        logger.error(f"상세 에러 정보:\n{traceback.format_exc()}")
        
        # 에러 발생 시 임시 파일 삭제 시도
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info("에러 발생 후 임시 파일 삭제 완료")
        except:
            logger.error("임시 파일 삭제 실패")
            
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    memory_info = {}
    if torch.cuda.is_available():
        memory_info = {
            'allocated': f"{torch.cuda.memory_allocated(0)/1024**2:.2f}MB",
            'cached': f"{torch.cuda.memory_reserved(0)/1024**2:.2f}MB"
        }
    
    status_info = {
        'status': 'healthy',
        'model': 'whisper-large',
        'device': DEVICE,
        'gpu_info': gpu_info,
        'memory_info': memory_info,
        'uptime': time.time() - start_time,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info("상태 확인 요청 처리됨")
    return jsonify(status_info)

if __name__ == '__main__':
    logger.info("Whisper API 서버 시작...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
