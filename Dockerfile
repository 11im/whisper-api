# Dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 기본 패키지 설치 및 설정
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 생성
WORKDIR /app

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install -U openai-whisper
RUN pip3 install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

# 애플리케이션 코드 복사
COPY . .

# 업로드 디렉토리 생성
RUN mkdir -p uploads

# 포트 설정
EXPOSE 5000

# 실행 명령
CMD ["python3", "app.py"]
