FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3-pip && pip install --upgrade pip
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
COPY req.txt .
RUN pip install -r req.txt
EXPOSE 7860
CMD ["python3", "gradio_demo/app.py"]