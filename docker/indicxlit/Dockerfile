# AI4Bharat IndicXlit (Roman -> native script) in a container: it needs fairseq, which has no macOS/Windows build.
FROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "pip<24.1" && \
    pip install --no-cache-dir torch==2.0.1 && \
    pip install --no-cache-dir "numpy<2" ai4bharat-transliteration==1.1.3
# the official IndicXlit checkpoint is a full pickle; torch >= 2.6 refuses those by default
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
COPY xlit_server.py /app/xlit_server.py
# downloads the model weights into the image once
RUN python /app/xlit_server.py --warmup hi
ENTRYPOINT ["python", "/app/xlit_server.py"]
