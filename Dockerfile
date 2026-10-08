FROM python:3.11-slim

# libmagic1 is needed by python-magic for byte-level MIME sniffing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Make OpenCV refuse to decode images over 40 MP from the header alone
# (decompression-bomb guard; also set in app/splitter.py).
ENV OPENCV_IO_MAX_IMAGE_PIXELS=40000000

# Don't run as root.
RUN useradd --system --no-create-home --uid 10001 app
USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
