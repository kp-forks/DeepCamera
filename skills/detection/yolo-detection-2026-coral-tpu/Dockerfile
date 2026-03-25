# ─── Coral TPU Detection — Runtime Image ─────────────────────────────────────
# Runs YOLO inference on Google Coral Edge TPU via pycoral/tflite-runtime.
# Built locally on user's machine. No external model registry.
#
# Build:  docker build -t aegis-coral-tpu .
# Run:    docker run -i --rm --device /dev/bus/usb \
#           -v /tmp/aegis_detection:/tmp/aegis_detection \
#           aegis-coral-tpu

FROM debian:bullseye-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ─── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        gnupg2 \
        ca-certificates \
        curl \
        usbutils \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ─── Add Google Coral package repository ─────────────────────────────────────
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key add - \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

# ─── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# ─── Application code ────────────────────────────────────────────────────────
COPY scripts/ /app/scripts/
COPY models/ /app/models/

WORKDIR /app

# ─── Shared volume for frame exchange ────────────────────────────────────────
VOLUME ["/tmp/aegis_detection"]

# ─── Entry point ─────────────────────────────────────────────────────────────
# stdin/stdout JSONL protocol — same as yolo-detection-2026
ENTRYPOINT ["python3", "scripts/detect.py"]
