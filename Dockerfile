FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    pkg-config \
    libopencv-dev \
    libcjson-dev \
    v4l-utils \
    can-utils \
    curl \
    wget \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── onnxruntime (CPU) 설치 ──
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.17.0.tgz \
    && mv onnxruntime-linux-x64-1.17.0 /opt/onnxruntime \
    && rm onnxruntime-linux-x64-1.17.0.tgz

WORKDIR /workspace/dms_project
COPY . .
RUN mkdir -p build && cd build && \
    cmake .. -DONNXRUNTIME_ROOT=/opt/onnxruntime && \
    make -j$(nproc)

# =========================================================
# Runtime Stage
# =========================================================
FROM ubuntu:22.04 AS runner
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-videoio4.5d \
    libopencv-highgui4.5d \
    libopencv-dnn4.5d \
    libcjson1 \
    v4l-utils \
    can-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# onnxruntime 공유 라이브러리는 apt에 없으니 builder에서 그대로 가져옴
COPY --from=builder /opt/onnxruntime/lib/libonnxruntime.so* /usr/lib/
RUN ldconfig

COPY --from=builder /workspace/dms_project/build/dms_node /app/dms_node
COPY --from=builder /workspace/dms_project/models /app/models
COPY --from=builder /workspace/dms_project/configs /app/configs

CMD ["/bin/bash"]