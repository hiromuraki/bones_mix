#!/bin/bash

# ==========================================
# 1. 定义虚拟环境 site-packages 的基础路径
# ==========================================
SITE_PACKAGES=".venv/lib/python3.8/site-packages"

# ==========================================
# 2. 暴力拼接所有可能的 CUDA 库路径
# ==========================================
export LD_LIBRARY_PATH="$PWD/$SITE_PACKAGES/torch/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$PWD/$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$PWD/$SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$PWD/$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$PWD/$SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

# ==========================================
# 3. 自动修复 PyTorch 官方包缺失无版本号软链接的 Bug
# ==========================================
TORCH_LIB_DIR="$PWD/$SITE_PACKAGES/torch/lib"

if [ -d "$TORCH_LIB_DIR" ]; then
    # 如果不存在 libnvrtc.so，就尝试去建一个
    if [ ! -f "$TORCH_LIB_DIR/libnvrtc.so" ]; then
        # 寻找类似于 libnvrtc-xxx.so.11.2 的文件
        TARGET=$(ls "$TORCH_LIB_DIR"/libnvrtc-*.so.* 2>/dev/null | head -n 1)
        if [ -n "$TARGET" ]; then
            echo "🔧 检测到软链接缺失，正在自动修复 libnvrtc.so ..."
            # 创建相对路径的软链接
            ln -s "$(basename "$TARGET")" "$TORCH_LIB_DIR/libnvrtc.so"
        fi
    fi
fi

# ==========================================
# 4. 启动真正的 Python 进程
# ==========================================
echo "🚀 CUDA 环境变量配置完毕，正在启动..."

base_dir=$(dirname "$0")

cd "$base_dir/src" && uv run test.py
cd "$base_dir/tools/video_renderer/src" && uv run test.py
cd "$base_dir" && echo 处理完成