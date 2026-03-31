#!/usr/bin/env python3
"""
启动 Qwen3-14B 的 OpenAI 兼容推理服务，可供任意任务调用（如独立语义评估、其他文本生成）。
用法: python run_qwen14b_server.py
"""
import subprocess
import sys


def main():
    # ========== 配置区 ==========
    MODEL_NAME = "Qwen/Qwen3-14B"
    HOST = "0.0.0.0"
    PORT = 8091  # 与 VL/Guard 错开端口
    TENSOR_PARALLEL = 1
    MAX_MODEL_LEN = 32768
    GPU_MEMORY_UTILIZATION = 0.45
    # ============================

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", HOST,
        "--port", str(PORT),
        "--tensor-parallel-size", str(TENSOR_PARALLEL),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--max-num-seqs", "256",
    ]

    print("=" * 60)
    print("  Qwen3-14B Server (OpenAI-compatible)")
    print("=" * 60)
    print(f"  模型:   {MODEL_NAME}")
    print(f"  地址:   http://{HOST}:{PORT}")
    print(f"  GPU:    {TENSOR_PARALLEL} 张 (tensor parallel)")
    print(f"  上下文: {MAX_MODEL_LEN} tokens")
    print("=" * 60)
    print()
    print("启动命令:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
