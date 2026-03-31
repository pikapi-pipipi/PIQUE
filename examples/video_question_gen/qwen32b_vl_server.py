"""
Qwen3-VL-32B-Thinking 本地部署 HTTP Server
==========================================
使用 vLLM 部署，提供 OpenAI 兼容 API。

前置安装:
    pip install "vllm>=0.11.0" qwen-vl-utils

启动方式:
    python server.py

启动后可通过以下地址访问:
    - Chat API:          POST http://localhost:8000/v1/chat/completions
    - Transcription API:  POST http://localhost:8000/v1/audio/transcriptions  (如有音频)
    - 模型列表:           GET  http://localhost:8000/v1/models
    - 健康检查:           GET  http://localhost:8000/health
"""




import subprocess
import sys


# ========== 兼容补丁：必须在 vllm 任何导入之前执行 ==========
try:
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):

        @property
        def all_special_tokens_extended(self):
            return list(getattr(self, "all_special_tokens", []))

        Qwen2Tokenizer.all_special_tokens_extended = all_special_tokens_extended
        print("[Patch] Qwen2Tokenizer.all_special_tokens_extended 已注入")
except Exception as e:
    print(f"[Patch] 警告: patch 失败，可能影响启动: {e}")
# ===========================================================


def main():
    # ========== 配置区 ==========
    MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking"
    HOST = "0.0.0.0"
    PORT = 8088
    # GPU 数量 (32B 模型建议至少 2 张 24GB 显卡, 或 1 张 80GB 显卡)
    TENSOR_PARALLEL = 1
    # 最大模型长度 (根据显存调整)
    MAX_MODEL_LEN = 32768
    # GPU 显存利用率
    GPU_MEMORY_UTILIZATION = 0.90
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
        # 限制每个请求的图片数量和 token 数
        "--limit-mm-per-prompt", '{"image": 5, "video": 2}',
    ]

    print("=" * 60)
    print("  Qwen3-VL-32B-Thinking Server")
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


