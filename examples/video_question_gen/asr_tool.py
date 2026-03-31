# examples/video_question_gen/asr_tool.py
# ASR 工具：对视频整段音频进行转录，辅助问题生成。
# 使用条件：在 video_slice 和/或 entity_search 之后仍难以生成高质量问题时调用。
# 用法：在 config/tool_config.yaml 中 class_name 指向本模块 ASRTool。

import logging
import os
import subprocess
import tempfile
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# 本地 Whisper 模型只加载一次，多轮 asr 调用复用
_whisper_model = None


def _get_whisper_model():
    """返回缓存的 Whisper base 模型，首次调用时加载并缓存。"""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _extract_audio_to_wav(video_path: str) -> str | None:
    """使用 ffmpeg 从视频提取音频为临时 wav 文件，返回临时路径。失败返回 None。"""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = tmp.name
        tmp.close()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return path
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("asr: ffmpeg extract audio failed for %s: %s", video_path, e)
        return None


def _transcribe_audio(audio_path: str, api_base: str = "", api_key: str = "") -> str:
    """
    对音频文件进行转录。
    - 若已安装 openai-whisper（pip install openai-whisper），会优先使用本地 Whisper，无需配置。
    - 若在 config 或 create_kwargs 中提供了 api_base + api_key，可在此接入兼容 OpenAI 的语音识别 API。
    """
    if api_base and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_base)
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(model="whisper-1", file=f)
            return (response.text or "").strip()
        except Exception as e:
            logger.warning("asr: remote API failed: %s", e)
    try:
        model = _get_whisper_model()
        result = model.transcribe(audio_path, language="zh", fp16=False)
        return (result.get("text") or "").strip()
    except ImportError:
        return "【ASR】未安装 Whisper。请执行: pip install openai-whisper。或配置 api_base/api_key 使用远程 ASR API。"
    except Exception as e:
        logger.warning("asr: whisper failed: %s", e)
        return f"【ASR】转录失败: {e}"


class ASRTool(BaseTool):
    """
    对整段视频音频进行转录，返回文本，用于在 video_slice/entity_search 之后辅助问题生成。
    create_kwargs 需提供 video_path；可选 api_base、api_key 用于远程 ASR。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}
        self._api_base = config.get("api_base", "")
        self._api_key = config.get("api_key", "")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        video_path: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        video_path = video_path or create_kwargs.get("video_path")
        api_base = create_kwargs.get("api_base") or self._api_base
        api_key = create_kwargs.get("api_key") or self._api_key
        self._instance_dict[instance_id] = {
            "video_path": video_path,
            "api_base": api_base,
            "api_key": api_key,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instance_dict.get(instance_id, {})
        video_path = state.get("video_path")
        api_base = state.get("api_base", "")
        api_key = state.get("api_key", "")

        if not video_path or not os.path.exists(video_path):
            return ToolResponse(text="Error: video_path missing or file not found."), 0.0, {}

        audio_path = None
        try:
            audio_path = _extract_audio_to_wav(video_path)
            if not audio_path:
                return ToolResponse(text="Error: failed to extract audio from video (ffmpeg required)."), 0.0, {}
            text = _transcribe_audio(audio_path, api_base=api_base, api_key=api_key)
            return ToolResponse(text=f"【ASR 转录结果】\n\n{text}"), 0.0, {}
        except Exception as e:
            logger.exception("asr: execute failed")
            return ToolResponse(text=f"Error: ASR failed: {e}"), 0.0, {}
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
