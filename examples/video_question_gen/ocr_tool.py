# examples/video_question_gen/ocr_tool.py
# OCR 工具：从视频关键帧中识别文字，辅助问题生成。
# 使用条件：在 video_slice 和/或 entity_search 之后仍难以生成高质量问题时调用。
# 用法：在 config/tool_config.yaml 中 class_name 指向本模块 OCRTool。
# 可选：配置 api_base + api_model 使用远程/本地 VL 服务（如 Qwen3-VL）做 OCR，否则走本地 PaddleOCR。

import base64
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

# 关键帧采样间隔（秒），可根据视频长度调节
KEYFRAME_INTERVAL = 2.0
# VL API 单次请求最大图片数（需与 vLLM --limit-mm-per-prompt 的 image 一致，如 5）
MAX_OCR_IMAGES_PER_REQUEST = 5

# 本地 PaddleOCR 只初始化一次，多轮 ocr 调用复用
_paddle_ocr = None


def _get_paddle_ocr():
    """返回缓存的 PaddleOCR 实例，首次调用时创建并缓存。"""
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        try:
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        except (TypeError, Exception) as init_e:
            if "show_log" in str(init_e).lower() or "unknown argument" in str(init_e).lower():
                _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch")
            else:
                raise
    return _paddle_ocr


def _get_openai_client(api_base: str, api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key or "dummy", base_url=api_base)


def _run_ocr_via_vl_api(
    image_paths: list[str],
    api_base: str,
    api_key: str,
    api_model: str,
) -> str:
    """用 VL 服务（如 Qwen3-VL）对多张图片做 OCR，合并为一段文本。单次请求最多 MAX_OCR_IMAGES_PER_REQUEST 张。"""
    if not image_paths or not api_base or not api_model:
        return ""
    all_lines = []
    for i in range(0, len(image_paths), MAX_OCR_IMAGES_PER_REQUEST):
        batch = image_paths[i : i + MAX_OCR_IMAGES_PER_REQUEST]
        content = []
        for path in batch:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({
            "type": "text",
            "text": "请识别以上每张图片中的所有文字（字幕、标题、画面内文字），按图片顺序逐条输出，每条之间用换行分隔。若无文字请输出「无」。/no_think",
        })
        try:
            client = _get_openai_client(api_base, api_key)
            response = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
                temperature=0.1,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                all_lines.append(text)
        except Exception as e:
            logger.warning("ocr: VL API failed for batch: %s", e)
            all_lines.append(f"【本批识别失败: {e}】")
    return "\n\n".join(all_lines) if all_lines else "未识别到文字。"


def _extract_keyframes(video_path: str, interval: float = KEYFRAME_INTERVAL) -> list[str]:
    """使用 ffmpeg 按间隔抽取关键帧，返回临时图片路径列表。"""
    paths = []
    try:
        out_dir = tempfile.mkdtemp(prefix="ocr_frames_")
        # -vsync vfr：按时间戳取帧；-q:v 2 控制质量
        out_pattern = os.path.join(out_dir, "frame_%04d.jpg")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps=1/{interval}",
                "-q:v", "2",
                out_pattern,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        for name in sorted(os.listdir(out_dir)):
            if name.endswith(".jpg"):
                paths.append(os.path.join(out_dir, name))
        return paths
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("ocr: ffmpeg extract keyframes failed for %s: %s", video_path, e)
        return []


def _run_ocr_on_images(
    image_paths: list[str],
    api_base: str = "",
    api_key: str = "",
    api_model: str = "",
) -> str:
    """
    对多张图片做 OCR，合并为一段文本。
    - 若在 config 或 create_kwargs 中提供了 api_base + api_model，则使用 VL 服务（如 Qwen3-VL）做 OCR。
    - 否则若已安装 PaddleOCR，则使用本地 PaddleOCR。
    """
    if not image_paths:
        return ""
    if api_base and api_model:
        return _run_ocr_via_vl_api(image_paths, api_base, api_key, api_model)
    try:
        ocr = _get_paddle_ocr()

        def _run_one(img_path: str):
            try:
                return ocr.ocr(img_path, cls=True)
            except TypeError as te:
                if "cls" in str(te).lower():
                    return ocr.ocr(img_path)
                raise

        lines = []
        for path in image_paths:
            result = _run_one(path)
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        lines.append(line[1][0])
        return "\n".join(lines) if lines else "未识别到文字。"
    except ImportError:
        return "【OCR】未安装 PaddleOCR。请执行: pip install paddlepaddle paddleocr。或在此函数中接入 EasyOCR/远程 OCR API。"
    except Exception as e:
        logger.warning("ocr: PaddleOCR failed: %s", e)
        return f"【OCR】识别失败: {e}"


def _cleanup_frames(paths: list[str], out_dir: str | None) -> None:
    if not paths and not out_dir:
        return
    for p in paths or []:
        if os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass
    if out_dir and os.path.isdir(out_dir):
        try:
            os.rmdir(out_dir)
        except OSError:
            pass


class OCRTool(BaseTool):
    """
    从视频关键帧中识别文字（字幕、标题、画面内文字），用于在 video_slice/entity_search 之后辅助问题生成。
    create_kwargs 需提供 video_path；可选 api_base、api_key、api_model 使用 VL 服务（如 Qwen3-VL）做 OCR。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}
        self._api_base = config.get("api_base", "")
        self._api_key = config.get("api_key", "")
        self._api_model = config.get("api_model", "")
        self._interval = config.get("keyframe_interval", KEYFRAME_INTERVAL)

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
        api_model = create_kwargs.get("api_model") or self._api_model
        self._instance_dict[instance_id] = {
            "video_path": video_path,
            "api_base": api_base,
            "api_key": api_key,
            "api_model": api_model,
            "interval": create_kwargs.get("keyframe_interval", self._interval),
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
        api_model = state.get("api_model", "")
        interval = state.get("interval", KEYFRAME_INTERVAL)

        if not video_path or not os.path.exists(video_path):
            return ToolResponse(text="Error: video_path missing or file not found."), 0.0, {}

        frame_paths = []
        out_dir = None
        try:
            frame_paths = _extract_keyframes(video_path, interval=interval)
            if not frame_paths:
                return ToolResponse(text="Error: failed to extract keyframes (ffmpeg required)."), 0.0, {}
            if frame_paths:
                out_dir = os.path.dirname(frame_paths[0])
            text = _run_ocr_on_images(
                frame_paths,
                api_base=api_base,
                api_key=api_key,
                api_model=api_model,
            )
            return ToolResponse(text=f"【OCR 关键帧文字】\n\n{text}"), 0.0, {}
        except Exception as e:
            logger.exception("ocr: execute failed")
            return ToolResponse(text=f"Error: OCR failed: {e}"), 0.0, {}
        finally:
            _cleanup_frames(frame_paths, out_dir)

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)





