# examples/video_question_gen/video_slice_tool.py
# 自定义工具：使用 TransNetV2 检测视频转场，按场景切分后分别调用远程大模型生成描述，
# 最终拼接成 slice1: [时间范围] 描述 的格式。
# 用法：在 config/tool_config.yaml 中 class_name 指向本模块 VideoSliceTool。
'''
import asyncio
import base64
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MAX_DATA_URI_BYTES = 10 * 1024 * 1024
MAX_VIDEO_SIZE_BYTES = 7 * 1024 * 1024

_RE_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """移除模型返回中的 <think>...</think> 部分，与项目其他模块保持一致。"""
    return _RE_THINK_BLOCK.sub('', text).strip()


# TransNetV2 全局单例，避免重复加载权重
_transnet_model = None


def _get_transnet_model():
    global _transnet_model
    if _transnet_model is None:
        from transnetv2_pytorch import TransNetV2
        _transnet_model = TransNetV2(device="auto")
        _transnet_model.eval()
    return _transnet_model


def _get_openai_client(api_base: str, api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=api_base)


def encode_video_to_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _check_ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=False)
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def get_video_duration(video_path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip() or "0.0")
    except Exception:
        return 0.0


def compress_video(video_path: str, target_size_mb: float = 5.0) -> str:
    file_size = os.path.getsize(video_path)
    if file_size <= MAX_VIDEO_SIZE_BYTES:
        return video_path

    if not _check_ffmpeg_available():
        logger.warning(
            "ffprobe/ffmpeg not found; skipping video compression. "
            "Install ffmpeg to compress large videos (e.g. apt-get install ffmpeg)."
        )
        return video_path

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("ffprobe failed for %s: %s; skipping compression", video_path, e)
        return video_path

    duration = float(result.stdout.strip() or "1.0")
    target_bitrate = int((target_size_mb * 8 * 1024) / duration)

    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-b:v", f"{target_bitrate}k",
                "-vf", "scale='min(720,iw)':'-2'",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-b:a", "64k",
                "-movflags", "+faststart",
                tmp_path,
            ],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("ffmpeg compression failed for %s: %s; using original", video_path, e)
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return video_path

    return tmp_path


def detect_scenes(video_path: str, threshold: float = 0.5,
                  min_scene_len: float = 1.0) -> list[tuple[float, float]]:
    """使用 TransNetV2 检测视频转场，返回 [(start_sec, end_sec), ...] 场景列表。

    Args:
        video_path: 视频文件路径
        threshold: 转场检测阈值，越小越敏感
        min_scene_len: 最短场景时长（秒），短于此值的场景会被合并到前一个
    """
    model = _get_transnet_model()
    scenes = model.detect_scenes(video_path, threshold=threshold)

    duration = get_video_duration(video_path)

    if not scenes:
        return [(0.0, duration)] if duration > 0 else [(0.0, 1.0)]

    raw_ranges: list[tuple[float, float]] = []
    for scene in scenes:
        start = float(scene["start_time"])
        end = float(scene["end_time"])
        if end > start:
            raw_ranges.append((start, end))

    if not raw_ranges:
        return [(0.0, duration)] if duration > 0 else [(0.0, 1.0)]

    raw_ranges.sort(key=lambda x: x[0])

    merged: list[tuple[float, float]] = [raw_ranges[0]]
    for start, end in raw_ranges[1:]:
        if (end - start) < min_scene_len and merged:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return merged


def extract_segment(video_path: str, start: float, end: float,
                    target_size_mb: float = 5.0) -> str:
    """用 ffmpeg 从视频中裁剪出 [start, end] 片段并压缩，返回临时文件路径。"""
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    segment_duration = end - start
    target_bitrate = int((target_size_mb * 8 * 1024) / max(segment_duration, 0.5))

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{segment_duration:.3f}",
                "-b:v", f"{target_bitrate}k",
                "-vf", "scale='min(720,iw)':'-2'",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-b:a", "64k",
                "-movflags", "+faststart",
                tmp_path,
            ],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("ffmpeg segment extraction failed: %s", e)
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return ""

    if os.path.getsize(tmp_path) > MAX_VIDEO_SIZE_BYTES:
        compressed = compress_video(tmp_path, target_size_mb=target_size_mb)
        if compressed != tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return compressed

    return tmp_path


def _format_time(seconds: float) -> str:
    """将秒数格式化为 M:SS 或 H:MM:SS 形式。"""
    seconds = max(0.0, seconds)
    total_sec = int(round(seconds))
    h, remainder = divmod(total_sec, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


SEGMENT_SYSTEM_PROMPT = """\
你是一个专业的视频分析助手。请仔细观看用户提供的视频片段，
为该片段生成详细、信息丰富的描述。

要求：
1. 描述画面中的关键元素、人物动作、场景变化
2. 注意对话、文字、音效等信息
3. 描述应当简洁准确，但包含足够的细节以便理解该片段的内容
4. 直接输出描述内容，不要添加时间范围或编号前缀
/no_think
"""


class VideoSliceTool(BaseTool):
    """
    视频分段工具：先用 TransNetV2 检测转场，将视频切分为场景片段，
    再分别调用远程大模型对每个片段生成详细描述，最终拼接成标准输出格式。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}
        self._api_base = config.get("api_base", "")
        self._api_key = config.get("api_key", "")
        self._api_model = config.get("api_model", "qwen3-vl-plus")
        self._scene_threshold = config.get("scene_threshold", 0.5)
        self._min_scene_len = config.get("min_scene_len", 1.0)

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
        }
        return instance_id, ToolResponse()

    def _describe_segment(
        self, client, api_model: str, video_b64: str,
        start: float, end: float,
    ) -> str:
        """调用远程大模型对单个视频片段生成描述。"""
        time_range = f"{_format_time(start)}-{_format_time(end)}"
        try:
            response = client.chat.completions.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                            },
                            {
                                "type": "text",
                                "text": f"这是视频的 [{time_range}] 片段，请为它生成详细描述。",
                            },
                        ],
                    },
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            raw = (response.choices[0].message.content or "").strip()
            return _strip_think_tags(raw)
        except Exception as e:
            logger.warning("Segment [%s] API call failed: %s", time_range, e)
            return f"(描述生成失败: {e})"

    def _process_one_segment(
        self,
        video_path: str,
        client,
        api_model: str,
        idx: int,
        start: float,
        end: float,
    ) -> tuple[int, str, Optional[str]]:
        """单片段处理：裁剪 → 编码 → 调用 LLM 描述。返回 (idx, result_line, segment_path_for_cleanup)。"""
        time_range = f"{_format_time(start)}-{_format_time(end)}"
        segment_path: Optional[str] = None
        try:
            segment_path = extract_segment(video_path, start, end)
            if not segment_path:
                return (idx, f"slice{idx}: [{time_range}] (片段提取失败)", None)
            try:
                video_b64 = encode_video_to_base64(segment_path)
            except Exception as e:
                logger.warning("Failed to encode segment %d: %s", idx, e)
                return (idx, f"slice{idx}: [{time_range}] (片段编码失败: {e})", segment_path)
            if len(video_b64) > MAX_DATA_URI_BYTES:
                return (idx, f"slice{idx}: [{time_range}] (片段过大，跳过)", segment_path)
            description = self._describe_segment(
                client, api_model, video_b64, start, end,
            )
            return (idx, f"slice{idx}: [{time_range}] {description}", segment_path)
        except Exception as e:
            logger.warning("Segment %d [%s] failed: %s", idx, time_range, e)
            return (
                idx,
                f"slice{idx}: [{time_range}] (处理失败: {e})",
                segment_path,
            )

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instance_dict.get(instance_id, {})
        video_path = state.get("video_path")
        api_base = state.get("api_base")
        api_key = state.get("api_key")
        api_model = state.get("api_model")

        if not video_path or not os.path.exists(video_path):
            return ToolResponse(text="Error: video_path missing or file not found."), 0.0, {}
        if not api_base:
            return ToolResponse(text="Error: api_base not configured for video_slice tool."), 0.0, {}
        if api_key is None:
            api_key = ""

        # Step 1: TransNetV2 场景检测
        try:
            scenes = detect_scenes(
                video_path,
                threshold=self._scene_threshold,
                min_scene_len=self._min_scene_len,
            )
            logger.info("TransNetV2 detected %d scenes for %s", len(scenes), video_path)
        except Exception as e:
            logger.exception("TransNetV2 scene detection failed, falling back to whole-video mode")
            duration = get_video_duration(video_path)
            scenes = [(0.0, duration if duration > 0 else 1.0)]

        # Step 2: 并行处理各片段（裁剪 → 编码 → 调用 LLM 描述）
        client = _get_openai_client(api_base, api_key)
        slice_results: list[str] = []
        tmp_files: list[str] = []
        try:
            tasks = [
                asyncio.to_thread(
                    self._process_one_segment,
                    video_path,
                    client,
                    api_model,
                    idx,
                    start,
                    end,
                )
                for idx, (start, end) in enumerate(scenes, 1)
            ]
            results = await asyncio.gather(*tasks)
            # 按 idx 排序保证输出顺序与场景顺序一致
            slice_results = [line for _, line, _ in sorted(results, key=lambda x: x[0])]
            for _, _, seg_path in results:
                if seg_path:
                    tmp_files.append(seg_path)
        finally:
            for f in tmp_files:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except OSError:
                        pass

        if not slice_results:
            return ToolResponse(text="Error: no scene segments could be processed."), 0.0, {}

        return ToolResponse(text="\n".join(slice_results)), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
    '''

    # examples/video_question_gen/video_slice_tool.py
# 自定义工具：使用 TransNetV2 检测视频转场，按场景切分后对每个片段均匀抽取10帧图片，
# 发送给远程大模型生成描述，最终拼接成 slice1: [时间范围] 描述 的格式。
# 用法：在 config/tool_config.yaml 中 class_name 指向本模块 VideoSliceTool。

import asyncio
import base64
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

NUM_FRAMES_PER_SEGMENT = 5

_RE_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """移除模型返回中的 <think>...</think> 部分，与项目其他模块保持一致。"""
    return _RE_THINK_BLOCK.sub('', text).strip()


_transnet_model = None


def _get_transnet_model():
    global _transnet_model
    if _transnet_model is None:
        from transnetv2_pytorch import TransNetV2
        _transnet_model = TransNetV2(device="auto")
        _transnet_model.eval()
    return _transnet_model


def _get_openai_client(api_base: str, api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=api_base)


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _check_ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=False)
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def get_video_duration(video_path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip() or "0.0")
    except Exception:
        return 0.0


def detect_scenes(video_path: str, threshold: float = 0.5,
                  min_scene_len: float = 1.0) -> list[tuple[float, float]]:
    """使用 TransNetV2 检测视频转场，返回 [(start_sec, end_sec), ...] 场景列表。

    Args:
        video_path: 视频文件路径
        threshold: 转场检测阈值，越小越敏感
        min_scene_len: 最短场景时长（秒），短于此值的场景会被合并到前一个
    """
    model = _get_transnet_model()
    scenes = model.detect_scenes(video_path, threshold=threshold)

    duration = get_video_duration(video_path)

    if not scenes:
        return [(0.0, duration)] if duration > 0 else [(0.0, 1.0)]

    raw_ranges: list[tuple[float, float]] = []
    for scene in scenes:
        start = float(scene["start_time"])
        end = float(scene["end_time"])
        if end > start:
            raw_ranges.append((start, end))

    if not raw_ranges:
        return [(0.0, duration)] if duration > 0 else [(0.0, 1.0)]

    raw_ranges.sort(key=lambda x: x[0])

    merged: list[tuple[float, float]] = [raw_ranges[0]]
    for start, end in raw_ranges[1:]:
        if (end - start) < min_scene_len and merged:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return merged


def extract_frames(video_path: str, start: float, end: float,
                   num_frames: int = NUM_FRAMES_PER_SEGMENT) -> list[str]:
    """从视频的 [start, end] 区间均匀抽取 num_frames 帧，返回临时 JPEG 文件路径列表。"""
    duration = end - start
    if duration <= 0:
        return []

    if num_frames == 1:
        timestamps = [start + duration / 2]
    else:
        timestamps = [start + i * duration / (num_frames - 1) for i in range(num_frames)]

    frame_paths: list[str] = []
    for ts in timestamps:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{ts:.3f}",
                    "-i", video_path,
                    "-vframes", "1",
                    "-vf", "scale='min(720,iw)':'-2'",
                    "-q:v", "2",
                    tmp_path,
                ],
                capture_output=True, text=True, check=True,
            )
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                frame_paths.append(tmp_path)
            else:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("Failed to extract frame at %.3fs: %s", ts, e)
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return frame_paths


def _format_time(seconds: float) -> str:
    """将秒数格式化为 M:SS 或 H:MM:SS 形式。"""
    seconds = max(0.0, seconds)
    total_sec = int(round(seconds))
    h, remainder = divmod(total_sec, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


SEGMENT_SYSTEM_PROMPT = """\
你是一个专业的视频分析助手。用户会提供从某个视频片段中均匀抽取的若干帧画面截图，
请根据这些截图推断并生成该片段详细、信息丰富的描述。

要求：
1. 描述画面中的关键元素、人物动作、场景变化
2. 注意画面中出现的文字信息
3. 描述应当简洁准确，但包含足够的细节以便理解该片段的内容
4. 直接输出描述内容，不要添加时间范围或编号前缀
/no_think
"""


class VideoSliceTool(BaseTool):
    """
    视频分段工具：先用 TransNetV2 检测转场，将视频切分为场景片段，
    对每个片段均匀抽取10帧图片发送给远程大模型生成描述，最终拼接成标准输出格式。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}
        self._api_base = config.get("api_base", "")
        self._api_key = config.get("api_key", "")
        self._api_model = config.get("api_model", "qwen3-vl-plus")
        self._scene_threshold = config.get("scene_threshold", 0.5)
        self._min_scene_len = config.get("min_scene_len", 1.0)

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
        }
        return instance_id, ToolResponse()

    def _describe_segment(
        self, client, api_model: str, frames_b64: list[str],
        start: float, end: float,
    ) -> str:
        """调用远程大模型，基于抽取的帧图片对单个视频片段生成描述。"""
        time_range = f"{_format_time(start)}-{_format_time(end)}"
        try:
            content: list[dict] = []
            for frame_b64 in frames_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                })
            content.append({
                "type": "text",
                "text": (
                    f"以上是视频 [{time_range}] 片段中均匀抽取的 {len(frames_b64)} 帧截图，"
                    "请根据这些截图为该片段生成详细描述。"
                ),
            })

            response = client.chat.completions.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            raw = (response.choices[0].message.content or "").strip()
            return _strip_think_tags(raw)
        except Exception as e:
            logger.warning("Segment [%s] API call failed: %s", time_range, e)
            return f"(描述生成失败: {e})"

    def _process_one_segment(
        self,
        video_path: str,
        client,
        api_model: str,
        idx: int,
        start: float,
        end: float,
    ) -> tuple[int, str, list[str]]:
        """单片段处理：抽帧 → 编码 → 调用 LLM 描述。返回 (idx, result_line, frame_paths_for_cleanup)。"""
        time_range = f"{_format_time(start)}-{_format_time(end)}"
        frame_paths: list[str] = []
        try:
            frame_paths = extract_frames(video_path, start, end)
            if not frame_paths:
                return (idx, f"slice{idx}: [{time_range}] (帧提取失败)", [])

            frames_b64: list[str] = []
            for fp in frame_paths:
                try:
                    frames_b64.append(encode_image_to_base64(fp))
                except Exception as e:
                    logger.warning("Failed to encode frame %s: %s", fp, e)

            if not frames_b64:
                return (idx, f"slice{idx}: [{time_range}] (帧编码失败)", frame_paths)

            description = self._describe_segment(
                client, api_model, frames_b64, start, end,
            )
            return (idx, f"slice{idx}: [{time_range}] {description}", frame_paths)
        except Exception as e:
            logger.warning("Segment %d [%s] failed: %s", idx, time_range, e)
            return (
                idx,
                f"slice{idx}: [{time_range}] (处理失败: {e})",
                frame_paths,
            )

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instance_dict.get(instance_id, {})
        video_path = state.get("video_path")
        api_base = state.get("api_base")
        api_key = state.get("api_key")
        api_model = state.get("api_model")

        if not video_path or not os.path.exists(video_path):
            return ToolResponse(text="Error: video_path missing or file not found."), 0.0, {}
        if not api_base:
            return ToolResponse(text="Error: api_base not configured for video_slice tool."), 0.0, {}
        if api_key is None:
            api_key = ""

        if not _check_ffmpeg_available():
            return ToolResponse(
                text="Error: ffmpeg is required for frame extraction but not found. "
                "Install ffmpeg (e.g. apt-get install ffmpeg)."
            ), 0.0, {}

        try:
            scenes = detect_scenes(
                video_path,
                threshold=self._scene_threshold,
                min_scene_len=self._min_scene_len,
            )
            logger.info("TransNetV2 detected %d scenes for %s", len(scenes), video_path)
        except Exception:
            logger.exception("TransNetV2 scene detection failed, falling back to whole-video mode")
            duration = get_video_duration(video_path)
            scenes = [(0.0, duration if duration > 0 else 1.0)]

        client = _get_openai_client(api_base, api_key)
        slice_results: list[str] = []
        all_tmp_files: list[str] = []
        try:
            tasks = [
                asyncio.to_thread(
                    self._process_one_segment,
                    video_path,
                    client,
                    api_model,
                    idx,
                    start,
                    end,
                )
                for idx, (start, end) in enumerate(scenes, 1)
            ]
            results = await asyncio.gather(*tasks)
            slice_results = [line for _, line, _ in sorted(results, key=lambda x: x[0])]
            for _, _, paths in results:
                all_tmp_files.extend(paths)
        finally:
            for f in all_tmp_files:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except OSError:
                        pass

        if not slice_results:
            return ToolResponse(text="Error: no scene segments could be processed."), 0.0, {}

        return ToolResponse(text="\n".join(slice_results)), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)

































