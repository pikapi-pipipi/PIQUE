"""
问题质量评估奖励函数（改进版 + 课程学习支持）：
- 格式：本地规则检查
- 外延性：调用已部署的 Qwen3-VL-32B-Thinking，针对视频内容评估
- 综合质量：调用已部署的 Qwen3-14B 服务（替代原独立语义，增加人类吸引力判断）
- 安全性：调用已部署的 Qwen3Guard-Gen-4B 服务

改进点：
1. 加权连续评分（替代二元 0/1），提升 RL 训练信号质量
2. 分数兜底 max(0, score)，防止负分
3. 三个远程评估并行化（ThreadPoolExecutor），降低延迟、减少 GPU 空转
4. 统一异常处理策略：所有维度异常均视为不合格（fail-safe）
5. HTTP 请求增加重试机制（指数退避）
6. CONFIG 可通过 config_override 参数注入，方便测试
7. acc 与 score 解耦，acc 仅反映质量维度通过率
8. 修复 max_question_len 异常值、禁用词误伤等格式检查问题
9. 独立语义维度升级为综合质量评估，增加人类吸引力判断，输出改为"有吸引力/无吸引力"
10. 课程学习支持：config_override 或 extra_info 传入维度/权重覆盖

与 verl 奖励接口一致：compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
返回 dict（含 "score"），其余键进入 reward_extra_info。
"""

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

# ============== 默认配置 ==============

DEFAULT_CONFIG = {
    "vl_extensibility_base_url": "",
    "vl_extensibility_model": "",
    "qwen14b_base_url": "",
    "qwen14b_model": "",
    "safety_base_url": "",
    "safety_model": "",
    "api_key": "",
    "max_video_base64_bytes": 10 * 1024 * 1024,
    "max_video_size_before_compress_bytes": 7 * 1024 * 1024,
    "min_question_len": 10,
    "max_question_len": 30,
    "forbidden_words": ["你", "我"],
    "dimensions": ["format", "extensibility", "safety", "quality", "timing"],
    "http_timeout_seconds": 90,
    "http_timeout_vl_seconds": 180,
    "http_max_retries": 2,
    "http_retry_base_delay": 1.0,
    "dimension_weights": {
        "format": 0.2,
        "extensibility": 0.3,
        "safety": 0.2,
        "quality": 0.3,
        "timing": 0.2,
    },
    "max_response_tokens": 16384,
    "tool_violation_penalty": 0.1,
}


def _get_config(config_override: Optional[Dict] = None) -> Dict:
    """合并默认配置与用户覆盖。"""
    cfg = DEFAULT_CONFIG.copy()
    cfg["dimension_weights"] = DEFAULT_CONFIG["dimension_weights"].copy()
    if config_override:
        for k, v in config_override.items():
            if k == "dimension_weights" and isinstance(v, dict):
                cfg["dimension_weights"].update(v)
            else:
                cfg[k] = v
    return cfg


# ============== 结果类型 ==============

class FormatResult:
    def __init__(self, is_valid: bool, quality: str, fail_reasons: List[str], chinese_char_count: int):
        self.is_valid = is_valid
        self.quality = quality
        self.fail_reasons = fail_reasons
        self.chinese_char_count = chinese_char_count


class ExtensibilityResult:
    def __init__(self, is_valid: bool, label: str, raw_answer: Optional[str] = None,
                 skipped: bool = False, skip_reason: Optional[str] = None):
        self.is_valid = is_valid
        self.label = label
        self.raw_answer = raw_answer
        self.skipped = skipped
        self.skip_reason = skip_reason


class SafetyResult:
    def __init__(self, is_safe: bool, label: str, raw_answer: Optional[str] = None):
        self.is_safe = is_safe
        self.label = label
        self.raw_answer = raw_answer


class QualityResult:
    """综合质量评估结果（替代原 IndependenceResult）。label: '有吸引力' / '无吸引力'"""
    def __init__(self, is_attractive: bool, label: str, raw_answer: Optional[str] = None,
                 skipped: bool = False, skip_reason: Optional[str] = None):
        self.is_attractive = is_attractive
        self.label = label
        self.raw_answer = raw_answer
        self.skipped = skipped
        self.skip_reason = skip_reason


class TimingResult:
    """时机相关性评估结果（问题标注时间点是否与视频内容对应）。label: '时机相关' / '时机不相关'"""
    def __init__(self, is_relevant: bool, label: str, raw_answer: Optional[str] = None,
                 skipped: bool = False, skip_reason: Optional[str] = None):
        self.is_relevant = is_relevant
        self.label = label
        self.raw_answer = raw_answer
        self.skipped = skipped
        self.skip_reason = skip_reason


# ============== 工具函数 ==============

def _extract_question_from_response(solution_str: str) -> str:
    if not solution_str or not isinstance(solution_str, str):
        return ""
    m = re.search(r"<question>\s*([\s\S]*?)\s*</question>", solution_str, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return solution_str.strip()


def _parse_question_and_timestamp(solution_str: str) -> Tuple[Optional[float], str]:
    """从 <question>[0.0]问题文本</question> 中解析出 (秒级时间戳, 问题正文)。无时间戳时返回 (None, 全文)。"""
    raw = _extract_question_from_response(solution_str)
    if not raw:
        return None, ""
    m = re.match(r"^\[(\d+(?:\.\d+)?)\]\s*(.*)$", raw.strip(), re.DOTALL)
    if m:
        try:
            ts = float(m.group(1))
            text = (m.group(2) or "").strip()
            return (ts, text if text else raw)
        except ValueError:
            pass
    return None, raw


def _get_video_path_from_extra_info(extra_info: Optional[Dict]) -> Optional[str]:
    if not extra_info:
        return None
    path = extra_info.get("video_path")
    if path:
        return path
    tk = extra_info.get("tools_kwargs") or {}
    vs = (tk.get("video_slice") or {}).get("create_kwargs") or {}
    return vs.get("video_path")


def count_chinese_chars(text: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', text))


def _compress_video(video_path: str, config: Dict) -> str:
    if not os.path.exists(video_path):
        return video_path
    size = os.path.getsize(video_path)
    if size <= config["max_video_size_before_compress_bytes"]:
        return video_path
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=False)
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
    except FileNotFoundError:
        return video_path
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True, timeout=10,
        )
        duration = float((r.stdout or "").strip() or "1.0")
    except Exception:
        return video_path
    target_mb = 5.0
    target_bitrate = int((target_mb * 8 * 1024) / duration)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-b:v", f"{target_bitrate}k",
             "-vf", "scale='min(720,iw)':'-2'", "-c:v", "libx264", "-preset", "fast",
             "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", tmp_path],
            capture_output=True, text=True, check=True, timeout=120,
        )
        return tmp_path
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return video_path


def _video_to_base64(video_path: Optional[str], video_base64: Optional[str],
                     config: Dict) -> Tuple[Optional[str], Optional[str]]:
    if video_base64:
        return video_base64.strip(), None
    if not video_path or not os.path.exists(video_path):
        return None, None
    compressed = _compress_video(video_path, config)
    cleanup = compressed if compressed != video_path else None
    with open(compressed, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    if len(b64.encode("utf-8")) > config["max_video_base64_bytes"]:
        return None, cleanup
    return b64, cleanup


# ============== HTTP 客户端（带重试） ==============

def _openai_client(base_url: str, config: Dict, timeout: Optional[float] = None):
    from openai import OpenAI
    t = timeout if timeout is not None else config.get("http_timeout_seconds", 90)
    return OpenAI(api_key=config.get("api_key", "dummy"), base_url=base_url, timeout=t)


def _call_with_retry(fn, config: Dict):
    max_retries = config.get("http_max_retries", 2)
    base_delay = config.get("http_retry_base_delay", 1.0)
    last_exc = None
    for attempt in range(1 + max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


# ============== 工具调用检查 ==============

VALID_TOOLS = {"video_slice", "entity_search", "asr", "ocr"}
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _extract_tool_calls(solution_str: str) -> List[str]:
    if not solution_str:
        return []
    matches = _TOOL_CALL_RE.findall(solution_str)
    tool_names: List[str] = []
    for match in matches:
        try:
            call = json.loads(match.strip())
            name = call.get("name", "")
            if name in VALID_TOOLS:
                tool_names.append(name)
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return tool_names


def check_tool_duplicate(tool_names: List[str]) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    counts = Counter(tool_names)
    for tool, count in counts.items():
        if count > 1:
            violations.append(f"工具 {tool} 被调用了 {count} 次（最多1次）")
    return len(violations) == 0, violations


def check_tool_order(tool_names: List[str]) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    if not tool_names:
        return True, violations

    if "video_slice" in tool_names and "entity_search" in tool_names:
        vs_idx = tool_names.index("video_slice")
        es_idx = tool_names.index("entity_search")
        if vs_idx > es_idx:
            violations.append("video_slice 必须在 entity_search 之前调用")

    has_primary = ("video_slice" in tool_names) or ("entity_search" in tool_names)
    for aux_tool in ("asr", "ocr"):
        if aux_tool not in tool_names:
            continue
        if not has_primary:
            violations.append(f"{aux_tool} 不能在未调用 video_slice 或 entity_search 的情况下使用")
        else:
            aux_idx = tool_names.index(aux_tool)
            for primary in ("video_slice", "entity_search"):
                if primary in tool_names:
                    primary_idx = tool_names.index(primary)
                    if aux_idx < primary_idx:
                        violations.append(f"{aux_tool} 必须在 {primary} 之后调用")

    return len(violations) == 0, violations


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text))
    other_chars = len(text) - chinese_chars
    return chinese_chars + max(1, other_chars // 3) if other_chars > 0 else chinese_chars


# ============== 格式检查 ==============

_FORBIDDEN_WORD_WHITELIST = {"你好", "我们的", "自我", "忘我", "无我"}


def check_format_rules(question: str, config: Dict) -> FormatResult:
    fail_reasons = []
    chinese_count = count_chinese_chars(question)
    min_len = config["min_question_len"]
    max_len = config["max_question_len"]

    if chinese_count < min_len or chinese_count > max_len:
        fail_reasons.append(f"字数不符({chinese_count}字，要求{min_len}-{max_len}字)")

    qm = question.count("？") + question.count("?")
    if qm != 1:
        fail_reasons.append(f"问号数量不符({qm}个，要求1个)")

    cleaned = question
    for safe_word in _FORBIDDEN_WORD_WHITELIST:
        cleaned = cleaned.replace(safe_word, "")
    found = [w for w in config.get("forbidden_words", []) if w in cleaned]
    if found:
        fail_reasons.append(f"含指代词({','.join(found)})")

    is_valid = len(fail_reasons) == 0
    return FormatResult(
        is_valid=is_valid,
        quality="合格" if is_valid else "不合格",
        fail_reasons=fail_reasons,
        chinese_char_count=chinese_count,
    )


# ============== 外延性（VL 视频） ==============

EXTENSIBILITY_SYSTEM = "你是专业的问题质量评估专家"
EXTENSIBILITY_USER_TEMPLATE = """你是问题外延性评估专家。请根据视频内容判断：以下问题的答案是否能直接从视频中找到或推断出来。

视频已随本消息提供。

问题：
{question}

评估标准：
- 如果问题的答案可以直接从视频内容中找到或近似表述，判定为"不合格"
- 如果问题的答案需要视频以外的知识或逻辑推断才能回答，判定为"合格"

请先基于视频内容分析，然后直接输出"合格"或"不合格"。/no_think"""


def _parse_extensibility_answer(answer: str) -> Tuple[bool, str]:
    if "<think>" in answer and "</think>" in answer:
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    if "合格" in answer and "不合格" not in answer:
        return True, "合格"
    return False, "不合格"


def _extensibility_evaluate_single(
    question: str,
    video_b64: Optional[str],
    config: Dict,
) -> ExtensibilityResult:
    if not video_b64:
        return ExtensibilityResult(
            is_valid=False,
            label="不合格",
            skipped=True,
            skip_reason="未提供视频(video_path 或 video_base64)，按不合格处理",
        )
    content = [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        {"type": "text", "text": EXTENSIBILITY_USER_TEMPLATE.format(question=question)},
    ]

    def _do_call():
        vl_timeout = config.get("http_timeout_vl_seconds", 180)
        client = _openai_client(config["vl_extensibility_base_url"], config, timeout=vl_timeout)
        resp = client.chat.completions.create(
            model=config["vl_extensibility_model"],
            messages=[
                {"role": "system", "content": EXTENSIBILITY_SYSTEM},
                {"role": "user", "content": content},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        answer = _call_with_retry(_do_call, config)
    except Exception as e:
        return ExtensibilityResult(
            is_valid=False,
            label="不合格",
            skipped=True,
            skip_reason=f"服务调用失败（已重试），按不合格处理: {e}",
        )
    is_valid, label = _parse_extensibility_answer(answer)
    return ExtensibilityResult(is_valid=is_valid, label=label, raw_answer=answer)


# ============== 时机相关性（VL：问题时间点与视频内容是否对应） ==============

TIMING_SYSTEM = "你是视频与问题时机匹配评估专家。"
TIMING_USER_TEMPLATE = """请根据视频内容判断：在视频的第 {timestamp} 秒附近，画面或内容是否与以下问题相关，即该问题是否适合在此时刻提出。

问题：{question}

请先基于视频该时间点附近的内容分析，然后直接输出"时机相关"或"时机不相关"。/no_think"""


def _parse_timing_answer(answer: str) -> Tuple[bool, str]:
    if "<think>" in answer and "</think>" in answer:
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    if "时机相关" in answer and "时机不相关" not in answer:
        return True, "时机相关"
    return False, "时机不相关"


def _timing_evaluate_single(
    question_text: str,
    timestamp_sec: float,
    video_b64: Optional[str],
    config: Dict,
) -> TimingResult:
    if not video_b64:
        return TimingResult(
            is_relevant=False,
            label="时机不相关",
            skipped=True,
            skip_reason="未提供视频，按时机不相关处理",
        )
    content = [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        {"type": "text", "text": TIMING_USER_TEMPLATE.format(timestamp=timestamp_sec, question=question_text)},
    ]

    def _do_call():
        vl_timeout = config.get("http_timeout_vl_seconds", 180)
        client = _openai_client(config["vl_extensibility_base_url"], config, timeout=vl_timeout)
        resp = client.chat.completions.create(
            model=config["vl_extensibility_model"],
            messages=[
                {"role": "system", "content": TIMING_SYSTEM},
                {"role": "user", "content": content},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        answer = _call_with_retry(_do_call, config)
    except Exception as e:
        return TimingResult(
            is_relevant=False,
            label="时机不相关",
            skipped=True,
            skip_reason=f"服务调用失败（已重试）: {e}",
        )
    is_relevant, label = _parse_timing_answer(answer)
    return TimingResult(is_relevant=is_relevant, label=label, raw_answer=answer)


# ============== 统一返回 key 集合 ==============

_ALL_RESULT_KEYS = (
    "score", "quality_score", "quality", "fail_reasons", "penalty_reasons",
    "question", "acc", "dim_passed", "tool_calls", "estimated_tokens",
    "format_result", "extensibility_result", "safety_result",
    "quality_eval_result", "timing_result", "question_timestamp",
)


def _build_result_dict(**kwargs: Any) -> Dict[str, Any]:
    """Build a result dict that always contains every key in _ALL_RESULT_KEYS.

    Missing keys default to None so that reward_extra_info is consistent
    across all samples regardless of curriculum stage or early-exit path.
    """
    return {k: kwargs.get(k) for k in _ALL_RESULT_KEYS}


# ============== 安全性（Guard 服务） ==============

def _parse_safety_answer(content: str) -> Tuple[bool, str]:
    m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", content)
    label = m.group(1) if m else "Unknown"
    return label == "Safe", label


def _safety_evaluate_single(question: str, config: Dict) -> SafetyResult:
    def _do_call():
        client = _openai_client(config["safety_base_url"], config)
        resp = client.chat.completions.create(
            model=config["safety_model"],
            messages=[{"role": "user", "content": question}],
            max_tokens=128,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        answer = _call_with_retry(_do_call, config)
    except Exception as e:
        return SafetyResult(is_safe=False, label="Error", raw_answer=str(e))
    is_safe, label = _parse_safety_answer(answer)
    return SafetyResult(is_safe=is_safe, label=label, raw_answer=answer)


# ============== 综合质量评估（替代原独立语义，增加人类吸引力判断） ==============

QUALITY_SYSTEM = "你是个专业的文本质量评估专家"
QUALITY_USER_TEMPLATE = """你是极其严格的问答数据集质检专家。你的任务是从多个维度综合评估给定问题的质量。

待评估问题：
{question}

评估标准（必须严格执行，所有条件均需满足方可判定为"有吸引力"）：

一、独立语义（自包含性）
1. 零上下文独立性：问题必须是完全自包含的。假设读者没有任何前文背景、没看过相关文章、也不知道之前的对话。如果读者在阅读问题时会产生"具体指哪一个？"、"这是指文章里的什么？"等疑问，则该条不通过。
2. 指代明确性：严查依赖上下文的指代。如果问题中包含指向性不明的代词或名词短语（例如指代不清的"这个/这些/该/此 X"、"文中提到的"、"作者认为"等），且没有后续定语从句修饰使其具体化，则该条不通过。
3. 实体完整性：问题所讨论的核心对象（主语/宾语）必须是通用的、具体的概念，而非临时的、模糊的指代。

二、人类吸引力
4. 好奇心激发：站在一个普通人的角度，这个问题是否能引起好奇心或探索欲？好的问题应该让人看了之后想知道答案，而非觉得无聊、显而易见或过于晦涩。
5. 话题价值：问题是否涉及有趣的、有价值的、或能引发思考的话题？纯粹机械式的、过于琐碎的、或毫无信息增量的问题应判定为无吸引力。

判定逻辑：
- 先检查独立语义（条件1-3）：模拟一个"完全不知道背景信息的路人"视角，如果不看上下文就不知道在问什么，或者问题脱离了某段特定文本就无法成立，则直接判定为"无吸引力"。
- 再检查人类吸引力（条件4-5）：如果独立语义通过，继续判断该问题是否能勾起普通人的兴趣。如果问题过于平淡、无聊、琐碎，或答案显而易见到毫无悬念，则判定为"无吸引力"。
- 只有当所有条件均满足时，才判定为"有吸引力"。

输出要求：
仅输出"有吸引力"或"无吸引力"，不要输出任何解释或其他字符。"""


def _parse_quality_answer(answer: str) -> Tuple[bool, str]:
    if "<think>" in answer and "</think>" in answer:
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    if "有吸引力" in answer and "无吸引力" not in answer:
        return True, "有吸引力"
    return False, "无吸引力"


def _quality_evaluate_single(question: str, config: Dict) -> QualityResult:
    def _do_call():
        client = _openai_client(config["qwen14b_base_url"], config)
        resp = client.chat.completions.create(
            model=config["qwen14b_model"],
            messages=[
                {"role": "system", "content": QUALITY_SYSTEM},
                {"role": "user", "content": QUALITY_USER_TEMPLATE.format(question=question)},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        answer = _call_with_retry(_do_call, config)
    except Exception as e:
        return QualityResult(
            is_attractive=False,
            label="无吸引力",
            skipped=True,
            skip_reason=f"服务调用失败（已重试），按无吸引力处理: {e}",
        )
    is_attractive, label = _parse_quality_answer(answer)
    return QualityResult(is_attractive=is_attractive, label=label, raw_answer=answer)


# ============== 奖励函数入口 ==============

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
    config_override: Optional[Dict] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """与 verl 奖励函数接口一致：根据生成的问题做多维度评估。

    课程学习集成：
    - config_override 可传入 {"dimensions": [...], "dimension_weights": {...}} 覆盖默认维度与权重
    - extra_info 中的 curriculum_stage / curriculum_dimensions 也会被转为 config_override
    - 优先级：config_override > extra_info.curriculum_dimensions > extra_info.curriculum_stage > DEFAULT_CONFIG
    """
    extra_info = extra_info or {}

    # 课程学习：从 extra_info 提取阶段信息，合并到 config_override
    curriculum_override = _extract_curriculum_override(extra_info)
    if curriculum_override:
        if config_override is None:
            config_override = curriculum_override
        else:
            merged = curriculum_override.copy()
            merged.update(config_override)
            config_override = merged

    config = _get_config(config_override)
    dimensions = config.get("dimensions", ["format", "extensibility", "safety", "quality"])
    weights = config.get("dimension_weights", {})

    # ---------- 提取问题（支持 <question>[0.0]问题</question> 秒级时间戳） ----------
    timestamp_sec, question = _parse_question_and_timestamp(solution_str)
    if not question:
        return _build_result_dict(
            score=0.0,
            quality_score=0.0,
            quality="不合格",
            fail_reasons=["未解析到问题文本"],
            penalty_reasons=[],
            question="",
            acc=0.0,
            dim_passed={},
            tool_calls=[],
            estimated_tokens=0,
            question_timestamp=None,
        )

    # ---------- 准备视频 ----------
    video_path = _get_video_path_from_extra_info(extra_info)
    video_base64_input = extra_info.get("video_base64")
    video_b64: Optional[str] = None
    cleanup_path: Optional[str] = None

    if "extensibility" in dimensions or "timing" in dimensions:
        video_b64, cleanup_path = _video_to_base64(video_path, video_base64_input, config)

    # ---------- 各维度评估 ----------
    fail_reasons: List[str] = []
    dim_passed: Dict[str, bool] = {}
    format_result: Optional[FormatResult] = None
    ext_result: Optional[ExtensibilityResult] = None
    safety_result: Optional[SafetyResult] = None
    quality_result: Optional[QualityResult] = None
    timing_result: Optional[TimingResult] = None

    if "format" in dimensions:
        format_result = check_format_rules(question, config)
        dim_passed["format"] = format_result.is_valid
        if not format_result.is_valid:
            fail_reasons.extend(format_result.fail_reasons)

    # 远程评估并行化
    remote_dims = [d for d in ["extensibility", "safety", "quality", "timing"] if d in dimensions]

    if remote_dims:
        futures = {}
        with ThreadPoolExecutor(max_workers=len(remote_dims)) as pool:
            if "extensibility" in remote_dims:
                futures["extensibility"] = pool.submit(
                    _extensibility_evaluate_single, question, video_b64, config
                )
            if "safety" in remote_dims:
                futures["safety"] = pool.submit(
                    _safety_evaluate_single, question, config
                )
            if "quality" in remote_dims:
                futures["quality"] = pool.submit(
                    _quality_evaluate_single, question, config
                )
            if "timing" in remote_dims and timestamp_sec is not None and video_b64:
                futures["timing"] = pool.submit(
                    _timing_evaluate_single, question, timestamp_sec, video_b64, config
                )

        # timing 在维度中但未提交（无时间戳或无视频）时直接记为不通过
        if "timing" in dimensions and "timing" not in futures:
            dim_passed["timing"] = False
            fail_reasons.append("时机不相关(缺少时间戳或视频)")

        for dim_name, future in futures.items():
                try:
                    result = future.result()
                except Exception as e:
                    if dim_name == "extensibility":
                        result = ExtensibilityResult(
                            is_valid=False, label="不合格", skipped=True,
                            skip_reason=f"线程异常: {e}")
                    elif dim_name == "safety":
                        result = SafetyResult(is_safe=False, label="Error", raw_answer=str(e))
                    elif dim_name == "timing":
                        result = TimingResult(
                            is_relevant=False, label="时机不相关", skipped=True,
                            skip_reason=f"线程异常: {e}")
                    else:
                        result = QualityResult(
                            is_attractive=False, label="无吸引力", skipped=True,
                            skip_reason=f"线程异常: {e}")

                if dim_name == "extensibility":
                    ext_result = result
                    dim_passed["extensibility"] = result.is_valid
                    if not result.is_valid:
                        reason = "外延性不合格"
                        if result.skipped:
                            reason += f"({result.skip_reason})"
                        fail_reasons.append(reason)

                elif dim_name == "safety":
                    safety_result = result
                    dim_passed["safety"] = result.is_safe
                    if not result.is_safe:
                        fail_reasons.append(f"安全性:{result.label}")

                elif dim_name == "quality":
                    quality_result = result
                    dim_passed["quality"] = result.is_attractive
                    if not result.is_attractive:
                        reason = f"综合质量:{result.label}"
                        if result.skipped:
                            reason += f"({result.skip_reason})"
                        fail_reasons.append(reason)

                elif dim_name == "timing":
                    timing_result = result
                    dim_passed["timing"] = result.is_relevant
                    if not result.is_relevant:
                        reason = "时机不相关"
                        if result.skipped:
                            reason += f"({result.skip_reason})"
                        fail_reasons.append(reason)

    if cleanup_path and os.path.exists(cleanup_path):
        try:
            os.unlink(cleanup_path)
        except OSError:
            pass

    # ---------- 加权连续评分 ----------
    total_weight = 0.0
    weighted_score = 0.0
    for dim in dimensions:
        w = weights.get(dim, 0.0)
        total_weight += w
        if dim_passed.get(dim, False):
            weighted_score += w

    if total_weight > 0:
        quality_score = weighted_score / total_weight
    else:
        quality_score = 0.0

    num_evaluated = len(dim_passed)
    num_passed = sum(1 for v in dim_passed.values() if v)
    acc = num_passed / num_evaluated if num_evaluated > 0 else 0.0

    # ---------- 工具调用 & token 约束惩罚 ----------
    penalty = config.get("tool_violation_penalty", 0.1)
    penalty_reasons: List[str] = []
    total_penalty = 0.0

    tool_names = _extract_tool_calls(solution_str)

    dup_ok, dup_violations = check_tool_duplicate(tool_names)
    if not dup_ok:
        total_penalty += penalty
        penalty_reasons.extend(dup_violations)

    order_ok, order_violations = check_tool_order(tool_names)
    if not order_ok:
        total_penalty += penalty
        penalty_reasons.extend(order_violations)

    max_tokens = config.get("max_response_tokens", 16384)
    estimated_tokens = _estimate_token_count(solution_str)
    if estimated_tokens > max_tokens:
        total_penalty += penalty
        penalty_reasons.append(
            f"回答 token 数（约{estimated_tokens}）超过上限（{max_tokens}）"
        )

    score = max(0.0, min(1.0, quality_score - total_penalty))

    overall_quality = "合格" if len(fail_reasons) == 0 and len(penalty_reasons) == 0 else "不合格"

    # ---------- 组装返回 ----------
    return _build_result_dict(
        score=round(score, 4),
        quality_score=round(quality_score, 4),
        quality=overall_quality,
        fail_reasons=fail_reasons,
        penalty_reasons=penalty_reasons,
        question=question,
        acc=round(acc, 4),
        dim_passed=dim_passed,
        tool_calls=tool_names,
        estimated_tokens=estimated_tokens,
        format_result={
            "is_valid": format_result.is_valid,
            "quality": format_result.quality,
            "fail_reasons": format_result.fail_reasons,
            "chinese_char_count": format_result.chinese_char_count,
        } if format_result is not None else None,
        extensibility_result={
            "is_valid": ext_result.is_valid,
            "label": ext_result.label,
            "skipped": ext_result.skipped,
            "skip_reason": ext_result.skip_reason,
        } if ext_result is not None else None,
        safety_result={
            "is_safe": safety_result.is_safe,
            "label": safety_result.label,
        } if safety_result is not None else None,
        quality_eval_result={
            "is_attractive": quality_result.is_attractive,
            "label": quality_result.label,
            "skipped": quality_result.skipped,
            "skip_reason": quality_result.skip_reason,
        } if quality_result is not None else None,
        timing_result={
            "is_relevant": timing_result.is_relevant,
            "label": timing_result.label,
            "skipped": timing_result.skipped,
            "skip_reason": timing_result.skip_reason,
        } if timing_result is not None else None,
        question_timestamp=timestamp_sec,
    )


# ============== 课程学习辅助 ==============

_CURRICULUM_STAGES = [
    {"dimensions": ["format", "safety", "timing"],
     "dimension_weights": {"format": 0.33, "safety": 0.34, "timing": 0.33}},
    {"dimensions": ["format", "extensibility", "safety", "timing"],
     "dimension_weights": {"format": 0.15, "extensibility": 0.35, "safety": 0.35, "timing": 0.15}},
    {"dimensions": ["format", "extensibility", "safety", "quality", "timing"],
     "dimension_weights": {"format": 0.15, "extensibility": 0.25, "safety": 0.15, "quality": 0.25, "timing": 0.2}},
]


def _extract_curriculum_override(extra_info: Dict) -> Optional[Dict]:
    """从 extra_info 中提取课程学习阶段信息并转为 config_override 字典。

    优先级：curriculum_config_override > curriculum_dimensions > curriculum_stage
    """
    if extra_info.get("curriculum_config_override") is not None:
        return dict(extra_info["curriculum_config_override"])

    if extra_info.get("curriculum_dimensions") is not None:
        dims = list(extra_info["curriculum_dimensions"])
        weights = extra_info.get("curriculum_dimension_weights")
        override: Dict[str, Any] = {"dimensions": dims}
        if weights is not None:
            override["dimension_weights"] = dict(weights)
        return override

    if extra_info.get("curriculum_stage") is not None:
        try:
            stage = min(max(0, int(extra_info["curriculum_stage"])), len(_CURRICULUM_STAGES) - 1)
            return dict(_CURRICULUM_STAGES[stage])
        except (TypeError, ValueError):
            pass

    return None

