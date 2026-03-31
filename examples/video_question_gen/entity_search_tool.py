# examples/video_question_gen/entity_search_tool.py
# 实体搜索工具：调用已部署的 Qwen3-14B 根据关键词生成背景知识（不修改项目内任何文件）。
# 用法：在 config/tool_config.yaml 中增加本工具，class_name 指向 EntitySearchTool。
# 需在 config 中配置 api_base、api_model 指向 Qwen3-14B 的 OpenAI 兼容服务。

import logging
import os
import re
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# 默认与 evaluation_service 中 Qwen3-14B 服务一致，可在 tool_config.yaml 中覆盖
DEFAULT_QWEN14B_BASE_URL = ""
DEFAULT_QWEN14B_MODEL = ""
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RESULTS = 5
DEFAULT_SEARCH_MODE = "model"
DEFAULT_GOOGLE_SEARCH_URL = ""

ENTITY_SEARCH_SYSTEM = "你是一个知识检索助手。根据用户给出的关键词，用简洁、准确的语言介绍相关背景知识，便于理解视频或对话中的提及。只输出事实性内容，不要编造。"
ENTITY_SEARCH_USER_TEMPLATE = "请针对以下关键词简要介绍相关背景知识（如人物、作品、梗、专有名词等），用于辅助理解视频内容。用简洁的条目或短段落形式回答，总长度控制在 500 字以内。\n\n关键词：{keyword}. /no_think"


def search_with_qwen(
    keyword: str,
    api_base: str,
    api_model: str,
    api_key: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """调用已部署的 Qwen3-14B（OpenAI 兼容接口）根据关键词生成背景知识。"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        resp = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": ENTITY_SEARCH_SYSTEM},
                {"role": "user", "content": ENTITY_SEARCH_USER_TEMPLATE.format(keyword=keyword)},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content if content else ""
    except Exception as e:
        logger.warning("entity_search: Qwen3-14B search failed for %s: %s", keyword, e)
        return ""


def search_with_google(
    keyword: str,
    api_key: str,
    cse_id: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: int = DEFAULT_TIMEOUT,
    search_url: str = DEFAULT_GOOGLE_SEARCH_URL,
) -> list[dict]:
    """调用 Google Custom Search API，返回检索结果。"""
    if not api_key or not cse_id:
        logger.warning("entity_search: google_api_key or google_cse_id is missing")
        return []

    try:
        resp = requests.get(
            search_url,
            params={
                "key": api_key,
                "cx": cse_id,
                "q": keyword,
                "num": max(1, min(max_results, 10)),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", []) or []
        results = []
        for item in items[:max_results]:
            results.append(
                {
                    "title": item.get("title", "").strip(),
                    "link": item.get("link", "").strip(),
                    "snippet": item.get("snippet", "").strip(),
                }
            )
        return results
    except Exception as e:
        logger.warning("entity_search: Google search failed for %s: %s", keyword, e)
        return []


class EntitySearchTool(BaseTool):
    """
    实体搜索工具：
    1) mode=model：调用已部署模型返回关键词相关知识；
    2) mode=google：调用 Google Search API 返回 topN 检索结果。
    create_kwargs 可选；config 中可配置模型与 Google API 参数。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}
        self._api_base = config.get("api_base", DEFAULT_QWEN14B_BASE_URL)
        self._api_model = config.get("api_model", DEFAULT_QWEN14B_MODEL)
        self._api_key = config.get("api_key", "")
        self._max_tokens = config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self._timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self._search_mode = config.get("search_mode", DEFAULT_SEARCH_MODE)
        self._max_results = config.get("max_results", DEFAULT_MAX_RESULTS)
        self._google_api_key = config.get("google_api_key", "")
        self._google_cse_id = config.get("google_cse_id", "")
        self._google_search_url = config.get("google_search_url", DEFAULT_GOOGLE_SEARCH_URL)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        from uuid import uuid4

        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        self._instance_dict[instance_id] = {
            "api_base": create_kwargs.get("api_base", self._api_base),
            "api_model": create_kwargs.get("api_model", self._api_model),
            "api_key": create_kwargs.get("api_key", self._api_key),
            "max_tokens": create_kwargs.get("max_tokens", self._max_tokens),
            "timeout": create_kwargs.get("timeout", self._timeout),
            "search_mode": create_kwargs.get("search_mode", self._search_mode),
            "max_results": create_kwargs.get("max_results", self._max_results),
            "google_api_key": create_kwargs.get("google_api_key", self._google_api_key),
            "google_cse_id": create_kwargs.get("google_cse_id", self._google_cse_id),
            "google_search_url": create_kwargs.get("google_search_url", self._google_search_url),
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
        api_base = state.get("api_base", self._api_base)
        api_model = state.get("api_model", self._api_model)
        api_key = state.get("api_key", self._api_key)
        max_tokens = state.get("max_tokens", self._max_tokens)
        timeout = state.get("timeout", self._timeout)
        search_mode = str(state.get("search_mode", self._search_mode)).strip().lower()
        max_results = state.get("max_results", self._max_results)
        google_api_key = state.get("google_api_key", self._google_api_key)
        google_cse_id = state.get("google_cse_id", self._google_cse_id)
        google_search_url = state.get("google_search_url", self._google_search_url)

        # 支持 keywords 为 list[str] 或 str（逗号/顿号分隔）
        keywords_raw = parameters.get("keywords")
        if isinstance(keywords_raw, list):
            keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
        elif isinstance(keywords_raw, str):
            keywords_str = keywords_raw.strip().strip("[]【】")
            keywords = [
                kw.strip().strip("'\"")
                for kw in re.split(r"[,，、]", keywords_str)
                if kw.strip()
            ]
        else:
            keywords = []

        if not keywords:
            return (
                ToolResponse(text="【实体搜索】未提供有效关键词，请在 arguments 中传入 keywords（字符串或字符串列表）。"),
                0.0,
                {},
            )

        mode_from_params = parameters.get("mode")
        if isinstance(mode_from_params, str) and mode_from_params.strip():
            search_mode = mode_from_params.strip().lower()
        if search_mode not in {"model", "google"}:
            return (
                ToolResponse(text="【实体搜索】mode 仅支持 'model' 或 'google'。"),
                0.0,
                {},
            )

        all_results = []
        for keyword in keywords:
            if not keyword:
                continue
            if search_mode == "model":
                content = search_with_qwen(
                    keyword=keyword,
                    api_base=api_base,
                    api_model=api_model,
                    api_key=api_key,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                if not content:
                    all_results.append(f"### {keyword}\n未获取到相关背景知识。")
                    continue
                all_results.append(f"### {keyword}\n{content}")
            else:
                results = search_with_google(
                    keyword=keyword,
                    api_key=google_api_key,
                    cse_id=google_cse_id,
                    max_results=max_results,
                    timeout=timeout,
                    search_url=google_search_url,
                )
                if not results:
                    all_results.append(f"### {keyword}\n未获取到 Google 检索结果。")
                    continue

                lines = [f"### {keyword}"]
                for i, item in enumerate(results, 1):
                    title = item.get("title") or "无标题"
                    link = item.get("link") or ""
                    snippet = item.get("snippet") or ""
                    lines.append(f"  {i}. {title}")
                    if snippet:
                        lines.append(f"     {snippet[:300]}")
                    if link:
                        lines.append(f"     {link}")
                all_results.append("\n".join(lines))

        title = "【实体搜索结果 - 模型知识】" if search_mode == "model" else "【实体搜索结果 - Google Top5】"
        text = f"{title}\n\n" + "\n\n".join(all_results)
        return ToolResponse(text=text), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)

