import re
import logging
import torch
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

JUDGE_CONFIG = {
    "base_url": "",
    "model": "",
    "api_key": "",
    "timeout": 180,
}

JUDGE_SYSTEM_PROMPT = """你是一个工具调用合理性评估专家。给定一个视频问题生成Agent的完整trajectory，请判断其中每一步工具调用是否合理、是否有助于最终的问题生成。

对每个工具调用步骤，输出"合理"或"不合理"。"""

JUDGE_USER_TEMPLATE = """以下是一个Agent生成视频问题的完整trajectory：

{trajectory}

该trajectory中共有{num_steps}步工具调用。请逐步判断每一步工具调用是否合理，是否有助于最终的问题生成。

请严格按照以下JSON格式输出（不要输出其他内容）：
{{"steps": [{step_template}]}}

其中每个step的judgment为"合理"或"不合理"。"""

'''
def _find_think_toolcall_spans(text: str, tokenizer) -> list[tuple[int, int]]:
    """找到文本中每一步 <think>...</think><tool_call>...</tool_call> 对应的 token 范围。
    
    返回: [(start_token_idx, end_token_idx), ...] 每个元素是一个步骤的 token 范围（闭区间）
    """
    pattern = re.compile(
        r'(<think>.*?</think>\s*<tool_call>.*?</tool_call>)',
        re.DOTALL
    )
    spans = []
    for match in pattern.finditer(text):
        char_start, char_end = match.start(), match.end()
        prefix = text[:char_start]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        span_text = match.group(0)
        span_tokens = tokenizer.encode(span_text, add_special_tokens=False)
        token_start = len(prefix_tokens)
        token_end = token_start + len(span_tokens) - 1
        spans.append((token_start, token_end))
    return spans
'''

def _find_think_toolcall_spans(text: str, tokenizer) -> list[tuple[int, int]]:
    """找到文本中每一步工具调用对应的 thinking + tool_call token 范围。

    每一步的 span 从 <think> 开标签（若存在）或上一步 tool response 结尾 / 文本开头
    延伸到 </tool_call> 结束。
    """
    import re as _re

    tc_pattern = _re.compile(r'</think>\s*<tool_call>.*?</tool_call>', _re.DOTALL)
    think_open_positions = [m.start() for m in _re.finditer(r'<think>', text)]

    spans = []
    for match in tc_pattern.finditer(text):
        tc_end = match.end()           # </tool_call> 的末尾
        think_close_pos = match.start()  # </think> 的开头

        # 往前找配对的 <think>：取 think_close_pos 之前最近的 <think>
        preceding_opens = [p for p in think_open_positions if p < think_close_pos]
        if preceding_opens:
            span_char_start = preceding_opens[-1]  # 最近的 <think>
        else:
            # 没有 <think>（第一轮，开标签在 prompt 里）
            # 从文本开头或上一个 </tool_call> 之后开始
            span_char_start = 0
            for prev_span in spans:
                # 用之前记录的 char_end 来找上一步结尾
                pass  # 简化处理：直接从 0 开始

        span_char_end = tc_end

        # 转换为 token 范围
        prefix = text[:span_char_start]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        span_text = text[span_char_start:span_char_end]
        span_tokens = tokenizer.encode(span_text, add_special_tokens=False)
        token_start = len(prefix_tokens)
        token_end = token_start + len(span_tokens) - 1

        spans.append((token_start, token_end))
        print(f"[DEBUG] Step {len(spans)}: char[{span_char_start}:{span_char_end}], "
              f"token[{token_start}:{token_end}], {len(span_tokens)} tokens")

    print(f"[DEBUG] Total tool-call steps found: {len(spans)}")
    return spans

def _call_judge(trajectories: list[dict], config: dict) -> list[list[bool]]:
    """批量调用大模型判断每条trajectory中每步工具调用是否合理。
    
    Args:
        trajectories: [{"text": str, "num_steps": int}, ...]
        config: judge model config
    
    Returns:
        list[list[bool]]: 每条trajectory中每步是否合理(True=合理, False=不合理)
    """
    client = OpenAI(
        api_key=config.get("api_key", "dummy"),
        base_url=config["base_url"],
        timeout=config.get("timeout", 180),
    )
    results = []
    for traj in trajectories:
        if traj["num_steps"] == 0:
            results.append([])
            continue
        step_template = ', '.join(
            [f'{{"step": {i+1}, "judgment": "合理/不合理"}}' for i in range(traj["num_steps"])]
        )
        '''
        user_msg = JUDGE_USER_TEMPLATE.format(
            trajectory=traj["text"],
            num_steps=traj["num_steps"],
            step_template=step_template,
        )
        '''
        # 去掉 tool_response，只保留 <think> 和 <tool_call> 部分
        traj_text = traj["text"]
        traj_text = re.sub(
            r'(</tool_call>).*?(<think>)',
            r'\1\n...[tool_response omitted]...\n\2',
            traj_text,
            flags=re.DOTALL,
        )
        traj_text = re.sub(
            r'(</tool_call>)(?:(?!<think>).)*$',
            r'\1\n...[tool_response omitted]...',
            traj_text,
            flags=re.DOTALL,
        )
        print(f"[DEBUG] Filtered traj: {len(traj['text'])} -> {len(traj_text)} chars")

        user_msg = JUDGE_USER_TEMPLATE.format(
            trajectory=traj_text,
            num_steps=traj["num_steps"],
            step_template=step_template,
        )

        try:
            resp = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=8192,
                temperature=0.0,
            )
            answer = (resp.choices[0].message.content or "").strip()
            if "<think>" in answer:
                answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

            print(f"[DEBUG] Judge raw answer: '{answer[:500]}'")

            # Strip markdown code fences (```json ... ```) if present
            answer = re.sub(r'^```(?:json)?\s*\n?', '', answer)
            answer = re.sub(r'\n?```\s*$', '', answer)
            answer = answer.strip()

            import json
            parsed = json.loads(answer)
            judgments = [
                step.get("judgment", "") != "不合理"
                for step in parsed.get("steps", [])
            ]

            print(f"[DEBUG] Judgments: {judgments}")               # <-- 再加这一行，看解析结果
            if len(judgments) != traj["num_steps"]:
                judgments = [True] * traj["num_steps"]
        except Exception as e:
            logger.warning(f"Judge call failed: {e}, keeping all steps")
            judgments = [True] * traj["num_steps"]
        results.append(judgments)
    return results

'''
def apply_step_level_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    judge_config: Optional[dict] = None,
) -> torch.Tensor:
    """对 batch 中每条 trajectory 做逐步工具调用判断，修改 response_mask。
    
    Args:
        responses: (bs, response_length) response token ids
        response_mask: (bs, response_length) 原始 response mask
        tokenizer: tokenizer 用于解码
        judge_config: 配置，含 base_url, model 等
    
    Returns:
        modified_mask: (bs, response_length)
    """
    config = {**JUDGE_CONFIG, **(judge_config or {})}
    modified_mask = response_mask.clone()
    bs = responses.shape[0]
    
    # Step 1: decode 并找到每条 trajectory 中 think+tool_call 的 token 范围
    trajectories = []
    all_spans = []
    for i in range(bs):
        resp_ids = responses[i].tolist()
        text = tokenizer.decode(resp_ids, skip_special_tokens=False)
        spans = _find_think_toolcall_spans(text, tokenizer)
        all_spans.append(spans)
        trajectories.append({"text": text, "num_steps": len(spans)})
    
    # Step 2: 调用大模型判断
    judgments = _call_judge(trajectories, config)
    
    # Step 3: 根据判断结果修改 mask
    for i in range(bs):
        spans = all_spans[i]
        judg = judgments[i]
        for step_idx, (tok_start, tok_end) in enumerate(spans):
            if step_idx < len(judg) and not judg[step_idx]:
                end = min(tok_end + 1, modified_mask.shape[1])
                start = max(tok_start, 0)
                modified_mask[i, start:end] = 0
    
    return modified_mask
'''

import logging

logger = logging.getLogger("step_mask_judge")

def apply_step_level_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    judge_config=None,
) -> torch.Tensor:
    config = {**JUDGE_CONFIG, **(judge_config or {})}
    modified_mask = response_mask.clone()
    bs = responses.shape[0]

    # Step 1: decode + 找 span
    trajectories = []
    all_spans = []
    for i in range(bs):
        resp_ids = responses[i].tolist()
        text = tokenizer.decode(resp_ids, skip_special_tokens=False)
        spans = _find_think_toolcall_spans(text, tokenizer)
        all_spans.append(spans)
        trajectories.append({"text": text, "num_steps": len(spans)})

    # Step 2: 调用大模型判断
    judgments = _call_judge(trajectories, config)

    # Step 3: 根据判断结果修改 mask，并记录日志
    total_steps = 0
    masked_steps = 0
    for i in range(bs):
        spans = all_spans[i]
        judg = judgments[i]
        for step_idx, (tok_start, tok_end) in enumerate(spans):
            total_steps += 1
            if step_idx < len(judg) and not judg[step_idx]:
                masked_steps += 1
                end = min(tok_end + 1, modified_mask.shape[1])
                start = max(tok_start, 0)
                modified_mask[i, start:end] = 0

                # ---- 关键日志：检测到不合理工具调用 ----
                snippet = trajectories[i]["text"][tok_start*4 : tok_start*4+80]  # 粗略截取
                logger.info(
                    f"[StepMask] trajectory={i}, step={step_idx+1}: "
                    f"UNREASONABLE tool call detected, "
                    f"masking tokens [{start}:{end}]. "
                    f"Snippet: {snippet}..."
                )

    # ---- 汇总日志 ----
    if masked_steps > 0:
        print(
            f"\n{'='*60}\n"
            f"[StepMask] Step-level mask applied:\n"
            f"  Total trajectories: {bs}\n"
            f"  Total tool-call steps: {total_steps}\n"
            f"  Masked (unreasonable): {masked_steps}\n"
            f"  Kept (reasonable):     {total_steps - masked_steps}\n"
            f"{'='*60}\n"
        )
    else:
        print(f"[StepMask] All {total_steps} tool-call steps across {bs} trajectories are reasonable. No masking applied.")

    return modified_mask