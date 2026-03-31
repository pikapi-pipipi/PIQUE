# examples/video_question_gen/data/preprocess_video_question_data.py
# 将视频路径列表转为 agentic RL 所需的 parquet（不修改项目内任何文件）。
# 用法：python preprocess_video_question_data.py --video_list videos.txt --local_save_dir ~/data/video_question_gen [--data_root /path/to/videos]
import argparse
import json
import os
import random

import pandas as pd


SYSTEM_PROMPT = """\
你是一个视频理解专家。你的任务是观看用户提供的视频，然后生成 **一个高质量的、有深度的问题**。

## 工具
你有四个可选工具，可根据需要灵活组合使用，也可以跳过所有工具直接生成问题。

### 调用限制
- **video_slice、asr、ocr 每个最多调用一次**。
- **entity_search 可调用多次**，每次针对不同的实体关键词进行检索，以获取充分的背景知识。

### 工具说明
1. **video_slice**：当视频内容复杂、时间较长或需要分段理解时调用，获取视频各片段的摘要。格式：
   <tool_call>{"name": "video_slice", "arguments": {}}</tool_call>

2. **entity_search**：当从视频中识别出 IP、品牌、人物、流行语、专有名词等实体，需要获取背景知识时调用。可多次调用以检索不同实体。格式：
   <tool_call>{"name": "entity_search", "arguments": {"keywords": "关键词1, 关键词2"}}</tool_call>
   （keywords 必须是具体实体名称，禁止模糊搜索。）

3. **asr**：对整段视频音频进行转录，获取旁白/对白文本以辅助理解。格式：
   <tool_call>{"name": "asr", "arguments": {}}</tool_call>

4. **ocr**：从视频关键帧中识别出现的文字（字幕、标题、画面内文字）以辅助理解。格式：
   <tool_call>{"name": "ocr", "arguments": {}}</tool_call>

### 合法路径示例
- 直接生成问题
- video_slice → 生成
- asr → entity_search → 生成
- video_slice → entity_search → entity_search → 生成
- ocr → entity_search → 生成
- video_slice → asr → entity_search → entity_search → 生成
- 以及其他任意合理组合……

## 规则
1. 视频是出发点，需要针对视频的具体内容进行提问，问题本身需要强相关，但方向可以发散，以满足用户多样的求知欲；
2. 问题需要像用户提问般自然、语义完整，避免陈述句；
3. 问题以弹幕形式出现，所以需要简短，每个问题只能出现一次疑问，以一个问号结尾；
4. 问题需要启发用户思考，激起用户离开本视频去搜索问题的欲望，答案需要在视频内容以外。
5. 生成的问题要包含大致时间戳

## 格式要求
1. 每轮只输出一个动作：一次工具调用或一个最终问题，不要在同一轮中混合输出。
2. 最终生成问题时，输出格式：<question>[时间戳]你生成的问题内容</question>，例如 <question>[01:23]这个角色是谁？</question>
"""


def video_path_to_file_url(path: str) -> str:
    """转为 processor/vision 使用的 file:// URL。"""
    path = os.path.abspath(path)
    if not path.startswith("/"):
        path = "/" + path
    return "file://" + path


def build_row(
    item: dict,
    split: str,
    index: int,
) -> dict:
    """
    单条样本构建。

    item 来自 train_videos.json，包含：
      - video_path: 原始路径如 /mnt/cubefs/lilianrui/video/31492408383.mp4
      - category: 视频类别
      - reason: 分类理由
      - richness_score: 丰富度评分
    """
    fs_path = item["video_path"]
    file_url = video_path_to_file_url(fs_path)

    return {
        "data_source": "video_question_gen",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "<video>\n请观看这个视频，并生成一个高质量的问题。",
            },
        ],
        "videos": [{"video": file_url}],
        # NaiveRewardManager 要求 reward_model 中有 ground_truth；开放域问题生成无标准答案，用空字符串占位
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": index,
            # 保留原始 JSON 中的元信息，方便后续分析
            "category": item.get("category", ""),
            "reason": item.get("reason", ""),
            "richness_score": item.get("richness_score", 0),
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "video_slice": {
                    "create_kwargs": {
                        "video_path": fs_path,
                    },
                },
                "entity_search": {
                    "create_kwargs": {"_": 0},
                },
                "asr": {
                    "create_kwargs": {
                        "video_path": fs_path,
                    },
                },
                "ocr": {
                    "create_kwargs": {
                        "video_path": fs_path,
                    },
                },
            },
            "agent_name": "tool_agent",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Video question gen parquet from train_videos.json")
    parser.add_argument(
        "--json_path",
        default="/mnt/storage01/filtered_videos_under_3min.json",
        help="Path to the JSON file (train_videos.json).",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/mnt/storage01/verl_agentic_curr/examples/video_question_gen/data",
        help="Directory to save train.parquet and val.parquet.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of samples to use as validation (0~1).",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=0,
        help="Number of videos to randomly sample (0 = use all, default: 0).",
    )
    parser.add_argument(
        "--min_richness",
        type=int,
        default=0,
        help="Only keep videos with richness_score >= this value (default: 0, no filter).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Only keep videos in these categories (default: None, keep all). E.g. --categories 科普 影视",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible sampling and splitting.",
    )
    args = parser.parse_args()

    # ---------- 1. 读取 JSON ----------
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {args.json_path}")

    # ---------- 2. 过滤 ----------
    if args.min_richness > 0:
        data = [d for d in data if d.get("richness_score", 0) >= args.min_richness]
        print(f"After richness filter (>= {args.min_richness}): {len(data)}")

    if args.categories:
        cat_set = set(args.categories)
        data = [d for d in data if d.get("category", "") in cat_set]
        print(f"After category filter ({args.categories}): {len(data)}")

    if not data:
        raise SystemExit("No data left after filtering!")

    # ---------- 3. 抽样 ----------
    random.seed(args.seed)
    random.shuffle(data)

    if args.sample_num > 0 and len(data) > args.sample_num:
        data = data[: args.sample_num]
        print(f"Randomly sampled {args.sample_num} videos (seed={args.seed}).")
    else:
        print(f"Using all {len(data)} videos.")

    # ---------- 4. 划分 train / val ----------
    n = len(data)
    n_val = max(0, int(n * args.val_ratio))
    n_train = n - n_val
    train_items = data[:n_train]
    val_items = data[n_train:] if n_val else []

    rows_train = [build_row(item, "train", i) for i, item in enumerate(train_items)]
    rows_val = [build_row(item, "val", i) for i, item in enumerate(val_items)]

    # ---------- 5. 保存 ----------
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(save_dir, "train.parquet")
    val_path = os.path.join(save_dir, "val.parquet")

    pd.DataFrame(rows_train).to_parquet(train_path, index=False)
    pd.DataFrame(rows_val).to_parquet(val_path, index=False)

    # ---------- 6. 统计信息 ----------
    from collections import Counter
    cat_counter = Counter(item.get("category", "") for item in train_items + val_items)

    print(f"\n{'='*50}")
    print(f"Saved {len(rows_train)} train -> {train_path}")
    print(f"Saved {len(rows_val)} val   -> {val_path}")
    print(f"\nCategory distribution in dataset:")
    for cat, cnt in sorted(cat_counter.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt} ({cnt / n * 100:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
