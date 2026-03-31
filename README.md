# Long-Video Understanding & Question Generation with GRPO and Multi-Turn Tool Use

This repository is a focused build of **verl** for training a multimodal policy that watches long videos, calls tools (`video_slice`, `entity_search`, `asr`, `ocr`, etc.) over multiple turns, and produces **high-quality, substantive guiding questions** in a **danmaku-style** setting where **answers lie outside the video** (extrinsic knowledge).

![alt text](Gemini_Generated_Image_nb1tanb1tanb1tan.png)

## Features

- **GRPO (Group Relative Policy Optimization)**  
  For each prompt, multiple trajectories are sampled; advantages are estimated **within the group** for stable RL updates.

- **Multi-turn agent**  
  Supports multi-round tool calls in **Hermes-style** `<tool_call>...</tool_call>` format, with up to **five** assistant turns.

- **Tools**  
  - **video_slice** — segment-level video descriptions  
  - **entity_search** — entity retrieval  
  - **asr** — speech transcription  
  - **ocr** — text on key frames  

- **Curriculum learning**  
  Evaluation dimensions are introduced in stages: **format → extrinsic relevance → safety → quality**. When the reward shows no improvement for a period, training **advances to the next stage**.

- **Step-level mask**  
  Before computing advantages, a **large model scores each tool-call step** for plausibility; **implausible steps are excluded from the GRPO gradient**.

## Project layout

```
.
├── verl/                           # Training stack (PPO/GRPO, agent_loop, reward_loop, …)
├── examples/video_question_gen/    # Long-video question-generation example
│   ├── train.sh                  # Training entry script
│   ├── preprocess.py             # Preprocessing → train/val.parquet
│   ├── reward_fn.py              # Reward entry (→ curriculum_reward)
│   ├── curriculum_reward.py      # Curriculum-style multi-objective reward
│   ├── curriculum.py             # Curriculum stage manager
│   ├── step_mask_judge.py        # Step-level tool-call plausibility judge
│   ├── evaluation_service.py     # Multi-aspect eval (format / extrinsic / safety / quality)
│   ├── video_slice_tool.py
│   ├── entity_search_tool.py
│   ├── asr_tool.py / ocr_tool.py
│   ├── config/
│   │   ├── video_question_grpo.yaml   # Main training config
│   │   └── tool_config.yaml              # Tool definitions & API endpoints
│   └── data/                       # train.parquet, val.parquet (generate beforehand)
├── requirements.txt
├── setup.py
└── pyproject.toml
```

**Scope:** This repo keeps only what is needed for `examples/video_question_gen` and `python -m verl.trainer.main_ppo` on the PPO/GRPO path with multi-turn tools. Unrelated upstream examples were removed for a smaller, task-specific tree.

## Environment and dependencies

**Python:** 3.10+

```bash
pip install -e .
# or
pip install -r requirements.txt
```

Core deps include: `accelerate`, `hydra-core`, `ray`, `transformers`, `peft`, `wandb`, `tensorboard`, `pyarrow`, `tensordict`, and others as pinned in the requirement files.

Optional backends (pick what matches your hardware):

```bash
pip install -r requirements_sglang.txt
# or
pip install -r requirements-cuda.txt
# or
pip install -r requirements-npu.txt
```

If you use the **Megatron** backend, configure the **Megatron-LM** submodule per `.gitmodules`.

## Data preparation

1. Prepare a list of video paths (one path per line, or a directory root containing `.mp4` files).
2. Run preprocessing:

```bash
cd examples/video_question_gen
python preprocess.py \
  --video_list /path/to/video_list.txt \
  --local_save_dir ./data \
  [--data_root /path/to/videos]
```

This writes `train.parquet` and `val.parquet` under `./data`, with fields such as `prompt`, `videos`, and `extra_info` (including `tools_kwargs` where applicable).

## Configuration

| Area | Location / keys |
|------|------------------|
| Main training | `examples/video_question_gen/config/video_question_grpo.yaml` |
| Custom reward | `reward.custom_reward_function.path` → e.g. `examples.video_question_gen.reward_fn`, `name`: `compute_score` |
| Curriculum | `trainer.curriculum_learning` (stages and dimension weights) |
| Step-level mask | `algorithm.step_level_mask` (judge `base_url`, `timeout`, fallbacks, …) |
| Tools | `examples/video_question_gen/config/tool_config.yaml` |

In `tool_config.yaml`, each tool has `class_name`, `config` (`api_base`, `api_model`, `api_key`, …), and `tool_schema` (`name`, `description`, `parameters`).

Typical overrides (in `train.sh` or the CLI):

- `actor_rollout_ref.model.path` — policy / rollout model (e.g. Qwen3-VL-4B-Thinking)
- `actor_rollout_ref.rollout.multi_turn.tool_config_path` — **absolute** path to `tool_config.yaml`
- `data.train_files` / `data.val_files` — **absolute** paths to the parquet files
- `algorithm.step_level_mask.base_url` — HTTP endpoint for the step-judge model (can match `entity_search` if you reuse the same service)

## Run training

From the **repository root**, set `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

Edit `examples/video_question_gen/train.sh` for your machine:

- `PROJECT_DIR` — absolute repo root  
- `CONFIG_PATH` — e.g. `$PROJECT_DIR/examples/video_question_gen/config`  
- `TRAIN_FILES` / `VAL_FILES` — absolute paths to `train.parquet` / `val.parquet`  
- Model path and `tool_config.yaml` path as above  

Then:

```bash
bash examples/video_question_gen/train.sh
```

Equivalent one-shot launch (still from repo root; add more Hydra overrides as needed):

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/video_question_gen/config"
TRAIN_FILES="$PROJECT_DIR/examples/video_question_gen/data/train.parquet"
VAL_FILES="$PROJECT_DIR/examples/video_question_gen/data/val.parquet"

python -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name=video_question_grpo \
  "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PROJECT_DIR" \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  actor_rollout_ref.model.path=/path/to/your/Qwen3-VL-4B-Thinking \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/video_question_gen/config/tool_config.yaml"
```

Rollout uses **SGLang**; actor / reference training uses **FSDP**. Logging is controlled via `trainer.logger` (e.g. console, Weights & Biases, SwanLab).

## Tools and external services

| Tool | Notes |
|------|--------|
| **video_slice** | Remote **vLLM** (VL model) for segment descriptions; set `api_base`, `api_model` in `tool_config.yaml`. |
| **entity_search** | Retrieval / generation service (e.g. Qwen3-14B); the step-level mask judge can share the same endpoint. |
| **asr** / **ocr** | Configure `api_base` or a local API as needed; leaving them unused is OK for the core pipeline if you do not rely on those modalities. |

Start these services **before** training and keep names and URLs aligned across `tool_config.yaml` and `video_question_grpo.yaml`.

## License

This repository is under **Apache-2.0**; bundled dependencies and submodules follow their own licenses. See `LICENSE`.
