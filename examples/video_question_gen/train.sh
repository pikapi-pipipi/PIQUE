set -e
set -x
ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="$SCRIPT_DIR/config"

# 确保能 import examples.video_question_gen.*
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export VERL_LOG_TRAJECTORY=1
export VERL_LOGGING_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 数据路径（可按需覆盖）
TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/examples/video_question_gen/data/train.parquet}"
VAL_FILES="${VAL_FILES:-$PROJECT_DIR/examples/video_question_gen/data/val.parquet}"

NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name=video_question_grpo \
  "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PROJECT_DIR" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_batch_size=8 \
  data.max_prompt_length=16384 \
  data.max_response_length=16384 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  actor_rollout_ref.model.path=/mnt/storage01/Qwen/Qwen3-VL-4B-Thinking \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.actor.loss_agg_mode='"token-mean"'\
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.over_sample_rate=0.1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","swanlab"]' \
  trainer.project_name=video_question_grpo \
  trainer.experiment_name=qwen3-vl-4b-video-question-tool \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  "$@"