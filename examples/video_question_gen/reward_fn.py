# examples/video_question_gen/reward_fn.py
# 奖励函数入口：复用 evaluation_service 的多维度评估（格式 / 外延性 / 安全性 / 独立语义）。
# 配置中 custom_reward_function.path 可指向本文件或 evaluation_service，name 均为 compute_score。

#from .evaluation_service import compute_score  # noqa: F401
from .curriculum_reward import compute_score  # noqa: F401


