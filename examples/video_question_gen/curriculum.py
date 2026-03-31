# Copyright (c) 2025. 课程学习阶段管理：按奖励是否提升自适应切换阶段。
"""
课程学习阶段定义：
- Stage 0: format + safety（格式与安全性）
- Stage 1: format + extensibility + safety（加入外延性）
- Stage 2: format + extensibility + safety + quality（加入综合质量：独立语义 + 人类吸引力）

阶段切换策略：当前阶段连续 no_improvement_steps 步奖励无提升则进入下一阶段。

与新版 compute_score 配合：
- compute_score 通过 config_override["dimensions"] 控制当前阶段评估哪些维度
- compute_score 通过 config_override["dimension_weights"] 控制各维度权重
- CurriculumManager 提供 get_config_override() 方法，输出可直接传入 config_override
"""

from typing import Dict, List, Optional


# 各阶段评估维度
CURRICULUM_STAGE_DIMENSIONS: List[List[str]] = [
    ["format", "safety"],                              # 第一阶段：格式 + 安全性
    ["format", "extensibility", "safety"],             # 第二阶段：加入外延性
    ["format", "extensibility", "safety", "quality"],  # 第三阶段：加入综合质量（独立语义+人类吸引力）
]

# 各阶段对应的维度权重（仅含当前阶段激活的维度，权重和为 1）
CURRICULUM_STAGE_WEIGHTS: List[Dict[str, float]] = [
    {"format": 0.5, "safety": 0.5},
    {"format": 0.2, "extensibility": 0.4, "safety": 0.4},
    {"format": 0.2, "extensibility": 0.3, "safety": 0.2, "quality": 0.3},
]


class CurriculumManager:
    """课程学习阶段管理器：根据近期奖励是否提升自适应切换阶段。"""

    def __init__(
        self,
        stages: Optional[List[List[str]]] = None,
        stage_weights: Optional[List[Dict[str, float]]] = None,
        no_improvement_steps_to_advance: int = 20,
        initial_stage: int = 0,
    ):
        self.stages = stages or CURRICULUM_STAGE_DIMENSIONS
        self.stage_weights = stage_weights or CURRICULUM_STAGE_WEIGHTS
        # 若用户自定义 stages 但未提供 stage_weights，则各维度等权
        if stages is not None and stage_weights is None:
            self.stage_weights = [
                {d: 1.0 / len(dims) for d in dims} for dims in self.stages
            ]
        self.no_improvement_steps_to_advance = no_improvement_steps_to_advance
        self._current_stage = max(0, min(initial_stage, len(self.stages) - 1))
        self._best_reward_in_stage: float = float("-inf")
        self._steps_without_improvement: int = 0

    def get_stage(self) -> int:
        return self._current_stage

    def get_dimensions(self) -> List[str]:
        return list(self.stages[self._current_stage])

    def get_dimension_weights(self) -> Dict[str, float]:
        idx = self._current_stage
        if idx < len(self.stage_weights):
            return dict(self.stage_weights[idx])
        return {d: 1.0 / len(self.stages[idx]) for d in self.stages[idx]}

    def get_config_override(self) -> Dict:
        """返回可直接传给 compute_score(config_override=...) 的字典。"""
        return {
            "dimensions": self.get_dimensions(),
            "dimension_weights": self.get_dimension_weights(),
        }

    def update(self, mean_reward: float) -> dict:
        """
        用本步平均奖励更新状态，必要时进入下一阶段。

        Returns:
            dict: 含 curriculum_stage, curriculum_advanced, steps_without_improvement 等，便于打日志。
        """
        info = {
            "curriculum_stage": self._current_stage,
            "curriculum_dimensions": self.get_dimensions(),
            "curriculum_advanced": False,
            "steps_without_improvement": self._steps_without_improvement,
            "best_reward_in_stage": self._best_reward_in_stage,
        }
        if mean_reward > self._best_reward_in_stage:
            self._best_reward_in_stage = mean_reward
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        if (
            self._steps_without_improvement >= self.no_improvement_steps_to_advance
            and self._current_stage < len(self.stages) - 1
        ):
            self._current_stage += 1
            self._best_reward_in_stage = float("-inf")
            self._steps_without_improvement = 0
            info["curriculum_advanced"] = True
            info["curriculum_stage"] = self._current_stage
            info["curriculum_dimensions"] = self.get_dimensions()

        return info
