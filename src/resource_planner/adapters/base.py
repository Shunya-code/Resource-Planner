from abc import ABC, abstractmethod
from typing import Any

from ..schemas import (
    Environment,
    ExecutionPlan,
    ModelCandidate,
    PlanningRequest,
)


class ModelProvider(ABC):
    """
    Discovers model candidates.

    Examples:
        - Hugging Face
        - Ollama registry
        - ModelScope
        - private model registry
        - local model directory
    """

    @abstractmethod
    async def search(
        self,
        request: PlanningRequest,
        limit: int = 20,
    ) -> list[ModelCandidate]:
        ...


class ResourceProvider(ABC):
    """
    Describes the environment where execution could happen.

    Examples:
        - local machine
        - Colab
        - cloud VM
        - Kubernetes node
        - Slurm cluster
    """

    @abstractmethod
    async def environment(self) -> Environment:
        ...


class FitProvider(ABC):
    """
    Determines whether a model/configuration can satisfy a workload
    on an environment.

    Examples:
        - llmfit
        - custom analytical estimator
        - benchmark database
        - runtime-specific estimator
    """

    @abstractmethod
    async def evaluate(
        self,
        request: PlanningRequest,
        candidates: list[ModelCandidate],
        environment: Environment,
    ) -> list[ExecutionPlan]:
        ...


class BenchmarkProvider(ABC):
    """
    Optional evidence provider.

    This is deliberately separate from fitting.

    A model may theoretically fit while having poor real-world
    performance for a particular workload.
    """

    @abstractmethod
    async def benchmark(
        self,
        request: PlanningRequest,
        plan: ExecutionPlan,
    ) -> dict[str, Any]:
        ...
