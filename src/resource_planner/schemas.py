from typing import Any, Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------
# What are we trying to do?
# ------------------------------------------------------------

class Task(BaseModel):
    name: str = Field(min_length=1)

    description: str | None = None

    # Examples:
    # text-generation
    # coding
    # embedding
    # classification
    # vision
    # speech
    # multimodal
    category: str | None = None


# ------------------------------------------------------------
# What does the workload look like?
# ------------------------------------------------------------

class Workload(BaseModel):
    operation: Literal[
        "inference",
        "training",
        "fine_tuning",
        "embedding",
        "evaluation",
        "unknown",
    ] = "inference"

    context_length: int | None = Field(
        default=None,
        ge=1,
    )

    input_tokens: int | None = Field(
        default=None,
        ge=1,
    )

    output_tokens: int | None = Field(
        default=None,
        ge=1,
    )

    concurrency: int = Field(
        default=1,
        ge=1,
    )

    requests_per_second: float | None = Field(
        default=None,
        gt=0,
    )

    latency_target_ms: float | None = Field(
        default=None,
        gt=0,
    )

    throughput_target_tokens_per_second: float | None = Field(
        default=None,
        gt=0,
    )


# ------------------------------------------------------------
# What resources are available?
# ------------------------------------------------------------

class Environment(BaseModel):
    type: Literal[
        "local",
        "colab",
        "cloud",
        "cluster",
        "unknown",
    ] = "local"

    cpu_cores: int | None = Field(
        default=None,
        ge=1,
    )

    ram_gb: float | None = Field(
        default=None,
        ge=0,
    )

    gpu_count: int = Field(
        default=0,
        ge=0,
    )

    gpu_memory_gb: float | None = Field(
        default=None,
        ge=0,
    )

    gpu_name: str | None = None

    accelerator: str | None = None

    storage_gb: float | None = Field(
        default=None,
        ge=0,
    )

    network_bandwidth_mbps: float | None = Field(
        default=None,
        gt=0,
    )


# ------------------------------------------------------------
# What are we allowed to do?
# ------------------------------------------------------------

class Constraints(BaseModel):
    allow_quantization: bool = True

    allow_cpu_offload: bool = True

    allow_multi_gpu: bool = True

    allow_cloud: bool = True

    max_cost_per_hour: float | None = Field(
        default=None,
        ge=0,
    )

    required_runtime: str | None = None

    required_license: str | None = None


# ------------------------------------------------------------
# Complete planning request
# ------------------------------------------------------------

class PlanningRequest(BaseModel):
    task: Task

    workload: Workload = Workload()

    environment: Environment = Environment()

    constraints: Constraints = Constraints()


# ------------------------------------------------------------
# Evidence
# ------------------------------------------------------------

class Evidence(BaseModel):
    source: str

    type: Literal[
        "metadata",
        "measured",
        "community_measured",
        "analytical_estimate",
        "user_supplied",
        "unknown",
    ] = "unknown"

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    details: dict[str, Any] = {}


# ------------------------------------------------------------
# Candidate model
# ------------------------------------------------------------

class ModelCandidate(BaseModel):
    model_id: str

    name: str | None = None

    architecture: str | None = None

    parameter_count: int | None = None

    active_parameter_count: int | None = None

    is_moe: bool | None = None

    context_length: int | None = None

    capabilities: list[str] = []

    license: str | None = None

    source: str | None = None

    metadata: dict[str, Any] = {}


# ------------------------------------------------------------
# One possible way to execute a model
# ------------------------------------------------------------

class ExecutionConfiguration(BaseModel):
    runtime: str | None = None

    quantization: str | None = None

    precision: str | None = None

    gpu_count: int | None = None

    cpu_offload: bool | None = None

    tensor_parallelism: int | None = None

    pipeline_parallelism: int | None = None

    context_length: int | None = None


# ------------------------------------------------------------
# Resource requirements
# ------------------------------------------------------------

class ResourceEstimate(BaseModel):
    vram_gb: float | None = None

    ram_gb: float | None = None

    storage_gb: float | None = None

    tokens_per_second: float | None = None

    latency_ms: float | None = None

    cost_per_hour: float | None = None


# ------------------------------------------------------------
# Final execution plan
# ------------------------------------------------------------

class ExecutionPlan(BaseModel):
    model: ModelCandidate

    configuration: ExecutionConfiguration

    resources: ResourceEstimate

    fit: Literal[
        "unknown",
        "unlikely",
        "possible",
        "good",
    ] = "unknown"

    evidence: list[Evidence] = []

    warnings: list[str] = []

    explanation: str | None = None


# ------------------------------------------------------------
# Planner response
# ------------------------------------------------------------

class PlanningResponse(BaseModel):
    request: PlanningRequest

    plans: list[ExecutionPlan]

    warnings: list[str] = []

    sources: list[str] = []


# ------------------------------------------------------------
# Model artifact
# ------------------------------------------------------------

class ModelArtifact(BaseModel):
    """
    A concrete downloadable/usable representation of a model.

    Examples:

        model.gguf
        model.Q4_K_M.gguf
        model.safetensors
        model.bin
        multimodal projector
    """

    artifact_id: str

    model_id: str

    filename: str

    format: str | None = None

    quantization: str | None = None

    precision: str | None = None

    size_bytes: int | None = None

    architecture: str | None = None

    source: str | None = None

    metadata: dict[str, Any] = {}


# ------------------------------------------------------------
# Model relationship
# ------------------------------------------------------------

class ModelRelationship(BaseModel):
    """
    Relationship between models/repositories.

    Examples:

        derived_from
        fine_tune_of
        quantized_from
        adapter_for
        base_model
    """

    relationship: str

    source_model_id: str

    target_model_id: str

    metadata: dict[str, Any] = {}


# ------------------------------------------------------------
# Expanded model candidate
# ------------------------------------------------------------

class ModelCandidateV2(BaseModel):
    """
    Canonical representation of a model independent of any
    particular model registry.
    """

    model_id: str

    name: str | None = None

    architecture: str | None = None

    parameter_count: int | None = None

    active_parameter_count: int | None = None

    is_moe: bool | None = None

    context_length: int | None = None

    capabilities: list[str] = []

    license: str | None = None

    base_model_id: str | None = None

    artifacts: list[ModelArtifact] = []

    relationships: list[ModelRelationship] = []

    source: str | None = None

    metadata: dict[str, Any] = {}

    
# ------------------------------------------------------------
# Fit analysis
# ------------------------------------------------------------

class FitResult(BaseModel):
    """
    Result of evaluating a concrete model artifact against
    an environment and workload.
    """

    artifact_id: str

    fit: Literal[
        "unknown",
        "unlikely",
        "possible",
        "good",
    ] = "unknown"

    vram_required_gb: float | None = None

    ram_required_gb: float | None = None

    storage_required_gb: float | None = None

    estimated_tokens_per_second: float | None = None

    estimated_latency_ms: float | None = None

    estimated_cost_per_hour: float | None = None

    runtime: str | None = None

    quantization: str | None = None

    cpu_offload: bool = False

    warnings: list[str] = []

    evidence: list[Evidence] = []

    metadata: dict[str, Any] = {}


# ------------------------------------------------------------
# Artifact execution plan
# ------------------------------------------------------------

class ArtifactExecutionPlan(BaseModel):
    """
    A concrete way of executing a particular artifact.
    """

    model_id: str

    artifact: ModelArtifact

    configuration: ExecutionConfiguration

    fit: FitResult

    explanation: str | None = None
