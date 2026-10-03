
from typing import Any

import httpx

from ..schemas import (
    ModelArtifact,
    ModelCandidateV2,
    PlanningRequest,
)
from .base import ModelProvider


class HuggingFaceProvider(ModelProvider):
    """
    Hugging Face implementation of ModelProvider.

    Responsibilities:
    - discover candidate repositories
    - inspect individual repositories
    - normalize Hugging Face metadata
    - extract executable artifacts
    - preserve source metadata
    """

    def __init__(
        self,
        base_url: str = "https://huggingface.co/api",
        token: str | None = None,
        timeout: float = 20.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.headers = (
            {"Authorization": f"Bearer {token}"}
            if token
            else {}
        )

    async def search(
        self,
        request: PlanningRequest,
        limit: int = 20,
    ) -> list[ModelCandidateV2]:
        """
        Discover Hugging Face models for a planning request.

        Discovery is intentionally tolerant of free-form task names.
        Known task categories are mapped to Hugging Face pipeline tags.

        Unknown task names do not become restrictive search queries;
        instead, the adapter performs a broad model discovery request.
        """

        params: dict[str, Any] = {
            "limit": limit,
            "full": "true",
        }

        category = (
            request.task.category
            or ""
        ).strip().lower()

        task_name = (
            request.task.name
            or ""
        ).strip().lower()

        task_map = {
            "coding": "text-generation",
            "coding assistant": "text-generation",
            "chat": "text-generation",
            "text generation": "text-generation",
            "text-generation": "text-generation",
            "embedding": "feature-extraction",
            "classification": "text-classification",
            "summarization": "summarization",
            "translation": "translation",
            "question answering": "question-answering",
            "question-answering": "question-answering",
        }

        pipeline_tag = (
            task_map.get(category)
            or task_map.get(task_name)
        )

        if pipeline_tag:
            params["pipeline_tag"] = pipeline_tag

        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
        ) as client:

            response = await client.get(
                f"{self.base_url}/models",
                params=params,
            )

            response.raise_for_status()

            data = response.json()

        if not isinstance(data, list):
            return []

        candidates = []

        for item in data:
            if not isinstance(item, dict):
                continue

            if not item.get("id"):
                continue

            candidates.append(
                self._normalize_search_result(item)
            )

        return candidates

    async def inspect(
        self,
        model_id: str,
    ) -> ModelCandidateV2:

        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
        ) as client:

            response = await client.get(
                f"{self.base_url}/models/{model_id}"
            )

            response.raise_for_status()

            data = response.json()

        return self._normalize_repository(data)

    @classmethod
    def _normalize_search_result(
        cls,
        item: dict[str, Any],
    ) -> ModelCandidateV2:

        card = item.get("cardData") or {}

        tags = [
            tag
            for tag in item.get("tags", [])
            if isinstance(tag, str)
        ]

        base_model_id = cls._extract_base_model(
            card,
            tags,
        )

        return ModelCandidateV2(
            model_id=item["id"],
            name=item.get("id"),

            license=(
                card.get("license")
                or item.get("license")
            ),

            base_model_id=base_model_id,

            capabilities=tags,

            source="huggingface",

            metadata={
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "pipeline_tag": item.get("pipeline_tag"),
                "library_name": item.get("library_name"),
                "author": item.get("author"),
                "createdAt": item.get("createdAt"),
                "lastModified": item.get("lastModified"),
            },
        )

    @classmethod
    def _normalize_repository(
        cls,
        item: dict[str, Any],
    ) -> ModelCandidateV2:

        card = item.get("cardData") or {}
        config = item.get("config") or {}

        tags = [
            tag
            for tag in item.get("tags", [])
            if isinstance(tag, str)
        ]

        base_model_id = cls._extract_base_model(
            card,
            tags,
        )

        artifacts: list[ModelArtifact] = []

        for sibling in item.get("siblings", []):

            if not isinstance(sibling, dict):
                continue

            filename = sibling.get("rfilename")

            if not filename:
                continue

            artifact = cls._artifact_from_filename(
                model_id=item["id"],
                filename=filename,
                sibling=sibling,
            )

            if artifact:
                artifacts.append(artifact)

        architecture = cls._extract_architecture(
            config,
            tags,
        )

        parameter_count = cls._extract_parameter_count(
            config,
            card,
            item,
        )

        active_parameter_count = (
            cls._extract_active_parameter_count(
                config,
                card,
                item,
            )
        )

        is_moe = cls._extract_is_moe(
            config,
            tags,
        )

        context_length = cls._extract_context_length(
            config,
        )

        return ModelCandidateV2(
            model_id=item["id"],
            name=item.get("id"),

            architecture=architecture,

            parameter_count=parameter_count,

            active_parameter_count=active_parameter_count,

            is_moe=is_moe,

            context_length=context_length,

            capabilities=tags,

            license=(
                card.get("license")
                or item.get("license")
            ),

            base_model_id=base_model_id,

            artifacts=artifacts,

            source="huggingface",

            metadata={
                "author": item.get("author"),
                "createdAt": item.get("createdAt"),
                "lastModified": item.get("lastModified"),
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "library_name": item.get("library_name"),
                "pipeline_tag": item.get("pipeline_tag"),
                "usedStorage": item.get("usedStorage"),
                "card_base_model": card.get("base_model"),
                "config": config,
            },
        )

    @staticmethod
    def _extract_base_model(
        card: dict[str, Any],
        tags: list[str],
    ) -> str | None:

        base_model = card.get("base_model")

        if isinstance(base_model, list):

            for value in base_model:

                if isinstance(value, str) and value:
                    return value

                if isinstance(value, dict):

                    model_id = (
                        value.get("model")
                        or value.get("id")
                    )

                    if model_id:
                        return model_id

        if isinstance(base_model, str):
            return base_model

        for tag in tags:

            if not tag.startswith("base_model:"):
                continue

            value = tag.split(
                "base_model:",
                1,
            )[1]

            if value.startswith("quantized:"):
                continue

            if value:
                return value

        return None

    @staticmethod
    def _extract_architecture(
        config: dict[str, Any],
        tags: list[str],
    ) -> str | None:

        architecture = (
            config.get("model_type")
            or config.get("architectures")
        )

        if isinstance(architecture, list):

            if architecture:
                value = architecture[0]

                if isinstance(value, str):
                    return value

        if isinstance(architecture, str):
            return architecture

        architecture_tags = (
            "llama",
            "qwen",
            "gemma",
            "mistral",
            "phi",
            "deepseek",
            "glm",
        )

        for tag in tags:

            tag_lower = tag.lower()

            for architecture_name in architecture_tags:

                if architecture_name in tag_lower:
                    return architecture_name

        return None

    @staticmethod
    def _extract_parameter_count(
        config: dict[str, Any],
        card: dict[str, Any],
        item: dict[str, Any],
    ) -> int | None:

        candidates = (
            config.get("num_parameters"),
            config.get("parameter_count"),
            card.get("parameter_count"),
            item.get("parameter_count"),
        )

        for value in candidates:

            parsed = HuggingFaceProvider._parse_parameter_count(
                value
            )

            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _extract_active_parameter_count(
        config: dict[str, Any],
        card: dict[str, Any],
        item: dict[str, Any],
    ) -> int | None:

        candidates = (
            config.get("num_active_parameters"),
            config.get("active_parameter_count"),
            card.get("active_parameter_count"),
            item.get("active_parameter_count"),
        )

        for value in candidates:

            parsed = HuggingFaceProvider._parse_parameter_count(
                value
            )

            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _parse_parameter_count(
        value: Any,
    ) -> int | None:

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        if not isinstance(value, str):
            return None

        text = value.strip().lower()

        multipliers = {
            "k": 1_000,
            "m": 1_000_000,
            "b": 1_000_000_000,
            "t": 1_000_000_000_000,
        }

        try:

            if text[-1:] in multipliers:

                return int(
                    float(text[:-1])
                    * multipliers[text[-1]]
                )

            return int(float(text))

        except (ValueError, IndexError):

            return None

    @staticmethod
    def _extract_is_moe(
        config: dict[str, Any],
        tags: list[str],
    ) -> bool | None:

        explicit = config.get("is_moe")

        if isinstance(explicit, bool):
            return explicit

        if any(
            "moe" in tag.lower()
            for tag in tags
        ):
            return True

        expert_keys = (
            "num_experts",
            "num_local_experts",
            "num_experts_per_tok",
        )

        if any(
            key in config
            for key in expert_keys
        ):
            return True

        return None

    @staticmethod
    def _extract_context_length(
        config: dict[str, Any],
    ) -> int | None:

        keys = (
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "seq_length",
            "model_max_length",
        )

        for key in keys:

            value = config.get(key)

            if isinstance(value, int) and value > 0:
                return value

        return None

    @staticmethod
    def _artifact_from_filename(
        model_id: str,
        filename: str,
        sibling: dict[str, Any],
    ) -> ModelArtifact | None:

        lower = filename.lower()

        known_artifact_extensions = (
            ".gguf",
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
            ".onnx",
        )

        if not lower.endswith(
            known_artifact_extensions
        ):
            return None

        fmt = lower.rsplit(
            ".",
            1,
        )[-1]

        quantization = (
            HuggingFaceProvider
            ._extract_quantization(lower)
        )

        precision = (
            HuggingFaceProvider
            ._extract_precision(lower)
        )

        size_bytes = sibling.get("size")

        if not isinstance(size_bytes, int):
            size_bytes = None

        return ModelArtifact(
            artifact_id=(
                f"{model_id}:{filename}"
            ),

            model_id=model_id,

            filename=filename,

            format=fmt,

            quantization=quantization,

            precision=precision,

            size_bytes=size_bytes,

            source="huggingface",

            metadata={
                "lfs": sibling.get("lfs"),
            },
        )

    @staticmethod
    def _extract_quantization(
        filename_lower: str,
    ) -> str | None:

        markers = (
            "iq4_xs",
            "iq4_nl",
            "iq3_xxs",
            "iq3_xs",
            "iq2_xxs",
            "iq2_xs",
            "q8_0",
            "q8_k",
            "q6_k",
            "q5_k_m",
            "q5_k_s",
            "q5_0",
            "q5_1",
            "q4_k_m",
            "q4_k_s",
            "q4_0",
            "q4_1",
            "q3_k_l",
            "q3_k_m",
            "q3_k_s",
            "q2_k",
            "q2_0",
            "pq2_0",
            "ptq1_0",
        )

        for marker in markers:

            if marker in filename_lower:
                return marker.upper()

        return None

    @staticmethod
    def _extract_precision(
        filename_lower: str,
    ) -> str | None:

        markers = (
            "bf16",
            "fp16",
            "fp8",
            "fp32",
            "f16",
            "f8",
            "f32",
        )

        for marker in markers:

            if marker in filename_lower:
                return marker.upper()

        return None
