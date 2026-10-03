from __future__ import annotations

from typing import Any

import httpx

from ..schemas import ModelCandidateV2, PlanningRequest


class LlmfitProvider:
    """
    Adapter for the LLMFit HTTP API.

    The provider fetches LLMFit model records, matches them
    against ModelCandidateV2 objects, and preserves the complete
    LLMFit record inside candidate.metadata["llmfit"].
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ============================================================
    # SYSTEM
    # ============================================================

    async def system(self) -> Any:
        """
        Fetch the LLMFit system/environment description.

        Converts the API response into the project's Environment
        schema when possible.
        """

        async with httpx.AsyncClient(
            timeout=self.timeout
        ) as client:

            response = await client.get(
                f"{self.base_url}/api/v1/system"
            )

            response.raise_for_status()

            payload = response.json()

        system_data = {}

        if isinstance(payload, dict):
            system_data = payload.get(
                "system",
                {}
            )

        if not isinstance(system_data, dict):
            system_data = {}

        # Import here to avoid unnecessary module coupling at
        # import time.
        from ..schemas import Environment

        return Environment(
            type="local",
            cpu_cores=system_data.get("cpu_cores"),
            ram_gb=system_data.get("available_ram_gb"),
            gpu_count=system_data.get("gpu_count") or 0,
            gpu_memory_gb=system_data.get("gpu_available_gb"),
            gpu_name=system_data.get("gpu_name"),
            accelerator=system_data.get("backend"),
        )

    # ============================================================
    # EVALUATE
    # ============================================================

    async def evaluate(
        self,
        request: PlanningRequest,
        candidates: list[ModelCandidateV2],
        environment: Any = None,
    ) -> list[ModelCandidateV2]:

        raw_models = await self._fetch_models()

        evaluated: list[ModelCandidateV2] = []

        for candidate in candidates:

            match = self._match_candidate(
                candidate,
                raw_models,
            )

            self._apply_evaluation(
                candidate=candidate,
                match=match,
            )

            evaluated.append(candidate)

        return evaluated

    # ============================================================
    # HTTP
    # ============================================================

    async def _fetch_models(self) -> list[dict[str, Any]]:
        """
        Fetch all available LLMFit records.

        Primary endpoint:

            /api/v1/models

        A legacy fallback is retained for installations that
        expose /models or /api/models.
        """

        endpoints = (
            f"{self.base_url}/api/v1/models?limit=10000",
            f"{self.base_url}/api/v1/models",
            f"{self.base_url}/models",
            f"{self.base_url}/api/models",
        )

        async with httpx.AsyncClient(
            timeout=self.timeout
        ) as client:

            last_error: Exception | None = None

            for endpoint in endpoints:

                try:

                    response = await client.get(
                        endpoint
                    )

                    response.raise_for_status()

                    payload = response.json()

                    records = self._extract_records(
                        payload
                    )

                    if records:
                        return records

                except Exception as exc:

                    last_error = exc

        if last_error:
            raise last_error

        return []

    # ============================================================
    # RESPONSE NORMALIZATION
    # ============================================================

    @staticmethod
    def _extract_records(
        payload: Any,
    ) -> list[dict[str, Any]]:

        if isinstance(payload, list):

            return [
                item
                for item in payload
                if isinstance(item, dict)
            ]

        if not isinstance(payload, dict):
            return []

        for key in (
            "models",
            "results",
            "data",
            "items",
            "records",
        ):

            value = payload.get(key)

            if isinstance(value, list):

                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

        return []

    # ============================================================
    # MODEL NAME HELPERS
    # ============================================================

    @staticmethod
    def _normalize_model_id(
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        return value.lower()

    @staticmethod
    def _get_llmfit_name(
        item: dict[str, Any],
    ) -> str | None:

        for key in (
            "name",
            "model_id",
            "id",
            "ollama_name",
        ):

            value = item.get(key)

            normalized = (
                LlmfitProvider
                ._normalize_model_id(value)
            )

            if normalized:
                return normalized

        return None

    @staticmethod
    def _llmfit_base_models(
        item: dict[str, Any],
    ) -> list[str]:

        values: list[str] = []

        for key in (
            "base_model",
            "base_model_id",
            "model_id",
            "name",
        ):

            value = item.get(key)

            if isinstance(value, str) and value.strip():

                values.append(
                    value.strip().lower()
                )

        base_models = item.get("base_models")

        if isinstance(base_models, list):

            for value in base_models:

                if isinstance(value, str):
                    values.append(
                        value.strip().lower()
                    )

        return list(dict.fromkeys(values))

    def _extract_base_model_from_candidate(
        self,
        candidate: ModelCandidateV2,
    ) -> str | None:

        for value in (
            getattr(candidate, "base_model_id", None),
            getattr(candidate, "model_id", None),
        ):

            normalized = (
                self._normalize_model_id(value)
            )

            if normalized:
                return normalized

        return None

    # ============================================================
    # MATCHING
    # ============================================================

    def _match_candidate(
        self,
        candidate: ModelCandidateV2,
        raw_models: list[dict[str, Any]],
    ) -> dict[str, Any] | None:

        candidate_ids: list[str] = []

        for value in (
            getattr(candidate, "model_id", None),
            getattr(candidate, "name", None),
            getattr(candidate, "base_model_id", None),
        ):

            normalized = (
                self._normalize_model_id(value)
            )

            if normalized:
                candidate_ids.append(
                    normalized
                )

        if not candidate_ids:
            return None

        candidate_base = (
            self._extract_base_model_from_candidate(
                candidate
            )
        )

        for item in raw_models:

            if not isinstance(item, dict):
                continue

            llmfit_name = (
                self._get_llmfit_name(item)
            )

            # Exact model/name match.
            if (
                llmfit_name
                and llmfit_name in candidate_ids
            ):
                return item

            # Base-model match.
            base_models = (
                self._llmfit_base_models(item)
            )

            if candidate_base and candidate_base in base_models:
                return item

            # Normalized suffix/name matching.
            if llmfit_name:

                for candidate_id in candidate_ids:

                    if (
                        candidate_id.endswith(
                            "/" + llmfit_name
                        )
                        or llmfit_name.endswith(
                            "/" + candidate_id
                        )
                    ):
                        return item

        return None

    # ============================================================
    # SAFE CANDIDATE FIELD SETTER
    # ============================================================

    @staticmethod
    def _safe_set(
        candidate: ModelCandidateV2,
        field: str,
        value: Any,
    ) -> None:

        try:

            setattr(
                candidate,
                field,
                value,
            )

        except Exception:

            # Some ModelCandidateV2 implementations use strict
            # Pydantic fields. Unknown LLMFit fields are therefore
            # preserved in metadata instead of causing evaluation
            # to fail.
            pass

    # ============================================================
    # APPLY LLMFIT EVALUATION
    # ============================================================

    def _apply_evaluation(
        self,
        candidate: ModelCandidateV2,
        match: dict[str, Any] | None,
    ) -> None:

        # --------------------------------------------------------
        # NO LLMFIT MATCH
        #
        # Do not leave the candidate looking like it was evaluated.
        # Preserve the candidate, but explicitly record that LLMFit
        # has no resource evidence for it.
        # --------------------------------------------------------

        if not match:

            metadata = getattr(
                candidate,
                "metadata",
                None,
            )

            if not isinstance(metadata, dict):
                metadata = {}

            metadata["llmfit"] = {
                "matched": False,
                "evaluation_status": "not_evaluated",
            }

            self._safe_set(
                candidate,
                "metadata",
                metadata,
            )

            return

        # --------------------------------------------------------
        # MATCHED BY LLMFIT
        #
        # Existing behavior remains unchanged.
        # --------------------------------------------------------

        self._safe_set(
            candidate,
            "score",
            match.get("score"),
        )

        self._safe_set(
            candidate,
            "fit",
            match.get("fit_label"),
        )

        self._safe_set(
            candidate,
            "fit_label",
            match.get("fit_label"),
        )

        self._safe_set(
            candidate,
            "fit_level",
            match.get("fit_level"),
        )

        self._safe_set(
            candidate,
            "estimated_tps",
            match.get("estimated_tps"),
        )

        self._safe_set(
            candidate,
            "measured_tps",
            match.get("measured_tps"),
        )

        self._safe_set(
            candidate,
            "memory_required_gb",
            match.get("memory_required_gb"),
        )

        self._safe_set(
            candidate,
            "total_memory_gb",
            match.get("total_memory_gb"),
        )

        self._safe_set(
            candidate,
            "usable_context",
            match.get("usable_context"),
        )

        self._safe_set(
            candidate,
            "context_length",
            match.get("context_length"),
        )

        # --------------------------------------------------------
        # Runtime / execution
        # --------------------------------------------------------

        self._safe_set(
            candidate,
            "runtime",
            match.get("runtime"),
        )

        self._safe_set(
            candidate,
            "runtime_label",
            match.get("runtime_label"),
        )

        self._safe_set(
            candidate,
            "run_mode",
            match.get("run_mode"),
        )

        self._safe_set(
            candidate,
            "run_mode_label",
            match.get("run_mode_label"),
        )

        # --------------------------------------------------------
        # Model metadata
        # --------------------------------------------------------

        self._safe_set(
            candidate,
            "category",
            match.get("category"),
        )

        self._safe_set(
            candidate,
            "use_case",
            match.get("use_case"),
        )

        self._safe_set(
            candidate,
            "parameter_count",
            match.get("parameter_count"),
        )

        self._safe_set(
            candidate,
            "params_b",
            match.get("params_b"),
        )

        self._safe_set(
            candidate,
            "is_moe",
            match.get("is_moe"),
        )

        self._safe_set(
            candidate,
            "license",
            match.get("license"),
        )

        self._safe_set(
            candidate,
            "provider",
            match.get("provider"),
        )

        self._safe_set(
            candidate,
            "capabilities",
            match.get("capabilities"),
        )

        # --------------------------------------------------------
        # Scoring / hardware
        # --------------------------------------------------------

        self._safe_set(
            candidate,
            "utilization_pct",
            match.get("utilization_pct"),
        )

        self._safe_set(
            candidate,
            "score_components",
            match.get("score_components"),
        )

        self._safe_set(
            candidate,
            "notes",
            match.get("notes"),
        )

        # --------------------------------------------------------
        # COMPLETE LLMFIT RECORD
        # --------------------------------------------------------

        metadata = getattr(
            candidate,
            "metadata",
            None,
        )

        if not isinstance(metadata, dict):
            metadata = {}

        metadata["llmfit"] = {
            **match,
            "matched": True,
            "evaluation_status": "evaluated",
        }

        self._safe_set(
            candidate,
            "metadata",
            metadata,
        )

