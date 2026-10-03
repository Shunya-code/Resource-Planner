from typing import Any

from ..adapters.huggingface import HuggingFaceProvider
from ..adapters.llmfit import LlmfitProvider
from ..schemas import (
    PlanningRequest,
    FitResult,
    Environment,
)
from .workload_evaluator import WorkloadEvaluator


class ResourcePlanner:
    """
    Coordinates model discovery, LLMFit resource analysis,
    and workload compatibility analysis.
    """

    def __init__(
        self,
        huggingface: HuggingFaceProvider | None = None,
        llmfit: LlmfitProvider | None = None,
        evaluator: WorkloadEvaluator | None = None,
    ):
        self.huggingface = (
            huggingface
            or HuggingFaceProvider()
        )

        self.llmfit = (
            llmfit
            or LlmfitProvider()
        )

        self.evaluator = (
            evaluator
            or WorkloadEvaluator()
        )

    async def discover(
        self,
        request: PlanningRequest,
        limit: int = 20,
    ):
        """
        Discover candidate models from Hugging Face.
        """

        return await self.huggingface.search(
            request=request,
            limit=limit,
        )

    async def analyze(
        self,
        request: PlanningRequest,
        candidates: list,
    ) -> list[dict[str, Any]]:
        """
        Evaluate discovered ModelCandidateV2 objects with
        LLMFit and then run workload compatibility analysis.

        IMPORTANT:
        - candidates are Pydantic ModelCandidateV2 objects
        - LLMFit stores its complete result in candidate.metadata
        - FitResult.fit uses the project's vocabulary
        """

        environment = request.environment

        # --------------------------------------------------------
        # Resolve live environment from LLMFit when RAM is not
        # explicitly supplied.
        # --------------------------------------------------------

        if environment.ram_gb is None:

            system_data = await self.llmfit.system()

            if isinstance(system_data, dict):
                system = system_data.get(
                    "system",
                    system_data,
                )
            else:
                system = system_data

            if not isinstance(system, dict):
                raise TypeError(
                    "LLMFit system() must return a dict "
                    "containing system information."
                )

            environment = Environment(
                type="local",

                cpu_cores=system.get(
                    "cpu_cores"
                ),

                ram_gb=system.get(
                    "available_ram_gb"
                ),

                gpu_count=(
                    system.get("gpu_count")
                    or 0
                ),

                gpu_memory_gb=system.get(
                    "gpu_available_gb"
                ),

                gpu_name=system.get(
                    "gpu_name"
                ),

                accelerator=None,
                storage_gb=None,
                network_bandwidth_mbps=None,
            )

            request = request.model_copy(
                update={
                    "environment": environment
                }
            )

        # --------------------------------------------------------
        # LLMFit evaluation.
        #
        # LLMFit returns ModelCandidateV2 objects.
        # --------------------------------------------------------

        evaluated_candidates = await self.llmfit.evaluate(
            request=request,
            candidates=candidates,
            environment=environment,
        )

        results: list[dict[str, Any]] = []

        # --------------------------------------------------------
        # Convert each evaluated candidate into FitResult.
        # --------------------------------------------------------

        for candidate in evaluated_candidates:

            metadata = getattr(
                candidate,
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            llmfit_data = metadata.get(
                "llmfit",
                {},
            )

            if not isinstance(llmfit_data, dict):
                llmfit_data = {}

            # ----------------------------------------------------
            # Normalize LLMFit fit vocabulary.
            #
            # LLMFit may report:
            #
            #   good
            #   marginal
            #   unlikely
            #   unknown
            #
            # FitResult accepts:
            #
            #   good
            #   possible
            #   unlikely
            #   unknown
            # ----------------------------------------------------

            raw_fit = (
                llmfit_data.get("fit_level")
                or llmfit_data.get("fit_label")
                or "unknown"
            )

            normalized_fit = str(
                raw_fit
            ).strip().lower()

            if normalized_fit in (
                "good",
                "excellent",
            ):
                fit_value = "good"

            elif normalized_fit in (
                "marginal",
                "possible",
            ):
                fit_value = "possible"

            elif normalized_fit in (
                "unlikely",
                "poor",
                "bad",
            ):
                fit_value = "unlikely"

            else:
                fit_value = "unknown"

            # ----------------------------------------------------
            # Construct canonical FitResult.
            # ----------------------------------------------------

            fit = FitResult(
                artifact_id=(
                    llmfit_data.get("name")
                    or candidate.model_id
                ),

                fit=fit_value,

                vram_required_gb=(
                    llmfit_data.get(
                        "gpu_vram_gb"
                    )
                ),

                ram_required_gb=(
                    llmfit_data.get(
                        "memory_required_gb"
                    )
                ),

                storage_required_gb=(
                    llmfit_data.get(
                        "disk_size_gb"
                    )
                ),

                estimated_tokens_per_second=(
                    llmfit_data.get(
                        "estimated_tps"
                    )
                ),

                estimated_latency_ms=(
                    llmfit_data.get(
                        "ttft_ms"
                    )
                ),

                runtime=(
                    llmfit_data.get(
                        "runtime"
                    )
                ),

                quantization=(
                    llmfit_data.get(
                        "best_quant"
                    )
                ),

                cpu_offload=(
                    llmfit_data.get(
                        "run_mode"
                    ) == "cpu_only"
                ),

                warnings=(
                    llmfit_data.get(
                        "notes"
                    )
                    or []
                ),

                metadata={
                    **llmfit_data,
                },
            )

            # ----------------------------------------------------
            # Workload compatibility evaluation.
            # ----------------------------------------------------

            evaluation = self.evaluator.evaluate(
                request=request,
                model=candidate,
                fit=fit,
            )

            results.append(
                evaluation
            )

        return results

    async def plan(
        self,
        request: PlanningRequest,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Run the complete discovery -> LLMFit -> workload
        evaluation pipeline.
        """

        candidates = await self.discover(
            request=request,
            limit=limit,
        )

        evaluations = await self.analyze(
            request=request,
            candidates=candidates,
        )

        return {
            "request": request,

            "candidate_count": len(
                candidates
            ),

            "evaluated_count": len(
                evaluations
            ),

            "results": evaluations,
        }
