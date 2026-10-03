from typing import Any

from ..schemas import (
    Environment,
    Evidence,
    ExecutionConfiguration,
    FitResult,
    ModelCandidateV2,
    PlanningRequest,
)


class WorkloadEvaluator:
    """
    Evaluates resource-fit evidence against a concrete workload.

    This class does not benchmark models and does not invent
    performance characteristics. It interprets evidence supplied
    by resource-analysis providers such as llmfit.
    """

    def evaluate(
        self,
        request: PlanningRequest,
        model: ModelCandidateV2,
        fit: FitResult,
    ) -> dict[str, Any]:

        workload = request.workload
        environment = request.environment
        constraints = request.constraints

        warnings: list[str] = []
        requirements: list[str] = []
        evidence: list[Evidence] = list(fit.evidence)

        compatible = True
        uncertain = False

        # -----------------------------------------------------
        # Context
        # -----------------------------------------------------

        requested_context = workload.context_length

        usable_context = fit.metadata.get(
            "usable_context"
        )

        effective_context = fit.metadata.get(
            "effective_context_length"
        )

        if requested_context is not None:

            requirements.append(
                f"context_length={requested_context}"
            )

            if (
                usable_context is not None
                and requested_context > usable_context
            ):
                compatible = False

                warnings.append(
                    "Requested context exceeds the "
                    "provider-reported usable context."
                )

            elif (
                effective_context is not None
                and requested_context > effective_context
            ):
                uncertain = True

                warnings.append(
                    "Provider estimated performance using "
                    f"an effective context of {effective_context} "
                    f"rather than the requested {requested_context}."
                )

        # -----------------------------------------------------
        # RAM
        # -----------------------------------------------------

        required_ram = fit.ram_required_gb
        available_ram = environment.ram_gb

        if required_ram is not None:
            requirements.append(
                f"ram_required_gb={required_ram}"
            )

        if (
            required_ram is not None
            and available_ram is not None
            and required_ram > available_ram
        ):
            if constraints.allow_cpu_offload:
                uncertain = True

                warnings.append(
                    "Reported memory requirement exceeds "
                    "available RAM; CPU/offload behavior must "
                    "be verified."
                )
            else:
                compatible = False

                warnings.append(
                    "Reported memory requirement exceeds "
                    "available RAM."
                )

        # -----------------------------------------------------
        # GPU
        # -----------------------------------------------------

        if environment.gpu_count == 0:

            if fit.runtime:
                requirements.append(
                    f"runtime={fit.runtime}"
                )

            if fit.cpu_offload:
                warnings.append(
                    "Provider indicates CPU-only execution."
                )

        # -----------------------------------------------------
        # Concurrency
        # -----------------------------------------------------

        concurrency = workload.concurrency

        if concurrency and concurrency > 1:

            requirements.append(
                f"concurrency={concurrency}"
            )

            # llmfit's estimated_tps is generally a single
            # execution estimate. It must NOT automatically be
            # multiplied by concurrency.
            if fit.estimated_tokens_per_second is not None:

                uncertain = True

                warnings.append(
                    "Throughput estimate is not a validated "
                    f"concurrency={concurrency} measurement."
                )

        # -----------------------------------------------------
        # Output tokens
        # -----------------------------------------------------

        if workload.output_tokens is not None:

            requirements.append(
                f"output_tokens={workload.output_tokens}"
            )

        # -----------------------------------------------------
        # Runtime constraint
        # -----------------------------------------------------

        if constraints.required_runtime:

            requirements.append(
                f"required_runtime="
                f"{constraints.required_runtime}"
            )

            if (
                fit.runtime
                and fit.runtime.lower()
                != constraints.required_runtime.lower()
            ):
                compatible = False

                warnings.append(
                    "Provider runtime does not satisfy the "
                    "required runtime constraint."
                )

        # -----------------------------------------------------
        # License constraint
        # -----------------------------------------------------

        if constraints.required_license:

            requirements.append(
                f"required_license="
                f"{constraints.required_license}"
            )

            if (
                model.license
                and model.license.lower()
                != constraints.required_license.lower()
            ):
                compatible = False

                warnings.append(
                    "Model license does not match the "
                    "required license constraint."
                )

        # -----------------------------------------------------
        # Determine status
        # -----------------------------------------------------

        if not compatible:
            status = "incompatible"

        elif uncertain:
            status = "conditional"

        else:
            status = "compatible"

        configuration = ExecutionConfiguration(
            runtime=fit.runtime,

            quantization=fit.quantization,

            precision=(
                fit.metadata.get("precision")
            ),

            gpu_count=environment.gpu_count,

            cpu_offload=fit.cpu_offload,

            context_length=requested_context,
        )

        return {
            "status": status,

            "model_id": model.model_id,

            "artifact_id": fit.artifact_id,

            "configuration": configuration,

            "fit": fit,

            "requirements": requirements,

            "warnings": warnings,

            "evidence": evidence,

            "estimated_tokens_per_second": (
                fit.estimated_tokens_per_second
            ),

            "estimated_latency_ms": (
                fit.estimated_latency_ms
            ),
        }
