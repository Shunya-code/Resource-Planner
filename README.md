# Resource Planner

Workload-first model and execution planning.

The system answers:

> Given a task, workload, environment, and constraints,
> what execution plans are viable?

Initial architecture:

Task + Workload + Environment + Constraints
                    |
                    v
                 Planner
                /       \
               v         v
       Hugging Face     llmfit
               \         /
                v       v
             Plans

## Status

Initial repository structure.
