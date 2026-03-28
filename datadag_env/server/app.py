"""FastAPI app entrypoint for DataDAG environment."""

from fastapi import APIRouter

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import DatadagAction, DatadagObservation
    from .datadag_environment import DatadagEnvironment
except ImportError:
    try:
        from openenv.core.env_server.http_server import create_app
    except ImportError:
        from openenv_core.env_server.http_server import create_app

    from models import DatadagAction, DatadagObservation
    from server.datadag_environment import DatadagEnvironment

app = create_app(
    DatadagEnvironment,
    DatadagAction,
    DatadagObservation,
    env_name="datadag_env",
)

router = APIRouter()


@router.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task1_easy",
                "difficulty": "easy",
                "objective": "Build a two-stage DAG to compute user lifetime revenue.",
                "required_sources": ["raw_transactions", "raw_users"],
                "target_model": "mart_user_lifetime_revenue",
            },
            {
                "id": "task2_medium",
                "difficulty": "medium",
                "objective": "Pipeline evolution without cascading failures.",
                "status": "planned",
            },
            {
                "id": "task3_hard",
                "difficulty": "hard",
                "objective": "Open-ended architecture and optimization.",
                "status": "planned",
            },
        ],
        "action_schema": DatadagAction.model_json_schema(),
    }


@router.get("/grader")
def grader_result():
    return DatadagEnvironment.latest_report()


@router.post("/baseline")
def run_baseline():
    return DatadagEnvironment.run_baseline()


app.include_router(router)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
