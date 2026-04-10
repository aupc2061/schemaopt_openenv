"""FastAPI app entrypoint for the schema optimization environment."""

import os
from typing import Optional

from fastapi import APIRouter

try:
    from openenv.core.env_server.http_server import create_app

    from models import SchemaOptAction, SchemaOptObservation
    from tasks import list_task_summaries
    from .schemaopt_environment import SchemaOptEnvironment
except ImportError:
    try:
        from openenv.core.env_server.http_server import create_app
    except ImportError:
        from openenv_core.env_server.http_server import create_app

    from models import SchemaOptAction, SchemaOptObservation
    from tasks import list_task_summaries
    from server.schemaopt_environment import SchemaOptEnvironment


app = create_app(
    SchemaOptEnvironment,
    SchemaOptAction,
    SchemaOptObservation,
    env_name="schemaopt_env",
)

router = APIRouter()


@router.get("/tasks")
def list_tasks():
    return {
        "tasks": list_task_summaries(),
        "action_schema": SchemaOptAction.model_json_schema(),
    }

@router.post("/baseline")
def run_baseline():
    return SchemaOptEnvironment.run_baseline()

@router.get("/grader")
@router.post("/grader")
def grader_result(task_id: Optional[str] = None):
    return SchemaOptEnvironment.latest_report(task_id=task_id)


app.include_router(router)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
