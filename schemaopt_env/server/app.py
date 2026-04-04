"""FastAPI app entrypoint for the schema optimization environment."""

from fastapi import APIRouter

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import SchemaOptAction, SchemaOptObservation
    from ..tasks import list_spider_databases
    from .schemaopt_environment import SchemaOptEnvironment
except ImportError:
    try:
        from openenv.core.env_server.http_server import create_app
    except ImportError:
        from openenv_core.env_server.http_server import create_app

    from models import SchemaOptAction, SchemaOptObservation
    from tasks import list_spider_databases
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
    spider_dbs = list_spider_databases()
    return {
        "tasks": spider_dbs,
        "spider_databases": spider_dbs,
        "spider_database_count": len(spider_dbs),
        "action_schema": SchemaOptAction.model_json_schema(),
    }


@router.get("/grader")
def grader_result():
    return SchemaOptEnvironment.latest_report()


@router.post("/baseline")
def run_baseline():
    return SchemaOptEnvironment.run_baseline()


app.include_router(router)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
