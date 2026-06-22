"""Trivial health task to prove the Hatchet plane end-to-end.

Enqueue it from the dashboard or via the API facade and confirm it runs on the
extraction-worker. Defined as a TaskSpec (built in the registry) so client
construction stays centralized; carries no business logic.
"""

from pydantic import BaseModel
from hatchet_sdk import Context

from src.tasks.base import TaskSpec
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class PingInput(BaseModel):
    message: str = "ping"


class PingOutput(BaseModel):
    reply: str
    worker: str = "extraction-worker"


async def _handle(input: PingInput, ctx: Context) -> PingOutput:
    logger.info(f"ping task received: {input.message!r}")
    return PingOutput(reply=f"pong: {input.message}")


PING_SPEC = TaskSpec(
    name="ping",
    input_model=PingInput,
    output_model=PingOutput,
    handler=_handle,
)
