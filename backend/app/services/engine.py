"""The analysis engine and its job runner.

Building the engine (dataset ingestion, KYC index, anomaly model, knowledge base, LLM) takes tens of
seconds, so it happens in a background thread and the API reports ``ready`` when done. Cases are
processed one at a time by a single worker thread (one local LLM, one GPU).
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from agents.context import AgentContext
from agents.coordinator.workflow import CaseWorkflow

logger = logging.getLogger(__name__)


class EngineService:
    def __init__(self) -> None:
        self.workflow: CaseWorkflow | None = None
        self.context: AgentContext | None = None
        self.error: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="case-worker")
        self._lock = threading.Lock()

    @classmethod
    def from_context(cls, ctx: AgentContext) -> "EngineService":
        engine = cls()
        engine.context, engine.workflow = ctx, CaseWorkflow(ctx)
        return engine

    @property
    def ready(self) -> bool:
        return self.workflow is not None

    def start_loading(self, builder: Callable[[], AgentContext]) -> None:
        def load() -> None:
            try:
                ctx = builder()
                self.context, self.workflow = ctx, CaseWorkflow(ctx)
                logger.info("analysis engine ready")
            except Exception as exc:  # noqa: BLE001 - report, do not crash the API
                self.error = f"{type(exc).__name__}: {exc}"
                logger.exception("analysis engine failed to load")

        threading.Thread(target=load, name="engine-loader", daemon=True).start()

    def submit(self, job: Callable[[], None]) -> Future[None]:
        return self._executor.submit(job)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
