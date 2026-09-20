from datetime import datetime

from backend.app.schemas.domain import Discrepancy
from reconciliation.config import ReconciliationConfig
from reconciliation.matcher import MatchedRecords
from reconciliation.rules import RULES, Rule


class DiscrepancyDetector:
    """Applies every rule to a matched pair; output order is deterministic (by rule id)."""

    def __init__(
        self, config: ReconciliationConfig | None = None, rules: tuple[Rule, ...] = RULES
    ) -> None:
        self.config = config or ReconciliationConfig()
        self.rules = rules

    def detect(self, matched: MatchedRecords, as_of: datetime | None = None) -> list[Discrepancy]:
        found = (rule.check(matched, self.config, as_of) for rule in self.rules)
        return sorted((d for d in found if d is not None), key=lambda d: d.rule_id)
