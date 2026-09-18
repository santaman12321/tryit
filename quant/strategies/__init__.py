from .base import Signals, Strategy
from .breakout import BreakoutStrategy
from .formula import FormulaStrategy
from .gap_day import GapDayStrategy
from .lowturn import HighBreakoutStrategy, MLRankStrategy, PEADStrategy, XSMomentumStrategy
from .pullback import PullbackStrategy
from .surge import SurgeChaseStrategy, SurgePullbackStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    PullbackStrategy.name: PullbackStrategy,
    BreakoutStrategy.name: BreakoutStrategy,
    GapDayStrategy.name: GapDayStrategy,
    SurgeChaseStrategy.name: SurgeChaseStrategy,
    SurgePullbackStrategy.name: SurgePullbackStrategy,
    FormulaStrategy.name: FormulaStrategy,
    XSMomentumStrategy.name: XSMomentumStrategy,
    PEADStrategy.name: PEADStrategy,
    HighBreakoutStrategy.name: HighBreakoutStrategy,
    MLRankStrategy.name: MLRankStrategy,
}


def make_strategy(name: str, **params) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
    return cls(**params)


__all__ = ["Signals", "Strategy", "STRATEGIES", "make_strategy", "PullbackStrategy", "BreakoutStrategy", "GapDayStrategy", "SurgeChaseStrategy", "SurgePullbackStrategy"]
