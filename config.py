from dataclasses import dataclass
from pathlib import Path

# Number of trading days per year (standard convention)
TRADING_DAYS_PER_YEAR = 252

# Annual risk-free rate used for Sharpe ratio calculation
# Set this to current treasury yield if needed
RISK_FREE_ANNUAL = 0.0

# NOTE:
# INITIAL_CASH has been removed.
# Initial investments should be handled via external_cashflows.csv


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    cache: Path
    outputs: Path
    charts: Path


def get_paths() -> Paths:
    """
    Create and return standardized project paths.

    Structure:
    - root/
        - data/
        - cache/
        - outputs/
            - charts/
    """
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    cache = root / "cache"
    outputs = root / "outputs"
    charts = outputs / "charts"

    # Ensure all directories exist
    for p in (data, cache, outputs, charts):
        p.mkdir(parents=True, exist_ok=True)

    return Paths(
        root=root,
        data=data,
        cache=cache,
        outputs=outputs,
        charts=charts,
    )