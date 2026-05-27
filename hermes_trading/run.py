"""
run.py — entrypoint for the Hermes trading worker.
Reads asset from state/goal.yaml (override with --asset flag).
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml

from hermes_trading.loop import TradingLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    stream=sys.stdout,
)
log = logging.getLogger("hermes.run")

STATE_DIR = Path(__file__).parent.parent / "state"
GOAL_FILE = STATE_DIR / "goal.yaml"


def load_goal() -> dict:
    with open(GOAL_FILE) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Trading Worker")
    parser.add_argument("--asset", help="Override asset from goal.yaml (e.g. BTC/USDT)")
    parser.add_argument(
        "--mode",
        default=os.getenv("HERMES_TRADING_MODE", "paper"),
        choices=["paper", "live"],
        help="Trading mode (default: paper)",
    )
    args = parser.parse_args()

    goal = load_goal()
    asset = args.asset or goal["asset"]
    mode = args.mode

    if mode == "live":
        accept = os.getenv("HERMES_TRADING_I_ACCEPT_RISK", "false").lower()
        if accept != "true":
            log.error(
                "Live mode requires HERMES_TRADING_I_ACCEPT_RISK=true in .env. "
                "Refusing to start — set the flag explicitly to acknowledge real-money risk."
            )
            sys.exit(1)

    log.info("Booting hermes-trading worker | asset=%s mode=%s", asset, mode)

    loop = TradingLoop(asset=asset, mode=mode, state_dir=STATE_DIR, goal=goal)
    asyncio.run(loop.run_forever())


if __name__ == "__main__":
    main()
