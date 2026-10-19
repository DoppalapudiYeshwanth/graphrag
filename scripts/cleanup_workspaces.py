"""Delete idle visitor workspaces (uploaded documents and everything derived from them).

A workspace is idle when its newest document is older than WORKSPACE_TTL_DAYS. The public demo
corpus is never touched. Run it on a schedule (see DEPLOYMENT.md), or by hand:

    python -m scripts.cleanup_workspaces --dry-run
    python -m scripts.cleanup_workspaces
"""

from __future__ import annotations

import argparse
import logging

from graphrag.config import get_settings, setup_logging
from graphrag.graph import GraphStore

logger = logging.getLogger(__name__)


def main(days: float, dry_run: bool) -> int:
    with GraphStore() as store:
        stale = store.stale_workspaces(days)
        logger.info("%d workspace(s) idle for more than %s days", len(stale), days)
        for ws in stale:
            if dry_run:
                logger.info("would delete workspace %s...", ws[:8])
            else:
                store.delete_workspace(ws)
                logger.info("deleted workspace %s...", ws[:8])
    return len(stale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--days", type=float, default=None, help="Idle days (default: WORKSPACE_TTL_DAYS).")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be deleted.")
    args = parser.parse_args()
    setup_logging()
    main(args.days or get_settings().workspace_ttl_days, args.dry_run)
