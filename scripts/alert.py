#!/usr/bin/env python3
"""Send Slack notifications for monitoring run results.

Reads a MonitorResult JSON payload and sends a formatted Slack message
for every completed monitoring run (pass or degradation).

Exit codes:
    0: Success (notification sent)
    1: Error sending notification
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def format_slack_message(
    monitor_result: dict[str, Any],
    pipeline_run_url: str | None = None,
    eval_engine: str | None = None,
) -> dict[str, Any]:
    """Format a MonitorResult as a Slack Block Kit message.

    Args:
        monitor_result: Parsed JSON from monitor.py output.
        pipeline_run_url: Optional URL to the pipeline run in OpenShift console.
        eval_engine: Evaluation engine used (harbor, ase, a2a).

    Returns:
        Slack message payload with blocks.
    """
    submission = monitor_result.get("submission_name", "unknown")
    current = monitor_result.get("current_score")
    previous = monitor_result.get("previous_score")
    ratio = monitor_result.get("ratio")
    threshold = monitor_result.get("threshold", 0.85)
    message = monitor_result.get("message", "")
    current_run_id = monitor_result.get("current_run_id") or "N/A"
    degraded = monitor_result.get("degraded", False)

    current_pct = f"{current:.1%}" if current is not None else "N/A"
    previous_pct = f"{previous:.1%}" if previous is not None else "N/A"
    ratio_str = f"{ratio:.2f}" if ratio is not None else "N/A"
    engine_label = (eval_engine or "unknown").upper()

    if degraded:
        header_text = f"🚨 [{engine_label}] Performance Degradation Detected"
    else:
        header_text = f"✅ [{engine_label}] Monitoring Pass"

    run_id_text = (
        f"<{pipeline_run_url}|{current_run_id}>" if pipeline_run_url and current_run_id != "N/A" else current_run_id
    )

    fields = [
        {"type": "mrkdwn", "text": f"*Skill:*\n{submission}"},
        {"type": "mrkdwn", "text": f"*Run ID:*\n{run_id_text}"},
        {"type": "mrkdwn", "text": f"*Score:*\n{current_pct}"},
    ]

    if previous is not None:
        fields += [
            {"type": "mrkdwn", "text": f"*Baseline:*\n{previous_pct}"},
            {"type": "mrkdwn", "text": f"*Ratio:*\n{ratio_str}"},
            {"type": "mrkdwn", "text": f"*Threshold:*\n{threshold}"},
        ]

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": header_text,
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": fields,
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": message},
            ],
        },
    ]

    if pipeline_run_url:
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Pipeline Run"},
                        "url": pipeline_run_url,
                    }
                ],
            }
        )

    return {"blocks": blocks}


def send_slack_notification(
    webhook_url: str,
    payload: dict[str, Any],
) -> bool:
    """Send a message to Slack via webhook.

    Args:
        webhook_url: Slack Incoming Webhook URL.
        payload: Message payload (blocks, text, etc.).

    Returns:
        True if successful, False otherwise.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                logger.info("Slack notification sent successfully")
                return True
            else:
                logger.error("Slack returned status %d", response.status)
                return False
    except urllib.error.HTTPError as e:
        logger.error("HTTP error sending to Slack: %s", e)
        return False
    except urllib.error.URLError as e:
        logger.error("URL error sending to Slack: %s", e)
        return False


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Send Slack notification for monitoring run results",
    )
    parser.add_argument(
        "--payload",
        required=True,
        help="Path to MonitorResult JSON file (or '-' for stdin)",
    )
    parser.add_argument(
        "--webhook-url",
        required=True,
        help="Slack Incoming Webhook URL",
    )
    parser.add_argument(
        "--pipeline-run-url",
        default=None,
        help="Optional URL to the pipeline run in OpenShift console",
    )
    parser.add_argument(
        "--eval-engine",
        default=None,
        help="Evaluation engine used (harbor, ase, a2a)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message payload without sending",
    )

    args = parser.parse_args()

    try:
        if args.payload == "-":
            payload_data = json.load(sys.stdin)
        else:
            with open(args.payload) as f:
                payload_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error("Failed to read payload: %s", e)
        return 1

    slack_message = format_slack_message(
        payload_data,
        pipeline_run_url=args.pipeline_run_url,
        eval_engine=args.eval_engine,
    )

    if args.dry_run:
        print(json.dumps(slack_message, indent=2))
        return 0

    if send_slack_notification(args.webhook_url, slack_message):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
