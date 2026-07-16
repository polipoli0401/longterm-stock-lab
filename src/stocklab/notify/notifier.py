"""Notification module (Discord / LINE).

Destination URLs and tokens are read exclusively from environment
variables (GitHub Secrets on Actions) and never appear in code, config
files, or logs. Channels without configuration are skipped automatically,
so the system also works with notifications disabled.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import requests

from stocklab.config import get_secret
from stocklab.logger import get_logger, log_event

logger = get_logger(__name__)

DISCORD_WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
LINE_TOKEN_ENV = "LINE_CHANNEL_ACCESS_TOKEN"
LINE_TO_ENV = "LINE_TO"


class Notifier(Protocol):
    """Common interface for notification channels."""

    name: str

    def send(self, text: str) -> bool:
        """Send text and return success/failure."""
        ...


class DiscordNotifier:
    """Discord webhook notifications."""

    name = "discord"
    MAX_CHUNK = 1900  # margin below Discord's 2000-character limit

    def __init__(self, webhook_url: str) -> None:
        """Args:
        webhook_url: Webhook URL (secret; never log it).
        """
        self._url = webhook_url

    def send(self, text: str) -> bool:
        """Send the message in chunks."""
        try:
            for chunk in _chunks(text, self.MAX_CHUNK):
                response = requests.post(self._url, json={"content": chunk}, timeout=15)
                response.raise_for_status()
            return True
        except requests.RequestException:
            logger.exception("Discord notification failed")
            return False


class LineNotifier:
    """LINE Messaging API (push) notifications."""

    name = "line"
    ENDPOINT = "https://api.line.me/v2/bot/message/push"
    MAX_LEN = 4900

    def __init__(self, token: str, to: str) -> None:
        """Args:
        token: Channel access token (secret).
        to: Destination user/group ID (secret).
        """
        self._token = token
        self._to = to

    def send(self, text: str) -> bool:
        """Send the message (text beyond the limit is truncated)."""
        try:
            payload = {
                "to": self._to,
                "messages": [{"type": "text", "text": text[: self.MAX_LEN]}],
            }
            response = requests.post(
                self.ENDPOINT,
                headers={"Authorization": f"Bearer {self._token}"},
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            logger.exception("LINE notification failed")
            return False


def build_notifiers() -> list[Notifier]:
    """Construct the available channels from environment variables."""
    notifiers: list[Notifier] = []
    webhook = get_secret(DISCORD_WEBHOOK_ENV)
    if webhook:
        notifiers.append(DiscordNotifier(webhook))
    token = get_secret(LINE_TOKEN_ENV)
    to = get_secret(LINE_TO_ENV)
    if token and to:
        notifiers.append(LineNotifier(token, to))
    log_event(
        logger,
        "Notification channels configured",
        channels=[n.name for n in notifiers] or ["(none)"],
    )
    return notifiers


def notify_all(text: str) -> None:
    """Send to all configured channels (no-op when none are configured)."""
    for notifier in build_notifiers():
        success = notifier.send(text)
        log_event(logger, "Notification sent", channel=notifier.name, success=success)


def _chunks(text: str, size: int) -> Iterator[str]:
    """Split text into fixed-size chunks."""
    for start in range(0, len(text), size):
        yield text[start : start + size]
