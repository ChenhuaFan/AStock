from __future__ import annotations

from urllib.parse import quote

import requests


def send_bark(bark_url: str, title: str, body: str, timeout: float = 20.0) -> dict[str, object]:
    if not bark_url:
        return {"sent": False, "reason": "empty_bark_url"}

    if "{title}" in bark_url or "{body}" in bark_url:
        url = bark_url.format(title=quote(title, safe=""), body=quote(body, safe=""))
    else:
        url = f"{bark_url.rstrip('/')}/{quote(title, safe='')}/{quote(body, safe='')}"

    response = requests.get(url, timeout=timeout)
    payload: object
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    return {
        "sent": True,
        "status_code": response.status_code,
        "ok": response.ok,
        "response": payload,
    }
