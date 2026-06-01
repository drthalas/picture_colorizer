from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class TelegramUserInfo:
    user_id: int
    full_name: str
    username: str | None = None

    def display_name(self) -> str:
        username_part = f" @{self.username}" if self.username else ""
        return f"{self.full_name}{username_part} (id: {self.user_id})"


class AccessStore:
    def __init__(self, path: Path, admin_user_id: int | None) -> None:
        self.path = path
        self.admin_user_id = admin_user_id

    def is_admin(self, user_id: int | None) -> bool:
        return user_id is not None and self.admin_user_id is not None and user_id == self.admin_user_id

    def is_allowed(self, user_id: int | None) -> bool:
        if user_id is None:
            return False
        return self.is_admin(user_id) or user_id in self.load_allowed_user_ids()

    def approve(self, user_id: int) -> None:
        allowed_user_ids = self.load_allowed_user_ids()
        allowed_user_ids.add(user_id)
        self.save_allowed_user_ids(allowed_user_ids)

    def load_allowed_user_ids(self) -> set[int]:
        if not self.path.exists():
            return set()

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()

        values = data.get("allowed_user_ids", [])
        return {int(value) for value in values if str(value).strip().lstrip("-").isdigit()}

    def save_allowed_user_ids(self, allowed_user_ids: set[int]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"allowed_user_ids": sorted(allowed_user_ids)}
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
