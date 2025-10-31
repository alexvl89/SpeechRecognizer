import json
import logging
from pathlib import Path

logger = logging.getLogger("bot")


class UserManager:
    def __init__(self, file_path: str = "allowed_users.json", admin_id: int = None):
        self.file_path = Path(file_path)
        self.admin_id = admin_id
        self.allowed_users = self._load_users()

    def _load_users(self):
        if not self.file_path.exists():
            return set()
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data)
        except Exception as e:
            logger.error(f"Ошибка загрузки списка пользователей: {e}")
            return set()

    def _save_users(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(list(self.allowed_users), f,
                          ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения списка пользователей: {e}")

    def is_allowed(self, user_id: int) -> bool:
        return user_id in self.allowed_users

    def add_user(self, user_id: int):
        self.allowed_users.add(user_id)
        self._save_users()
        logger.info(f"Добавлен новый пользователь: {user_id}")

    def list_users(self):
        return list(self.allowed_users)
