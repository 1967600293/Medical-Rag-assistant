import os
import json
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from typing import Sequence
from log_config import default_logger as logger


def get_chat_history(session_id):
    return LocalChatHistory(session_id, "chat_history")


class LocalChatHistory(BaseChatMessageHistory):
    """基于本地 JSON 文件的对话历史存储"""

    def __init__(self, session_id: str, storage_path: str):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages) + list(messages)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([message_to_dict(m) for m in all_messages], f)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            # 首次对话或文件损坏，返回空列表
            logger.debug(f"会话 {self.session_id} 无有效历史记录")
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        logger.info(f"已清空会话 {self.session_id} 的历史记录")