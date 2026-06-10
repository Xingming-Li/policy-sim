from pydantic import BaseModel
from typing import Literal
from datetime import datetime


class Message(BaseModel):
    sender: str

    message_type: Literal[
        "proposal",
        "support",
        "oppose",
        "question",
        "evidence",
        "commitment"
    ]

    policy_position: float

    content: str

    timestamp: datetime