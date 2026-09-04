from pydantic import BaseModel
from typing import Literal
from datetime import datetime


class Message(BaseModel):

    sender: str

    target: str | None = None

    message_type: Literal[
        "proposal",
        "support",
        "oppose",
        "evidence",
        "question",
        "commitment"
    ]

    policy_position: float

    content: str

    timestamp: datetime
    