from dataclasses import dataclass
from typing import List


@dataclass
class LabelSpec:
    id: int
    name: str
    color: str
    is_background: bool = False


@dataclass
class HeadSpec:
    name: str
    display_name: str
    type: str
    labels: List[LabelSpec]