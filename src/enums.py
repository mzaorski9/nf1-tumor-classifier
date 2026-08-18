
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "Low tumor risk"
    MED = "Medium tumor risk"
    HIGH = "High tumor risk!"


