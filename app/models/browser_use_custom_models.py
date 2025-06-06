from pydantic import BaseModel
from enum import Enum 

class RunningStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    ERROR = "error"

# custom agent's final answer structure here
class BasicAgentResponse(BaseModel):
    status: RunningStatus = RunningStatus.DONE 
    message: str
