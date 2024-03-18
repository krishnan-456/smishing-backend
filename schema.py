from pydantic import BaseModel

class Schema(BaseModel):
    message: str
    model: str

