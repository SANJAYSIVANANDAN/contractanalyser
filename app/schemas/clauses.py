from pydantic import BaseModel, ConfigDict


class ClauseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    clause_type: str
    found: bool
    title: str | None
    summary: str | None
    supporting_text: str | None
    risk_level: str | None
    explanation: str | None
    page_number: int | None
