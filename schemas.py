from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    Schema for a source used by the agent to fetch data.
    """

    url: str = Field(..., description="The URL of the source")


class AgentResponse(BaseModel):
    """
    Schema for a response from the agent. This is the response that the agent will return with answer and sources.
    """

    answer: str = Field(..., description="The answer from the agent")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used by the agent to answer the question",
    )
