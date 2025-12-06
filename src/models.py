"""
MAJ (Memory Assisted Judge) Node Models

Core nodes and relationships:
- Attempt → SATISFIES → Policy
- Attempt → CAUSES → Issue
- Fix → RESOLVES → Issue
"""

import os
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
import uuid

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str) -> list[float]:
    """Get embedding from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


class Policy(BaseModel):
    """The task/requirement that attempts try to satisfy."""
    description: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[list[float]] = None

    def with_embedding(self) -> "Policy":
        self.embedding = get_embedding(self.description)
        return self

    def to_neo4j_props(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "embedding": self.embedding
        }


class Attempt(BaseModel):
    """An approach/attempt to solve a task."""
    description: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[list[float]] = None

    def with_embedding(self) -> "Attempt":
        self.embedding = get_embedding(self.description)
        return self

    def to_neo4j_props(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "embedding": self.embedding
        }


class Issue(BaseModel):
    """A problem found in an attempt."""
    description: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[list[float]] = None

    def with_embedding(self) -> "Issue":
        self.embedding = get_embedding(self.description)
        return self

    def to_neo4j_props(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "embedding": self.embedding
        }


class Fix(BaseModel):
    """A solution that resolves an issue."""
    description: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[list[float]] = None

    def with_embedding(self) -> "Fix":
        self.embedding = get_embedding(self.description)
        return self

    def to_neo4j_props(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "embedding": self.embedding
        }


# --- LLM Extraction Schema ---

class IssueFix(BaseModel):
    """An issue-fix pair."""
    issue: str
    fix: str


class JudgeResult(BaseModel):
    """Schema for LLM judge output."""
    attempt: str
    issue_fix_pairs: list[IssueFix]
