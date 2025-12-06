"""
Neo4j Graph Manager for MAJ

Handles CRUD operations and vector similarity search for:
- Attempt → CAUSES → Issue
- Fix → RESOLVES → Issue
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from models import Attempt, Issue, Fix

load_dotenv()

EMBEDDING_DIM = 1536  # text-embedding-3-small


class GraphManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        self._ensure_vector_indexes()

    def close(self):
        self.driver.close()

    def _ensure_vector_indexes(self):
        """Create vector indexes for similarity search."""
        with self.driver.session() as session:
            # Check and create indexes for each node type
            for label in ["Attempt", "Issue", "Fix"]:
                session.run(f"""
                    CREATE VECTOR INDEX {label.lower()}_embedding IF NOT EXISTS
                    FOR (n:{label})
                    ON (n.embedding)
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {EMBEDDING_DIM},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)

    # --- Create Nodes ---

    def create_attempt(self, attempt: Attempt) -> str:
        with self.driver.session() as session:
            session.run(
                "CREATE (a:Attempt {id: $id, description: $description, embedding: $embedding})",
                **attempt.to_neo4j_props()
            )
        return attempt.id

    def create_issue(self, issue: Issue) -> str:
        with self.driver.session() as session:
            session.run(
                "CREATE (i:Issue {id: $id, description: $description, embedding: $embedding})",
                **issue.to_neo4j_props()
            )
        return issue.id

    def create_fix(self, fix: Fix) -> str:
        with self.driver.session() as session:
            session.run(
                "CREATE (f:Fix {id: $id, description: $description, embedding: $embedding})",
                **fix.to_neo4j_props()
            )
        return fix.id

    # --- Create Relationships ---

    def link_attempt_causes_issue(self, attempt_id: str, issue_id: str):
        """Attempt → CAUSES → Issue"""
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Attempt {id: $attempt_id})
                MATCH (i:Issue {id: $issue_id})
                CREATE (a)-[:CAUSES]->(i)
            """, attempt_id=attempt_id, issue_id=issue_id)

    def link_fix_resolves_issue(self, fix_id: str, issue_id: str):
        """Fix → RESOLVES → Issue"""
        with self.driver.session() as session:
            session.run("""
                MATCH (f:Fix {id: $fix_id})
                MATCH (i:Issue {id: $issue_id})
                CREATE (f)-[:RESOLVES]->(i)
            """, fix_id=fix_id, issue_id=issue_id)

    # --- Vector Similarity Search ---

    def find_similar_attempts(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Find similar attempts using vector similarity."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('attempt_embedding', $top_k, $embedding)
                YIELD node, score
                RETURN node.id as id, node.description as description, score
            """, top_k=top_k, embedding=embedding)
            return [dict(r) for r in result]

    def find_similar_issues(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Find similar issues using vector similarity."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('issue_embedding', $top_k, $embedding)
                YIELD node, score
                RETURN node.id as id, node.description as description, score
            """, top_k=top_k, embedding=embedding)
            return [dict(r) for r in result]

    # --- Query ---

    def get_issues_for_attempt(self, attempt_id: str) -> list[dict]:
        """Get all issues caused by an attempt."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Attempt {id: $attempt_id})-[:CAUSES]->(i:Issue)
                RETURN i.id as id, i.description as description
            """, attempt_id=attempt_id)
            return [dict(r) for r in result]

    def get_fixes_for_issue(self, issue_id: str) -> list[dict]:
        """Get all fixes that resolve an issue."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:Fix)-[:RESOLVES]->(i:Issue {id: $issue_id})
                RETURN f.id as id, f.description as description
            """, issue_id=issue_id)
            return [dict(r) for r in result]

    def clear_all(self):
        """Clear all nodes and relationships (for testing)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
