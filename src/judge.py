"""
MAJ Judge - Evaluates agent outputs and extracts issues/fixes.

Input: task (policy) + agent_output
Output: {policy, attempt (with feedback), issue_fix_pairs}
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from models import Policy, Attempt, Issue, Fix, JudgeResult

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_PROMPT = """You are an AI Judge evaluating an agent's output for a given task.

TASK: {task}

AGENT OUTPUT:
{agent_output}

Evaluate and return:
1. attempt: summary of the approach taken
2. is_successful: true if the output correctly solves the task, false otherwise
3. reasoning: explanation of why the attempt succeeded or failed
4. issue_fix_pairs: list of {{issue, fix}} pairs (empty if successful)

Be strict but fair in your evaluation."""


def judge(task: str, agent_output: str) -> dict:
    """
    Judge an agent's output for a given task.

    Returns dict with policy, attempt (with feedback), issues, fixes, and relationships.
    """
    prompt = JUDGE_PROMPT.format(task=task, agent_output=agent_output)

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "You are an expert AI Judge."},
            {"role": "user", "content": prompt}
        ],
        text_format=JudgeResult,
    )

    data = response.output_parsed

    # Task becomes the Policy
    policy = Policy(description=task).with_embedding()

    # Attempt with feedback
    attempt = Attempt(
        description=data.attempt,
        is_successful=data.is_successful,
        reasoning=data.reasoning
    ).with_embedding()

    issues = []
    fixes = []
    relationships = []

    # Attempt SATISFIES Policy
    relationships.append({"type": "SATISFIES", "from_id": attempt.id, "to_id": policy.id})

    for pair in data.issue_fix_pairs:
        issue = Issue(description=pair.issue).with_embedding()
        fix = Fix(description=pair.fix).with_embedding()

        issues.append(issue)
        fixes.append(fix)

        relationships.append({"type": "CAUSES", "from_id": attempt.id, "to_id": issue.id})
        relationships.append({"type": "RESOLVES", "from_id": fix.id, "to_id": issue.id})

    return {
        "policy": policy,
        "attempt": attempt,
        "issues": issues,
        "fixes": fixes,
        "relationships": relationships
    }
