"""
MAJ Judge - Memory Assisted Judge for evaluating agent outputs.

Two modes:
1. judge() - Stateless evaluation (no memory)
2. judge_with_memory() - Uses past experiences for context
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from models import Policy, Attempt, Issue, Fix, JudgeResult, get_embedding
from prompts import build_judge_prompt, build_judge_with_memory_prompt, DEFAULT_GOAL

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _format_memory_context(contrastive: dict, similar_issues: list) -> str:
    """Format retrieved memory into prompt context."""
    parts = []

    # Filter by similarity threshold - only high-confidence matches
    # Positive examples: lower threshold (0.80) - good to learn patterns
    # Negative examples: higher threshold (0.90) - must be very similar to apply
    positive = [a for a in contrastive['positive'] if a.get('score', 0) >= 0.80]
    negative = [a for a in contrastive['negative'] if a.get('score', 0) >= 0.90]
    issues = [i for i in similar_issues if i.get('score', 0) >= 0.85]

    if positive:
        parts.append("SUCCESSFUL APPROACHES (similar code that passed):")
        for i, a in enumerate(positive, 1):
            score = a.get('score', 0)
            parts.append(f"  {i}. [similarity: {score:.0%}] Code: {a['agent_output'][:200]}...")
            parts.append(f"     Why it worked: {a['reasoning']}")

    if negative:
        parts.append("\nFAILED APPROACHES (similar code that failed - check if same issue applies):")
        for i, a in enumerate(negative, 1):
            score = a.get('score', 0)
            parts.append(f"  {i}. [similarity: {score:.0%}] Code: {a['agent_output'][:200]}...")
            parts.append(f"     Why it failed: {a['reasoning']}")

    if issues:
        parts.append("\nKNOWN ISSUES (from similar code):")
        for i, issue in enumerate(issues, 1):
            score = issue.get('score', 0)
            parts.append(f"  {i}. [similarity: {score:.0%}] {issue['description']}")

    return "\n".join(parts) if parts else "No highly similar past experiences found."


def judge(task: str, agent_output: str, goal: str = None, model: str = "gpt-4o-mini") -> dict:
    """
    Judge an agent's output for a given task (stateless, no memory).

    Args:
        task: The task description
        agent_output: The agent's code/response
        goal: What to evaluate for (defaults to DEFAULT_GOAL)
        model: OpenAI model to use (default: gpt-4o-mini)
    """
    goal = goal or DEFAULT_GOAL
    prompt = build_judge_prompt(task=task, agent_output=agent_output, goal=goal)

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "user", "content": prompt}
        ],
        text_format=JudgeResult,
    )

    return _build_result(task, agent_output, response.output_parsed)


def judge_with_memory(task: str, agent_output: str, graph_manager, goal: str = None, model: str = "gpt-4o-mini") -> dict:
    """
    Judge an agent's output using memory of past experiences.

    Args:
        task: The task description
        agent_output: The agent's code/response
        graph_manager: GraphManager instance for memory retrieval
        goal: What to evaluate for (defaults to DEFAULT_GOAL)
        model: OpenAI model to use (default: gpt-4o-mini)
    """
    goal = goal or DEFAULT_GOAL

    # Get embedding for the CODE to find similar implementations
    # (not task - task similarity conflates good/bad implementations of same task)
    code_embedding = get_embedding(agent_output)

    # Retrieve from memory based on code similarity
    contrastive = graph_manager.find_contrastive_attempts(code_embedding, top_k=3)
    similar_issues = graph_manager.find_similar_issues(code_embedding, top_k=5)

    # Format memory context
    memory_context = _format_memory_context(contrastive, similar_issues)

    prompt = build_judge_with_memory_prompt(
        task=task,
        agent_output=agent_output,
        goal=goal,
        memory_context=memory_context
    )

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "user", "content": prompt}
        ],
        text_format=JudgeResult,
    )

    result = _build_result(task, agent_output, response.output_parsed)
    result['memory_used'] = {
        'positive_examples': len(contrastive['positive']),
        'negative_examples': len(contrastive['negative']),
        'similar_issues': len(similar_issues)
    }

    return result


def _build_result(task: str, agent_output: str, data: JudgeResult) -> dict:
    """Build the result dict from parsed judge output."""
    policy = Policy(description=task).with_embedding()

    attempt = Attempt(
        agent_output=agent_output,
        is_successful=data.is_successful,
        reasoning=data.reasoning
    ).with_embedding()

    issues = []
    fixes = []
    relationships = []

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
