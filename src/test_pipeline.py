"""Test the MAJ pipeline with feedback (is_successful + reasoning)."""

from judge import judge
from graph_manager import GraphManager

gm = GraphManager()
gm.clear_all()


def store_result(gm, result):
    """Store judge result in Neo4j."""
    policy_id, is_new = gm.get_or_create_policy(result['policy'])

    gm.create_attempt(result['attempt'])
    for issue in result['issues']:
        gm.create_issue(issue)
    for fix in result['fixes']:
        gm.create_fix(fix)

    for rel in result['relationships']:
        if rel['type'] == 'SATISFIES':
            gm.link_attempt_satisfies_policy(rel['from_id'], policy_id)
        elif rel['type'] == 'CAUSES':
            gm.link_attempt_causes_issue(rel['from_id'], rel['to_id'])
        elif rel['type'] == 'RESOLVES':
            gm.link_fix_resolves_issue(rel['from_id'], rel['to_id'])

    return policy_id


def print_result(result):
    """Print judge result with feedback."""
    attempt = result['attempt']
    status = "✓ PASSED" if attempt.is_successful else "✗ FAILED"

    print(f"\n{status}")
    print(f"Attempt: {attempt.description[:70]}...")
    print(f"Reasoning: {attempt.reasoning[:100]}...")

    if result['issues']:
        print(f"\nIssues ({len(result['issues'])}):")
        for i, (issue, fix) in enumerate(zip(result['issues'], result['fixes'])):
            print(f"  {i+1}. {issue.description[:60]}...")
            print(f"     Fix: {fix.description[:60]}...")


# --- Test 1: Bad email validation (should fail) ---
print("=" * 60)
print("TEST 1: Email Validation - Bad Regex")
print("=" * 60)

task1 = "Write a function to validate email addresses"
agent_output1 = """
def validate_email(email):
    import re
    pattern = r'^[a-z]+@[a-z]+\.[a-z]+$'
    return bool(re.match(pattern, email))
"""

result1 = judge(task1, agent_output1)
print_result(result1)
store_result(gm, result1)


# --- Test 2: Good email validation (should pass) ---
print("\n" + "=" * 60)
print("TEST 2: Email Validation - Good Implementation")
print("=" * 60)

task2 = "Write a function to validate email addresses"
agent_output2 = """
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not email or not isinstance(email, str):
        return False
    return bool(re.match(pattern, email))
"""

result2 = judge(task2, agent_output2)
print_result(result2)
store_result(gm, result2)


# --- Test 3: SQL Injection vulnerability (should fail) ---
print("\n" + "=" * 60)
print("TEST 3: SQL Query - Injection Vulnerability")
print("=" * 60)

task3 = "Write a function to get user by username from database"
agent_output3 = """
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query)
"""

result3 = judge(task3, agent_output3)
print_result(result3)
store_result(gm, result3)


# --- Test 4: Safe SQL query (should pass) ---
print("\n" + "=" * 60)
print("TEST 4: SQL Query - Safe Implementation")
print("=" * 60)

task4 = "Write a function to get user by username from database"
agent_output4 = """
def get_user(username):
    query = "SELECT * FROM users WHERE username = ?"
    return db.execute(query, (username,))
"""

result4 = judge(task4, agent_output4)
print_result(result4)
store_result(gm, result4)


# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

results = [result1, result2, result3, result4]
passed = sum(1 for r in results if r['attempt'].is_successful)
failed = len(results) - passed

print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")


gm.close()
print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
