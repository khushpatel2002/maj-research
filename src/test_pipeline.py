"""Test the MAJ pipeline with Policy reuse."""

from judge import judge
from graph_manager import GraphManager
from models import get_embedding

gm = GraphManager()
gm.clear_all()


def store_result(gm, result):
    """Store judge result in Neo4j. Reuses existing Policy if similar."""
    # Get or create policy (reuse if similar exists)
    policy_id, is_new = gm.get_or_create_policy(result['policy'])
    print(f"  Policy: {'NEW' if is_new else 'REUSED'} (id: {policy_id[:8]}...)")

    gm.create_attempt(result['attempt'])
    for issue in result['issues']:
        gm.create_issue(issue)
    for fix in result['fixes']:
        gm.create_fix(fix)

    for rel in result['relationships']:
        if rel['type'] == 'SATISFIES':
            # Use the actual policy_id (might be existing one)
            gm.link_attempt_satisfies_policy(rel['from_id'], policy_id)
        elif rel['type'] == 'CAUSES':
            gm.link_attempt_causes_issue(rel['from_id'], rel['to_id'])
        elif rel['type'] == 'RESOLVES':
            gm.link_fix_resolves_issue(rel['from_id'], rel['to_id'])

    return policy_id


# --- Test 1: Email validation (first attempt) ---
print("=" * 50)
print("TEST 1: Email Validation - Attempt 1")
print("=" * 50)

task1 = "Write a function to validate email addresses"
agent_output1 = """
def validate_email(email):
    import re
    pattern = r'^[a-z]+@[a-z]+\.[a-z]+$'
    return bool(re.match(pattern, email))
"""

result1 = judge(task1, agent_output1)
print(f"\nTask: {task1}")
print(f"Attempt: {result1['attempt'].description[:60]}...")
policy_id_1 = store_result(gm, result1)


# --- Test 2: Email validation (second attempt - should reuse policy) ---
print("\n" + "=" * 50)
print("TEST 2: Email Validation - Attempt 2 (same task)")
print("=" * 50)

task2 = "Validate email addresses"  # Similar task, slightly different wording
agent_output2 = """
def check_email(email):
    return '@' in email and '.' in email
"""

result2 = judge(task2, agent_output2)
print(f"\nTask: {task2}")
print(f"Attempt: {result2['attempt'].description[:60]}...")
policy_id_2 = store_result(gm, result2)


# --- Test 3: Different task (should create new policy) ---
print("\n" + "=" * 50)
print("TEST 3: SQL Query (different task)")
print("=" * 50)

task3 = "Write a function to get user by username from database"
agent_output3 = """
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query)
"""

result3 = judge(task3, agent_output3)
print(f"\nTask: {task3}")
print(f"Attempt: {result3['attempt'].description[:60]}...")
policy_id_3 = store_result(gm, result3)


# --- Verify Policy Reuse ---
print("\n" + "=" * 50)
print("POLICY REUSE CHECK")
print("=" * 50)

print(f"\nTest 1 & 2 same policy? {policy_id_1 == policy_id_2}")
print(f"Test 3 different policy? {policy_id_1 != policy_id_3}")


# --- Show all attempts for email validation policy ---
print("\n" + "=" * 50)
print("ALL ATTEMPTS FOR EMAIL POLICY")
print("=" * 50)

attempts = gm.get_attempts_for_policy(policy_id_1)
print(f"\nPolicy ID: {policy_id_1[:8]}...")
print(f"Attempts ({len(attempts)}):")
for a in attempts:
    print(f"  - {a['description'][:60]}...")


gm.close()
print("\n" + "=" * 50)
print("DONE!")
print("=" * 50)
