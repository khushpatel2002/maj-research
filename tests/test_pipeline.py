"""Test MAJ pipeline: stateless judge vs memory-assisted judge."""

import sys
sys.path.insert(0, "src")

from judge import judge_with_memory, judge
from graph_manager import GraphManager

gm = GraphManager()
gm.clear_all()


def store_result(gm, result):
    """Store judge result in Neo4j."""
    policy_id, _ = gm.get_or_create_policy(result['policy'])

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


def print_result(result, title):
    """Print judge result."""
    attempt = result['attempt']
    status = "✓ PASSED" if attempt.is_successful else "✗ FAILED"

    print(f"\n{title}")
    print(f"{status}")
    print(f"Reasoning: {attempt.reasoning[:120]}...")

    if 'memory_used' in result:
        m = result['memory_used']
        print(f"Memory: {m['positive_examples']} positive, {m['negative_examples']} negative, {m['similar_issues']} issues")


# ============================================================
# PHASE 1: Build memory with stateless judge
# ============================================================
print("=" * 60)
print("PHASE 1: Building Memory (stateless judge)")
print("=" * 60)

# Store some examples in memory
examples = [
    ("Write a function to validate email addresses",
     "def validate(e): return '@' in e",
     "Bad: too simple"),

    ("Write a function to validate email addresses",
     "import re\ndef validate(e): return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$', e))",
     "Good: proper regex"),

    ("Write a function to query user from database",
     "def get_user(name): return db.execute(f\"SELECT * FROM users WHERE name='{name}'\")",
     "Bad: SQL injection"),

    ("Write a function to query user from database",
     "def get_user(name): return db.execute('SELECT * FROM users WHERE name=?', (name,))",
     "Good: parameterized"),
]

for task, code, label in examples:
    result = judge(task, code)
    store_result(gm, result)
    status = "✓" if result['attempt'].is_successful else "✗"
    print(f"  {status} {label}")


# ============================================================
# PHASE 2: Test with memory-assisted judge
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Testing Memory-Assisted Judge")
print("=" * 60)

# New attempt - similar to what we've seen before
new_task = "Write a function to validate email addresses"
new_code = """
def check_email(email):
    if '@' not in email:
        return False
    if '.' not in email:
        return False
    return True
"""

print(f"\nTask: {new_task}")
print(f"Code: Simple @ and . check")

result_with_memory = judge_with_memory(new_task, new_code, gm)
print_result(result_with_memory, "Memory-Assisted Judge:")


# ============================================================
# PHASE 3: Another test
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: SQL Query Test")
print("=" * 60)

sql_task = "Write a function to get user by id from database"
sql_code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""

print(f"\nTask: {sql_task}")
print(f"Code: f-string SQL query")

print("\n--- With Memory ---")
result_sql = judge_with_memory(sql_task, sql_code, gm)
print_result(result_sql, "Memory-Assisted Judge:")


gm.close()
print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
