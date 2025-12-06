"""Test the MAJ pipeline with multiple scenarios."""

from judge import judge
from graph_manager import GraphManager
from models import get_embedding

gm = GraphManager()
gm.clear_all()

# --- Test 1: Email validation (buggy regex) ---
print("=" * 50)
print("TEST 1: Email Validation")
print("=" * 50)

task1 = "Write a function to validate email addresses"
agent_output1 = """
def validate_email(email):
    import re
    pattern = r'^[a-z]+@[a-z]+\.[a-z]+$'
    return bool(re.match(pattern, email))
"""

result1 = judge(task1, agent_output1)
print(f"\nAttempt: {result1['attempt'].description[:100]}...")
print(f"\nIssue-Fix Pairs:")
for i, (issue, fix) in enumerate(zip(result1['issues'], result1['fixes'])):
    print(f"  {i+1}. Issue: {issue.description[:80]}...")
    print(f"     Fix: {fix.description[:80]}...")

# Store
gm.create_attempt(result1['attempt'])
for issue in result1['issues']:
    gm.create_issue(issue)
for fix in result1['fixes']:
    gm.create_fix(fix)
for rel in result1['relationships']:
    if rel['type'] == 'CAUSES':
        gm.link_attempt_causes_issue(rel['from_id'], rel['to_id'])
    elif rel['type'] == 'RESOLVES':
        gm.link_fix_resolves_issue(rel['from_id'], rel['to_id'])


# --- Test 2: Sorting function (inefficient) ---
print("\n" + "=" * 50)
print("TEST 2: Sorting Function")
print("=" * 50)

task2 = "Write an efficient sorting function for large arrays"
agent_output2 = """
def sort_array(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

result2 = judge(task2, agent_output2)
print(f"\nAttempt: {result2['attempt'].description[:100]}...")
print(f"\nIssue-Fix Pairs:")
for i, (issue, fix) in enumerate(zip(result2['issues'], result2['fixes'])):
    print(f"  {i+1}. Issue: {issue.description[:80]}...")
    print(f"     Fix: {fix.description[:80]}...")

# Store
gm.create_attempt(result2['attempt'])
for issue in result2['issues']:
    gm.create_issue(issue)
for fix in result2['fixes']:
    gm.create_fix(fix)
for rel in result2['relationships']:
    if rel['type'] == 'CAUSES':
        gm.link_attempt_causes_issue(rel['from_id'], rel['to_id'])
    elif rel['type'] == 'RESOLVES':
        gm.link_fix_resolves_issue(rel['from_id'], rel['to_id'])


# --- Test 3: SQL query (injection vulnerability) ---
print("\n" + "=" * 50)
print("TEST 3: SQL Query")
print("=" * 50)

task3 = "Write a function to get user by username from database"
agent_output3 = """
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query)
"""

result3 = judge(task3, agent_output3)
print(f"\nAttempt: {result3['attempt'].description[:100]}...")
print(f"\nIssue-Fix Pairs:")
for i, (issue, fix) in enumerate(zip(result3['issues'], result3['fixes'])):
    print(f"  {i+1}. Issue: {issue.description[:80]}...")
    print(f"     Fix: {fix.description[:80]}...")

# Store
gm.create_attempt(result3['attempt'])
for issue in result3['issues']:
    gm.create_issue(issue)
for fix in result3['fixes']:
    gm.create_fix(fix)
for rel in result3['relationships']:
    if rel['type'] == 'CAUSES':
        gm.link_attempt_causes_issue(rel['from_id'], rel['to_id'])
    elif rel['type'] == 'RESOLVES':
        gm.link_fix_resolves_issue(rel['from_id'], rel['to_id'])


# --- Test Vector Search ---
print("\n" + "=" * 50)
print("VECTOR SEARCH TESTS")
print("=" * 50)

queries = [
    "security vulnerability",
    "performance problem",
    "input validation",
]

for q in queries:
    print(f"\nQuery: '{q}'")
    embedding = get_embedding(q)
    results = gm.find_similar_issues(embedding, top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['description'][:70]}...")

gm.close()
print("\n" + "=" * 50)
print("DONE!")
print("=" * 50)
