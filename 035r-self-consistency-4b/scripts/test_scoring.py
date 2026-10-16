"""Verify scoring invariants that can materially change the experimental result."""
from analyse import parse,parse_strict,vote
assert parse('Answer: 1,250.00','off')=='1.25E+3'
assert parse(r'Final Answer: $$\boxed{2}$$','off')=='2'
assert parse(r'Final Answer: $$\text{Answer: } 50$$','off')=='5E+1'
assert parse('Answer: **$40**','off')=='4E+1'
assert parse('<think>Answer: 999</think>Answer: 2','on')=='2'
assert parse('<think>Answer: 999','on') is None
assert parse('<think>Answer: 999</think>No final answer','on') is None
assert parse('I calculated 12 then 42.','off') is None
assert parse_strict(r'Final Answer: $$\boxed{2}$$','off') is None
assert vote(['2','3','3','2'],4)=='2'
assert vote([None,None,'2'],3)=='2'
assert vote([None,None],2) is None
print('Scoring invariants passed')
