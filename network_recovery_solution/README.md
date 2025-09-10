# Network Recovery Solution

This solution solves the network recovery problem where nodes need to be recovered in order of proximity from a company node.

## Problem Description

Given a network of nodes connected by bidirectional edges, determine the order to recover nodes after a major incident, starting from a company node.

### Recovery Rules:
1. Start from the main node (company) - already online
2. Recover nodes in order of proximity (shortest hops from company)
3. If multiple nodes are equally distant, recover the one with lower node number first
4. Ignore isolated/unreachable nodes

## Solution

- **Algorithm**: Breadth-First Search (BFS)
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V + E)

## Files

- `solution.py` - LeetCode/HackerRank ready solution
- `README.md` - This documentation

## Usage

```python
from solution import Solution

sol = Solution()
result = sol.recoverNetwork(4, [1, 2, 2], [2, 3, 4], 1)
print(result)  # Output: [2, 3, 4]
```

## Test Cases

- Example: `recoverNetwork(4, [1, 2, 2], [2, 3, 4], 1)` → `[2, 3, 4]`
- Single node: `recoverNetwork(1, [], [], 1)` → `[]`
- Linear chain: `recoverNetwork(5, [1, 2, 3, 4], [2, 3, 4, 5], 1)` → `[2, 3, 4, 5]`
- Star topology: `recoverNetwork(4, [1, 1, 1], [2, 3, 4], 1)` → `[2, 3, 4]`
