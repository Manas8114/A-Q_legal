from typing import List
from collections import defaultdict, deque

class Solution:
    def recoverNetwork(self, network_nodes: int, network_from: List[int], network_to: List[int], company: int) -> List[int]:
        # Handle edge case: only company node
        if network_nodes == 1:
            return []
        
        # Build graph
        graph = defaultdict(list)
        for i in range(len(network_from)):
            from_node = network_from[i]
            to_node = network_to[i]
            graph[from_node].append(to_node)
            graph[to_node].append(from_node)
        
        # BFS to find shortest distances from company
        distances = defaultdict(list)
        queue = deque([(company, 0)])
        visited = {company}
        
        while queue:
            node, dist = queue.popleft()
            
            # Add to distance group (excluding company node)
            if node != company:
                distances[dist].append(node)
            
            # Explore all neighbors
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # Build result: nodes sorted by distance, then by node number
        result = []
        for dist in sorted(distances.keys()):
            nodes_at_distance = sorted(distances[dist])
            result.extend(nodes_at_distance)
        
        return result
