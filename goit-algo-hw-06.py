import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import math

# 1) створюю граф транспортної мережі міста(райони міста як вершини)
G = nx.Graph()
G.add_nodes_from(["A", "B", "C", "D", "E", "F"])  # A=Центр, B=Вокзал, C=Університет, D=Парк, E=ТЦ, F=Лікарня

# 2) дороги (ребра) з "часом у хвилинах" як вага
edges = [
    ("A", "B", 7),
    ("A", "C", 9),
    ("A", "F", 14),
    ("B", "C", 10),
    ("B", "D", 15),
    ("C", "D", 11),
    ("C", "F", 2),
    ("D", "E", 6),
    ("E", "F", 9),
]
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# 3) характеристики графа
print("= Характеристики графа =")
print("К-сть вершин:", G.number_of_nodes())
print("К-сть ребер:", G.number_of_edges())
print("Ступені вершин:", dict(G.degree()))
print()

# 4) візуалізація
pos = nx.spring_layout(G, seed=42)  # випадкова, але стабільна розкладка
nx.draw(G, pos, with_labels=True, node_color = "#cce5ff", node_size=1200, font_size=10)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
plt.title("Транспортна мережа міста (ваги = час у хвилинах)")
plt.tight_layout()
plt.show()

# 5) DFS (пошук у глибину)
def dfs_path(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for nei in graph.neighbors(node):
            if nei not in visited:
                stack.append((nei, path + [nei]))
    return None

# 6) BFS (пошук у ширину)
def bfs_path(graph, start, goal):
    q = deque([(start, [start])])
    visited = set([start])
    while q:
        node, path = q.popleft()
        if node == goal:
            return path
        for nei in graph.neighbors(node):
            if nei not in visited:
                visited.add(nei)
                q.append((nei, path + [nei]))
    return None

# 7) Дейкстра 

def dijkstra_all_from(graph, source):
    dist = {v: math.inf for v in graph.nodes()}
    prev = {v: None for v in graph.nodes()}
    dist[source] = 0
    unvisited = set(graph.nodes())

    while unvisited:

        # вибираю вершину з мінімальною дистанцією (просто лінійно)
        u = min(unvisited, key=lambda x: dist[x])
        unvisited.remove(u)

        # якщо вже "нескінченність", то далі сенсу немає
        if dist[u] == math.inf:
            break

        # релаксація сусідів
        for v in graph.neighbors(u):
            w = graph[u][v]["weight"]
            if v in unvisited:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u

    return dist, prev

def restore_path(prev, start, goal):
    if prev[goal] is None and start != goal:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

# 8) приклад порівняння DFS vs BFS (шлях між A і F)
start, goal = "A", "F"
path_dfs = dfs_path(G, start, goal)
path_bfs = bfs_path(G, start, goal)

print("=== DFS vs BFS (A -> F) ===")
print("DFS шлях:", path_dfs)
print("BFS шлях:", path_bfs)
print("Пояснення: BFS знаходить найкоротший шлях у кількості кроків (за ребрами),")
print("а DFS йде “в глибину” і може видати інший, не найкоротший за к-стю ребер шлях.")
print()

# 9) Дейкстра: найкоротші шляхи (за вагою) між усіма вершинами
print("=== Дейкстра: найкоротші шляхи (за часом) від кожної вершини ===")
all_pairs = {}
for s in G.nodes():
    dist, prev = dijkstra_all_from(G, s)
    all_pairs[s] = dist

    # приклад
    target = "E"
    p = restore_path(prev, s, target)
    print(f"Від {s} до {target}: шлях={p}, час={dist[target]}")

print("\nТаблиця мінімальних часів (dist):")
for s in G.nodes():
    print(s, ":", all_pairs[s])
