def bipartite(z):
    n = len(z)
    colors = [-1] * n
    for start in range(n):
        if colors[start] == -1:
            queue = [start]
            colors[start] = 0
            while queue:
                u = queue.pop(0)
                for v in z[u]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        queue.append(v)
                    elif colors[v] == colors[u]:
                        return False, []
    set1 = [i for i in range(n) if colors[i] == 0]
    set2 = [i for i in range(n) if colors[i] == 1]
    return True, (set1, set2)

def kuhn(a):
    n = len(a)
    match = [-1] * n
    visited = [False] * n
    _, parts = bipartite(a)
    L = parts[0]
    def dfs(v):
        for u in a[v]:
            if not visited[u]:
                visited[u] = True
                if match[u] == -1 or dfs(match[u]):
                    match[u] = v
                    return True
        return False

    max_matching = 0
    for v in L:
        visited = [False] * n
        if dfs(v):
            max_matching += 1

    return max_matching, match


def min_vertex_cover(graph):
    bin_ly, parts = bipartite(graph)
    L = parts[0]
    R = parts[1]
    max_matching, match = kuhn(graph)
    nepokr_L = []
    for v in L:
        if match[v] == -1:
            nepokr_L.append(v)
    bily_L = [False] * len(graph)
    bily_R = [False] * len(graph)
    och_bfs = []
    for v in nepokr_L:
        bily_L[v] = True
        och_bfs.append(('L', v))

    for i in range(len(och_bfs)):
        part, v = och_bfs.pop(0)
        if part == 'L':
            for u in graph[v]:
                if not bily_R[u]:
                    bily_R[u] = True
                    och_bfs.append(('R', u))
        else:
            if match[v] != -1 and not bily_L[match[v]]:
                bily_L[match[v]] = True
                och_bfs.append(('L', match[v]))
    min_pokr = []
    for v in L:
        if not bily_L[v]:
            min_pokr.append(v)
    for u in R:
        if bily_R[u]:
            min_pokr.append(u)
    return min_pokr


G = {
    0: [3, 4],
    1: [3, 4],
    2: [3],
    3: [0, 1, 2],
    4: [0, 1]
}

print(min_vertex_cover(G))