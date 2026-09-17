from collections import deque

import numpy as np

try:
    import settings as s

    COLS, ROWS = s.COLS, s.ROWS
    BOMB_POWER, BOMB_TIMER, EXPLOSION_TIMER = s.BOMB_POWER, s.BOMB_TIMER, s.EXPLOSION_TIMER
    MAX_STEPS = s.MAX_STEPS
except Exception:
    COLS = ROWS = 17
    BOMB_POWER, BOMB_TIMER, EXPLOSION_TIMER = 3, 4, 2
    MAX_STEPS = 400

ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

HORIZON = BOMB_TIMER + EXPLOSION_TIMER + 1
UNREACHABLE = np.int16(9999)


def blast_coords(field, x, y, power=BOMB_POWER):
    coords = [(x, y)]
    for dx, dy in DELTAS:
        for i in range(1, power + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def free_map(game_state, block_agents=True, block_bombs=True):
    free = game_state['field'] == 0
    if block_bombs:
        for (bx, by), _ in game_state['bombs']:
            free[bx, by] = False
    if block_agents:
        for _, _, _, (ox, oy) in game_state['others']:
            free[ox, oy] = False
    return free


def lethal_schedule(game_state, extra_bombs=()):
    field = game_state['field']
    lethal = np.zeros((HORIZON + 2, COLS, ROWS), dtype=bool)

    exploding = game_state['explosion_map']
    for m in range(1, min(int(exploding.max()) if exploding.size else 0, HORIZON + 1) + 1):
        lethal[m] |= exploding >= m

    for (bx, by), t in list(game_state['bombs']) + list(extra_bombs):
        coords = blast_coords(field, bx, by)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        # index = moves made; agents act before bombs tick, so countdown t is fatal from move t+1
        for m in range(t + 1, min(t + 1 + EXPLOSION_TIMER, HORIZON + 2)):
            lethal[m][xs, ys] = True
    return lethal


def danger_map(lethal):
    danger = np.full((COLS, ROWS), UNREACHABLE, dtype=np.int16)
    for m in range(lethal.shape[0] - 1, -1, -1):
        danger[lethal[m]] = m
    return danger


def multi_source_bfs(free, target_mask):
    dist = np.full((COLS, ROWS), UNREACHABLE, dtype=np.int16)
    queue = deque()
    for x, y in np.argwhere(target_mask):
        dist[x, y] = 0
        queue.append((int(x), int(y)))
    while queue:
        x, y = queue.popleft()
        d = dist[x, y] + 1
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            if free[nx, ny] and dist[nx, ny] == UNREACHABLE:
                dist[nx, ny] = d
                queue.append((nx, ny))
    return dist


def descent_directions(dist, pos, free):
    x, y = pos
    here = int(dist[x, y])
    if here >= UNREACHABLE:
        return None, ()
    best = []
    for a, (dx, dy) in enumerate(DELTAS):
        nx, ny = x + dx, y + dy
        if (free[nx, ny] or dist[nx, ny] == 0) and dist[nx, ny] == here - 1:
            best.append(a)
    return here, tuple(best)


def escape_masks(free, lethal, start, horizon=HORIZON):
    sx, sy = start
    walkable = free.copy()
    walkable[sx, sy] = True  # the agent may be standing on its own bomb
    walkable_u8 = walkable.view(np.uint8) if walkable.dtype == bool else walkable.astype(np.uint8)

    reach = np.zeros((COLS, ROWS), dtype=np.uint8)
    for a, (dx, dy) in enumerate(DELTAS):
        nx, ny = sx + dx, sy + dy
        if free[nx, ny] and not lethal[1][nx, ny]:
            reach[nx, ny] |= np.uint8(1 << a)
    if not lethal[1][sx, sy]:
        reach[sx, sy] |= np.uint8(1 << 4)

    for m in range(2, horizon + 1):
        nxt = reach.copy()
        nxt[1:, :] |= reach[:-1, :]
        nxt[:-1, :] |= reach[1:, :]
        nxt[:, 1:] |= reach[:, :-1]
        nxt[:, :-1] |= reach[:, 1:]
        nxt *= walkable_u8
        nxt[lethal[m]] = 0
        reach = nxt
        if not reach.any():
            return np.uint8(0)
    return np.uint8(int(np.bitwise_or.reduce(reach, axis=None)))


def escape_search(free, lethal, start):
    sx, sy = start
    if lethal[0][sx, sy]:
        return False, -1, -1
    mask = escape_masks(free, lethal, start)
    if not mask:
        return False, -1, -1
    for a in range(5):
        if mask & (1 << a):
            return True, (a if a < 4 else 4), 0
    return False, -1, -1


def can_escape_after_bomb(game_state, free=None):
    _, _, bombs_left, (x, y) = game_state['self']
    if not bombs_left:
        return False
    if free is None:
        free = free_map(game_state)
    free = free.copy()
    free[x, y] = False
    lethal = lethal_schedule(game_state, extra_bombs=(((x, y), BOMB_TIMER),))
    return bool(escape_masks(free, lethal, (x, y)))


def bomb_impact(game_state, x, y):
    field = game_state['field']
    coords = blast_coords(field, x, y)
    crates = sum(1 for (cx, cy) in coords if field[cx, cy] == 1)
    others = {xy for _, _, _, xy in game_state['others']}
    hits = sum(1 for c in coords if c in others)
    return crates, hits


def _rot(x, y):
    return ROWS - 1 - y, x


def _flip(x, y):
    return COLS - 1 - x, y


D4_COORD = []
D4_ACTION_PERM = []
for _mirror in (False, True):
    for _turns in range(4):
        def _make(mirror=_mirror, turns=_turns):
            def f(x, y):
                if mirror:
                    x, y = _flip(x, y)
                for _ in range(turns):
                    x, y = _rot(x, y)
                return x, y

            return f

        D4_COORD.append(_make())
        base = [0, 3, 2, 1] if _mirror else [0, 1, 2, 3]
        perm = [(base[a] + _turns) % 4 for a in range(4)]
        D4_ACTION_PERM.append(tuple(perm) + (4, 5))
D4_COORD = tuple(D4_COORD)
D4_ACTION_PERM = tuple(D4_ACTION_PERM)


def transform_array(arr, k):
    xs, ys = np.meshgrid(np.arange(COLS), np.arange(ROWS), indexing="ij")
    nx, ny = D4_COORD[k](xs, ys)
    out = np.empty_like(arr)
    out[nx, ny] = arr
    return out


def transform_state(game_state, k):
    if k == 0:
        return game_state
    coord = D4_COORD[k]
    name, score, bombs_left, (x, y) = game_state["self"]
    return {
        "round": game_state["round"],
        "step": game_state["step"],
        "field": transform_array(game_state["field"], k),
        "explosion_map": transform_array(game_state["explosion_map"], k),
        "bombs": [(coord(bx, by), t) for (bx, by), t in game_state["bombs"]],
        "coins": [coord(cx, cy) for cx, cy in game_state["coins"]],
        "self": (name, score, bombs_left, coord(x, y)),
        "others": [(n, sc, b, coord(ox, oy)) for n, sc, b, (ox, oy) in game_state["others"]],
        "user_input": game_state["user_input"],
    }
