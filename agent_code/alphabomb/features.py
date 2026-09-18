from collections import namedtuple

import numpy as np

from . import game_utils as g

FEATURE_VERSION = "v2"

BLOCKS = (
    ("neighbours", 12),
    ("danger", 10),
    ("escape", 6),
    ("coin", 6),
    ("crate", 6),
    ("enemy", 6),
    ("bomb", 3),
    ("context", 4),
    ("bias", 1),
    ("interactions", 12),
)
BLOCK_SIZE = dict(BLOCKS)

FEATURE_SETS = {
    "base": tuple(name for name, _ in BLOCKS if name != "interactions"),
    "v2": tuple(name for name, _ in BLOCKS),
}
for _block, _ in BLOCKS:
    if _block != "bias":
        FEATURE_SETS[f"no_{_block}"] = tuple(name for name, _ in BLOCKS if name != _block)
FEATURE_SETS["minimal"] = ("neighbours", "danger", "coin", "bias")
DEFAULT_SET = "v2"

DIR_NAMES = g.ACTIONS[:4]


def feature_names(feature_set=DEFAULT_SET):
    names = []
    for block in FEATURE_SETS[feature_set]:
        if block == "neighbours":
            names += [f"nb_{kind}_{d}" for kind in ("free", "crate", "blocked") for d in DIR_NAMES]
        elif block == "danger":
            names += [f"lethal_next_{d}" for d in DIR_NAMES]
            names += [f"urgency_{d}" for d in DIR_NAMES]
            names += ["lethal_next_HERE", "urgency_HERE"]
        elif block == "escape":
            names += [f"escape_{a}" for a in g.ACTIONS[:5]] + ["escape_BOMB"]
        elif block in ("coin", "crate", "enemy"):
            names += [f"{block}_dir_{d}" for d in DIR_NAMES] + [f"{block}_none", f"{block}_dist"]
        elif block == "bomb":
            names += ["bomb_crates", "bomb_hits_enemy", "bombs_left"]
        elif block == "context":
            names += ["step_frac", "coins_visible", "crates_left", "enemies_left"]
        elif block == "bias":
            names += ["bias"]
        elif block == "interactions":
            names += [f"{t}_x_escape_{d}" for t in ("coin", "crate", "enemy") for d in DIR_NAMES]
    return tuple(names)


def feature_dim(feature_set=DEFAULT_SET):
    return sum(BLOCK_SIZE[b] for b in FEATURE_SETS[feature_set])


StateInfo = namedtuple("StateInfo", [
    "pos", "field", "free_move", "free_nav", "lethal", "danger",
    "escape_mask", "escape_bomb",
    "coin_dist", "coin_dir", "crate_dist", "crate_dir", "enemy_dist", "enemy_dir",
    "bomb_crates", "bomb_hits", "bombs_left", "n_coins", "n_crates", "n_enemies", "step",
])


def analyse(game_state):
    field = game_state["field"]
    _, _, bombs_left, pos = game_state["self"]

    free_move = g.free_map(game_state, block_agents=True, block_bombs=True)
    free_nav = g.free_map(game_state, block_agents=False, block_bombs=True)

    lethal = g.lethal_schedule(game_state)
    danger = g.danger_map(lethal)

    escape_mask = int(g.escape_masks(free_move, lethal, pos))
    escape_bomb = g.can_escape_after_bomb(game_state, free=free_move) if bombs_left else False

    coin_mask = np.zeros_like(field, dtype=bool)
    for (cx, cy) in game_state["coins"]:
        coin_mask[cx, cy] = True
    coin_dist, coin_dir = g.descent_directions(g.multi_source_bfs(free_nav, coin_mask), pos, free_nav)

    crates = field == 1
    crate_mask = np.zeros_like(crates)
    crate_mask[1:, :] |= crates[:-1, :]
    crate_mask[:-1, :] |= crates[1:, :]
    crate_mask[:, 1:] |= crates[:, :-1]
    crate_mask[:, :-1] |= crates[:, 1:]
    crate_mask &= field == 0
    crate_dist, crate_dir = g.descent_directions(g.multi_source_bfs(free_nav, crate_mask), pos, free_nav)

    enemy_mask = np.zeros_like(field, dtype=bool)
    for _, _, _, (ox, oy) in game_state["others"]:
        enemy_mask[ox, oy] = True
    enemy_dist, enemy_dir = g.descent_directions(g.multi_source_bfs(free_nav, enemy_mask), pos, free_nav)

    bomb_crates, bomb_hits = g.bomb_impact(game_state, *pos)

    return StateInfo(
        pos=pos, field=field, free_move=free_move, free_nav=free_nav,
        lethal=lethal, danger=danger, escape_mask=escape_mask, escape_bomb=bool(escape_bomb),
        coin_dist=coin_dist, coin_dir=coin_dir,
        crate_dist=crate_dist, crate_dir=crate_dir,
        enemy_dist=enemy_dist, enemy_dir=enemy_dir,
        bomb_crates=bomb_crates, bomb_hits=bomb_hits, bombs_left=bool(bombs_left),
        n_coins=len(game_state["coins"]), n_crates=int(crates.sum()),
        n_enemies=len(game_state["others"]), step=game_state["step"],
    )


_MAX_DIST = float(g.COLS + g.ROWS)
_URGENCY_SCALE = float(g.HORIZON)


def _target_block(dist, directions, out):
    if dist is None:
        out[4] = 1.0
        out[5] = 1.0
    else:
        for a in directions:
            out[a] = 1.0
        out[5] = min(dist / _MAX_DIST, 1.0)


def features_from(info, feature_set=DEFAULT_SET):
    x = np.zeros(feature_dim(feature_set), dtype=np.float32)
    at = 0
    px, py = info.pos
    neighbours = [(px + dx, py + dy) for dx, dy in g.DELTAS]

    for block in FEATURE_SETS[feature_set]:
        n = BLOCK_SIZE[block]
        v = x[at:at + n]
        if block == "neighbours":
            for i, (nx, ny) in enumerate(neighbours):
                tile = info.field[nx, ny]
                v[i] = float(info.free_move[nx, ny])
                v[4 + i] = float(tile == 1)
                v[8 + i] = float(tile == 0 and not info.free_move[nx, ny])
        elif block == "danger":
            for i, (nx, ny) in enumerate(neighbours):
                v[i] = float(info.lethal[1][nx, ny])
                d = info.danger[nx, ny]
                v[4 + i] = 0.0 if d >= g.UNREACHABLE else max(0.0, (_URGENCY_SCALE + 1 - d) / _URGENCY_SCALE)
            v[8] = float(info.lethal[1][px, py])
            d = info.danger[px, py]
            v[9] = 0.0 if d >= g.UNREACHABLE else max(0.0, (_URGENCY_SCALE + 1 - d) / _URGENCY_SCALE)
        elif block == "escape":
            for a in range(5):
                v[a] = float(bool(info.escape_mask & (1 << a)))
            v[5] = float(info.escape_bomb)
        elif block == "coin":
            _target_block(info.coin_dist, info.coin_dir, v)
        elif block == "crate":
            _target_block(info.crate_dist, info.crate_dir, v)
        elif block == "enemy":
            _target_block(info.enemy_dist, info.enemy_dir, v)
        elif block == "bomb":
            v[0] = min(info.bomb_crates / 4.0, 1.0)
            v[1] = float(info.bomb_hits > 0)
            v[2] = float(info.bombs_left)
        elif block == "context":
            v[0] = min(info.step / float(g.MAX_STEPS), 1.0)
            v[1] = min(info.n_coins / 9.0, 1.0)
            v[2] = min(info.n_crates / 100.0, 1.0)
            v[3] = info.n_enemies / 3.0
        elif block == "bias":
            v[0] = 1.0
        elif block == "interactions":
            for j, directions in enumerate((info.coin_dir, info.crate_dir, info.enemy_dir)):
                for direction in directions:
                    if info.escape_mask & (1 << direction):
                        v[4 * j + direction] = 1.0
        at += n
    return x


def state_to_features(game_state, feature_set=DEFAULT_SET):
    if game_state is None:
        return None
    return features_from(analyse(game_state), feature_set)


_DIRECTION_SLOTS = {
    "neighbours": (0, 4, 8),
    "danger": (0, 4),
    "escape": (0,),
    "coin": (0,),
    "crate": (0,),
    "enemy": (0,),
    "interactions": (0, 4, 8),
}


def feature_permutation(k, feature_set=DEFAULT_SET):
    perm = g.D4_ACTION_PERM[k]
    P = np.arange(feature_dim(feature_set))
    at = 0
    for block in FEATURE_SETS[feature_set]:
        for offset in _DIRECTION_SLOTS.get(block, ()):
            base = at + offset
            for d in range(4):
                P[base + perm[d]] = base + d
        at += BLOCK_SIZE[block]
    return P
