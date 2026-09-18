from . import game_utils as g

MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
MOVED_TOWARDS_CRATE = "MOVED_TOWARDS_CRATE"
MOVED_AWAY_FROM_CRATE = "MOVED_AWAY_FROM_CRATE"
LEFT_DANGER = "LEFT_DANGER"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"
SUICIDAL_BOMB = "SUICIDAL_BOMB"
USELESS_BOMB = "USELESS_BOMB"
GOOD_BOMB = "GOOD_BOMB"
WAITED_IN_DANGER = "WAITED_IN_DANGER"
OSCILLATED = "OSCILLATED"

CUSTOM_EVENTS = (
    MOVED_TOWARDS_COIN, MOVED_AWAY_FROM_COIN, MOVED_TOWARDS_CRATE, MOVED_AWAY_FROM_CRATE,
    LEFT_DANGER, STAYED_IN_DANGER, ENTERED_DANGER, SUICIDAL_BOMB, USELESS_BOMB,
    GOOD_BOMB, WAITED_IN_DANGER, OSCILLATED,
)

TASK_REWARDS = {"COIN_COLLECTED": 1.0, "KILLED_OPPONENT": 5.0}

DEFAULT_EVENT_REWARDS = {
    "INVALID_ACTION": -0.4,
    "WAITED": -0.05,
    "KILLED_SELF": -8.0,
    "GOT_KILLED": -8.0,
    "CRATE_DESTROYED": 0.15,
    "COIN_FOUND": 0.1,
    "SURVIVED_ROUND": 1.0,
    MOVED_TOWARDS_COIN: 0.15,
    MOVED_AWAY_FROM_COIN: -0.18,
    MOVED_TOWARDS_CRATE: 0.05,
    MOVED_AWAY_FROM_CRATE: -0.06,
    LEFT_DANGER: 0.5,
    STAYED_IN_DANGER: -0.3,
    ENTERED_DANGER: -0.4,
    WAITED_IN_DANGER: -0.5,
    SUICIDAL_BOMB: -3.0,
    USELESS_BOMB: -0.5,
    GOOD_BOMB: 0.4,
    OSCILLATED: -0.1,
}

MINIMAL_EVENT_REWARDS = {
    "INVALID_ACTION": -0.4,
    "KILLED_SELF": -8.0,
    "GOT_KILLED": -8.0,
    "CRATE_DESTROYED": 0.1,
    "COIN_FOUND": 0.1,
}

DEATH_EVENTS = ("KILLED_SELF", "GOT_KILLED")

_MAX_DIST = float(g.COLS + g.ROWS)

CRATE_REFERENCE = 123.0


def _in_danger(info):
    return int(info.danger[info.pos]) < int(g.UNREACHABLE)


def derive_events(old_info, action, new_info, events, history=()):
    out = []
    if old_info is None:
        return out

    if new_info is not None:
        for towards, away, old_d, new_d in (
            (MOVED_TOWARDS_COIN, MOVED_AWAY_FROM_COIN, old_info.coin_dist, new_info.coin_dist),
            (MOVED_TOWARDS_CRATE, MOVED_AWAY_FROM_CRATE, old_info.crate_dist, new_info.crate_dist),
        ):
            if old_d is not None and new_d is not None:
                if new_d < old_d:
                    out.append(towards)
                elif new_d > old_d:
                    out.append(away)

        was, now = _in_danger(old_info), _in_danger(new_info)
        if was and not now:
            out.append(LEFT_DANGER)
        elif was and now:
            out.append(STAYED_IN_DANGER)
            if action == "WAIT":
                out.append(WAITED_IN_DANGER)
        elif not was and now and "BOMB_DROPPED" not in events:
            out.append(ENTERED_DANGER)

        if len(history) >= 3 and new_info.pos == history[-3]:
            out.append(OSCILLATED)

    if "BOMB_DROPPED" in events:
        if not old_info.escape_bomb:
            out.append(SUICIDAL_BOMB)
        elif old_info.bomb_crates == 0 and old_info.bomb_hits == 0:
            out.append(USELESS_BOMB)
        else:
            out.append(GOOD_BOMB)
    return out


def _proximity(distance):
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - distance / _MAX_DIST)


def potential(info, h):
    if info is None:
        return 0.0
    psi = (h.psi_coin * _proximity(info.coin_dist)
           + h.psi_crate * _proximity(info.crate_dist)
           + h.psi_enemy * _proximity(info.enemy_dist))
    cleared = max(0.0, 1.0 - info.n_crates / CRATE_REFERENCE)

    danger = int(info.danger[info.pos])
    safety = 1.0 if danger >= int(g.UNREACHABLE) else min(1.0, (danger - 1) / float(g.HORIZON))

    escapable = 1.0 if info.escape_mask else 0.0
    return (psi + h.psi_safe * safety + h.psi_cleared * cleared
            + h.psi_escape * escapable)


def event_table(scheme, h=None, custom=None):
    table = dict(TASK_REWARDS)
    if scheme == "task":
        pass
    elif scheme == "events":
        table.update(custom if custom is not None else DEFAULT_EVENT_REWARDS)
    elif scheme == "potential":
        pass
    elif scheme == "potential+events":
        table.update(MINIMAL_EVENT_REWARDS)
    else:
        raise ValueError(f"unknown reward scheme {scheme!r}")
    if h is not None and scheme in ("events", "potential+events"):
        table["KILLED_SELF"] = table["GOT_KILLED"] = float(h.death_penalty)
        table["INVALID_ACTION"] = float(h.invalid_penalty)
        table["CRATE_DESTROYED"] = float(h.crate_reward)
        table["COIN_FOUND"] = float(h.coin_found_reward)
    return table


def reward_from(events, table, h, old_info=None, new_info=None, shaped=True, bomb_bonus=0.0):
    priced = list(events)
    if all(ev in priced for ev in DEATH_EVENTS):
        priced.remove("GOT_KILLED")  # a suicide fires both events; count it once
    total = sum(table.get(ev, 0.0) for ev in priced) + h.step_penalty
    if bomb_bonus and GOOD_BOMB in priced:
        total += bomb_bonus
    if shaped:
        total += h.gamma * potential(new_info, h) - potential(old_info, h)
    return float(total)


def scheme_is_shaped(scheme):
    return scheme in ("potential", "potential+events")


def summarise_events(events):
    return {ev: events.count(ev) for ev in CUSTOM_EVENTS if ev in events}
