import numpy as np

from agent_code.alphabomb import rewards as R
from agent_code.alphabomb.learning import Hyper


class FakeInfo:
    def __init__(self, coin=None, crate=None, enemy=None, danger=10_000, crates=123,
                 escape_mask=0b11111):
        self.coin_dist, self.crate_dist, self.enemy_dist = coin, crate, enemy
        self.pos = (1, 1)
        self.danger = np.full((17, 17), danger, dtype=np.int16)
        self.n_crates = crates
        self.escape_mask = escape_mask


def test_suicide_is_priced_once():
    h = Hyper(reward_scheme="potential+events")
    table = R.event_table("potential+events")
    both = R.reward_from(["KILLED_SELF", "GOT_KILLED"], table, h, shaped=False)
    alone = R.reward_from(["KILLED_SELF"], table, h, shaped=False)
    assert both == alone == table["KILLED_SELF"]


def test_killed_by_someone_else_still_costs():
    h = Hyper(reward_scheme="potential+events")
    table = R.event_table("potential+events")
    assert R.reward_from(["GOT_KILLED"], table, h, shaped=False) == table["GOT_KILLED"]


def test_task_reward_matches_the_game_score():
    h = Hyper(reward_scheme="task")
    table = R.event_table("task")
    assert R.reward_from(["COIN_COLLECTED"], table, h, shaped=False) == 1.0
    assert R.reward_from(["KILLED_OPPONENT"], table, h, shaped=False) == 5.0
    assert R.reward_from(["CRATE_DESTROYED", "WAITED"], table, h, shaped=False) == 0.0


def test_potential_is_bounded_and_zero_at_terminal():
    h = Hyper()
    assert R.potential(None, h) == 0.0
    ceiling = (h.psi_coin + h.psi_crate + h.psi_enemy + h.psi_safe
               + h.psi_cleared + h.psi_escape)
    for info in (FakeInfo(), FakeInfo(coin=0, crate=0, enemy=0, crates=0),
                 FakeInfo(coin=30, crate=5, enemy=12, danger=0, crates=123),
                 FakeInfo(escape_mask=0)):
        psi = R.potential(info, h)
        assert 0.0 <= psi <= ceiling + 1e-9


def test_dying_never_pays():
    h = Hyper()
    for info in (FakeInfo(coin=3, crate=1, crates=60), FakeInfo(danger=0),
                 FakeInfo(escape_mask=0)):
        assert R.reward_from([], {}, h, old_info=info, new_info=None, shaped=True) <= 0.0


def test_shaping_telescopes_to_minus_psi_of_the_start():
    h = Hyper()
    trajectory = [FakeInfo(coin=8, crates=123), FakeInfo(coin=6, crates=120),
                  FakeInfo(coin=3, crates=110), FakeInfo(coin=0, crates=100)]
    extra, discount = 0.0, 1.0
    for old, new in zip(trajectory, trajectory[1:] + [None]):
        extra += discount * (h.gamma * R.potential(new, h) - R.potential(old, h))
        discount *= h.gamma
    assert abs(extra + R.potential(trajectory[0], h)) < 1e-9


def test_clearing_crates_raises_the_potential():
    h = Hyper()
    assert R.potential(FakeInfo(crates=20), h) > R.potential(FakeInfo(crates=120), h)


def test_bomb_events_are_derived_from_the_escape_analysis():
    old = FakeInfo()
    old.escape_bomb, old.bomb_crates, old.bomb_hits = False, 3, 0
    assert R.SUICIDAL_BOMB in R.derive_events(old, "BOMB", None, ["BOMB_DROPPED"])
    old.escape_bomb = True
    assert R.GOOD_BOMB in R.derive_events(old, "BOMB", None, ["BOMB_DROPPED"])
    old.bomb_crates = 0
    assert R.USELESS_BOMB in R.derive_events(old, "BOMB", None, ["BOMB_DROPPED"])


def test_every_custom_event_has_a_price_in_the_event_scheme():
    table = R.event_table("events", Hyper())
    missing = [ev for ev in R.CUSTOM_EVENTS if ev not in table]
    assert not missing, missing
    assert all(np.isfinite(list(table.values())))


def test_being_trapped_lowers_the_potential():
    h = Hyper()
    trapped = FakeInfo(coin=4, crates=60, escape_mask=0)
    free = FakeInfo(coin=4, crates=60, escape_mask=0b11111)
    assert R.potential(free, h) > R.potential(trapped, h)
    assert R.potential(free, h) - R.potential(trapped, h) == h.psi_escape
