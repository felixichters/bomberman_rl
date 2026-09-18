import settings as s

EXTRA_SCENARIOS = {
    "crates-sparse": {"CRATE_DENSITY": 0.25, "COIN_COUNT": 20},
    "crates-medium": {"CRATE_DENSITY": 0.50, "COIN_COUNT": 15},
    "crates-dense": {"CRATE_DENSITY": 0.75, "COIN_COUNT": 9},
}


def register():
    for name, config in EXTRA_SCENARIOS.items():
        s.SCENARIOS.setdefault(name, config)
    return tuple(EXTRA_SCENARIOS)


register()
