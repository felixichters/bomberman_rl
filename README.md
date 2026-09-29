# bomberman_rl AlphaBomb

RL agent for the MLE final project.
Team AlphaBomb: Felix Ichters, Lukas Dzielski.

The agent is in `agent_code/alphabomb/`.

(We moved the code to a clean repo to keep the report out, but feel free to contact us if you need the original commit history.)

## Setup

    uv venv --python 3.12 .venv
    uv pip install -r requirements-dev.txt

The agent itself needs only `numpy` and `scikit-learn`, both already installed by the
course Dockerfile, so it ships without a `requirements.txt`.

## Run

    .venv/bin/python main.py play --my-agent alphabomb

## Evaluate

    .venv/bin/python analysis/evaluate.py \
        --agents alphabomb "rule_based_agent x3" --rounds 200 --workers 5

## Tests

    .venv/bin/python -m pytest -q

## Layout

    agent_code/alphabomb   the agent
    training/              curriculum, drivers
    analysis/              evaluation, figures
    experiments/           experiment configs
    tests/                 tests
