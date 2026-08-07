# TEMPPO: TEmplate-Matched Proposals for PPO

## Installation
1. Clone this repository:
    * ``git clone https://github.com/Ole-rgb/TEMPPO.git``
2. Install the [uv](https://docs.astral.sh/uv/) package manager:
    * ``pip install uv``
3. Create a virtual environment:
    * ``uv venv --python 3.11``
4. Activate it:
    * ``source .venv/bin/activate``
5. Install the project (editable install + dev tools + pre-commit hooks):
    * ``make install``

## Reproduce the study
Run all experiments and regenerate the figures with one command:
```
make reproduce
```
