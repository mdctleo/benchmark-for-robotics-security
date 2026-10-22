# Vendored code

Third-party packages the RoboGuard defense and the RoboPAIR attack run on,
copied as they are so they behave exactly as in the RoboBench experiments.
They keep their own top-level package names (`roboguard`, `spine`,
`robopair`); `robobench.defenses.roboguard` and `robobench.attacks.robopair`
put this directory on `sys.path` when a `RoboGuard` or `RoboPAIR` is
constructed.

| package     | upstream                                                     | licence |
|-------------|--------------------------------------------------------------|---------|
| `roboguard` | https://github.com/ZacRavichandran/roboguard (robo-guard.github.io) | MIT, see `roboguard/LICENSE` |
| `spine`     | https://github.com/KumarRobotics/SPINE                       | BSD-3, see `spine/LICENSE` |
| `robopair`  | https://github.com/arobey1/robopair (robopair.org)            | no licence file upstream |

Left out of the copy: `roboguard/eval/` (upstream evaluation scripts) and
`spine/viz/viz_ros.py` (ROS visualisation). `spine/` and its subpackages got
empty `__init__.py` files. One local change: `roboguard/generator.py` takes
the OpenAI model as a `model` constructor argument (default `gpt-5.4-nano`,
the value upstream hard-coded) so `RoboGuard(model=...)` can select it.

`roboguard` needs the [spot](https://spot.lre.epita.fr/) LTL library with its
Python bindings, which is not on PyPI: `conda install -c conda-forge spot`,
`brew install spot`, or `apt install python3-spot`. If spot is installed for
another interpreter, point `ROBOBENCH_SPOT_SITE_PACKAGES` at that
`site-packages` directory. The remaining Python dependencies are the
`roboguard` extra: `pip install robobench[roboguard]`.

`robopair` is the attack search (`main.py`, `conversers.py`, `judges.py`,
`common.py`, `config.py`, `language_models.py`, `system_prompts/`). Left out:
`dolphin/` (the NVIDIA Dolphins model), `loggers.py` (Weights & Biases),
the launch scripts and data. Local changes:

- `main.py`: the command-line block is removed and `main(args)` returns the
  jailbreak prompt (the last one the judge scored 10, else `None`); the wandb
  logger is replaced by `common.RunLog`, which keeps the same bookkeeping.
- `language_models.py`: only the OpenAI class is kept; it takes an optional
  `client` and sends `max_completion_tokens` instead of `max_tokens`.
- `conversers.py`: local HuggingFace model loading is removed, so any model
  name is sent to the OpenAI API.
- fastchat is replaced by `common.Conversation`, which builds the same OpenAI
  message lists; the OpenAI retry loop and the attack loop are unchanged.
- Sibling imports are package-relative (`from . import common`).
