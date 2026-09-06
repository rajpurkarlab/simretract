"""SimRetract: simulation-trained tool hand-over policies for a wrist-camera robot arm.

The package is split by process role so that each part runs under the
interpreter that carries its dependencies:

* ``simretract.sim``   Isaac Sim scene, expert recorders, asset builders (sim env)
* ``simretract.data``  frame cache, plane sidecar, augmentation, selections (policy env)
* ``simretract.train`` the single training entry point with model/ablation switches
* ``simretract.serve`` the policy server and the shared plane fit / wire protocol
* ``simretract.eval``  the closed-loop simulation benchmark and its scorer
* ``simretract.real``  real-robot client, mask server and helpers
* ``simretract.cli``   console entry points (environment-agnostic launchers)

``simretract.config`` is the only place that knows where data, checkpoints,
interpreters and hub repositories live.
"""

from simretract.config import repo_root

__all__ = ["__version__", "repo_root"]
__version__ = "1.0.0"
