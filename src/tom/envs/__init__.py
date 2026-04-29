"""Lazy re-exports — only the env you actually use needs to be importable.

(ViZDoom envs require the ``vizdoom`` package; Overcooked envs require
``overcooked-ai``. We keep imports lazy so a project venv that has only one
of those still works.)
"""
from importlib import import_module

_LAZY = {
    "VizDoomMultiAgentEnv": ("tom.envs.vizdoom_multi", "VizDoomMultiAgentEnv"),
    "VecVizDoomMultiAgentEnv": ("tom.envs.vec_vizdoom", "VecVizDoomMultiAgentEnv"),
    "OvercookedMultiAgentEnv": ("tom.envs.overcooked_multi", "OvercookedMultiAgentEnv"),
    "VecOvercookedEnv": ("tom.envs.overcooked_multi", "VecOvercookedEnv"),
    "HanabiMultiAgentEnv": ("tom.envs.hanabi_multi", "HanabiMultiAgentEnv"),
    "VecHanabiEnv": ("tom.envs.hanabi_multi", "VecHanabiEnv"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        return getattr(import_module(mod_name), attr)
    raise AttributeError(name)
