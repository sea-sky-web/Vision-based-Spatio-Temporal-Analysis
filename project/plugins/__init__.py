from __future__ import annotations

from typing import List

from .base import InferencePlugin
from .tracking_plugin import TrackingPlugin


def build_plugins(cfg, device, output_dir: str) -> List[InferencePlugin]:
    """Instantiate inference plugins declared in the config.

    To keep backwards compatibility, declaring TRACKER in the config automatically
    registers the tracking plugin even if RUNTIME.PLUGINS is absent.
    """

    plugin_names = cfg.get('RUNTIME', {}).get('PLUGINS')
    if plugin_names is None:
        plugin_names = ['tracking'] if cfg.get('TRACKER') else []

    plugins: List[InferencePlugin] = []
    tracker_cfg = cfg.get('TRACKER', {})

    for name in plugin_names:
        name_lower = str(name).lower()
        if name_lower == 'tracking':
            if tracker_cfg.get('ENABLED', True):
                plugins.append(TrackingPlugin(tracker_cfg, device=device, output_dir=output_dir))
        else:
            raise ValueError(f"Unknown inference plugin '{name}'")

    return plugins


__all__ = ['InferencePlugin', 'build_plugins']
