from .ddpg import ddpg_update
from .misc import value_update, temporal_difference

__all__ = [
    'ddpg_update',
    'value_update',
    'temporal_difference'
]
