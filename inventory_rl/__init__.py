from gymnasium.envs.registration import register

from inventory_rl.env import CONFIGS, InventoryConfig, InventoryEnv

register(id="InventoryControl-v0", entry_point="inventory_rl.env:InventoryEnv")

__all__ = ["CONFIGS", "InventoryConfig", "InventoryEnv"]
