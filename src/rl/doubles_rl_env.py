"""
Reinforcement Learning Environment for VGC Doubles battles in Pokemon Showdown.
Wraps Showdown matches into an episodic trajectory collector for Policy Gradient (REINFORCE / PPO).
"""

import asyncio
import time
import random
import torch
from typing import List, Dict, Any, Optional, Tuple
from src.player.doubles_player import DoublesPlayer
from src.player.doubles_stall_player import StallDoublesPlayer
from src.player.doubles_baselines import SimpleHeuristicsDoublesPlayer, RandomDoublesPlayer
from src.player.battle_order import BattleOrder, DoubleBattleOrder, DefaultBattleOrder
from src.client.account_configuration import AccountConfiguration
from src.environment.double_battle import DoubleBattle
from src.data.static.vgc_teams import VGC_REG_C_TEAM_1, VGC_REG_C_TEAM_2

class RLAgentDoublesPlayer(DoublesPlayer):
    """
    Player controlled directly by a learning policy (LLM / Action Policy).
    Stores prompts, actions, log_probs, and rewards across turns of an episode.
    """

    def __init__(self, policy_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_fn = policy_fn
        self.episode_transitions = []
        self.last_opp_fainted_count = 0
        self.last_my_fainted_count = 0

    def teampreview(self, battle: DoubleBattle) -> str:
        return "/team 1234"

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        # Generate state observation prompt
        state_prompt = self.state_translate(battle)

        # Count delta faints for dense intermediate reward
        my_fainted = sum(1 for p in battle.team.values() if p.fainted)
        opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
        
        # Dense reward components:
        # 1. Delta faints: +0.2 per enemy KO, -0.2 per ally KO
        step_reward = (opp_fainted - self.last_opp_fainted_count) * 0.2 - (my_fainted - self.last_my_fainted_count) * 0.2
        self.last_opp_fainted_count = opp_fainted
        self.last_my_fainted_count = my_fainted

        # 2. Reward for stat boosts on stats that matter (threshold: base_stat >= 80 or high damage/speed)
        # Check active Pokemon for positive stat stages (atk, spa, spe, def, spd)
        boost_reward = 0.0
        for slot_idx in (0, 1):
            mon = battle.active_pokemon[slot_idx]
            if mon and not mon.fainted and hasattr(mon, "boosts"):
                base_stats = mon.base_stats or {}
                for stat_name, boost_stage in mon.boosts.items():
                    if boost_stage > 0:
                        # Check if this stat matters for this Pokemon (base stat >= 80)
                        # e.g., atk >= 80 for physical attackers, spa >= 80 for special, spe >= 80 for fast mons
                        stat_key = stat_name.lower()
                        # Map showdown boost names to base stat keys
                        mapped_stat = {"atk": "atk", "def": "def", "spa": "spa", "spd": "spd", "spe": "spe"}.get(stat_key)
                        if mapped_stat and base_stats.get(mapped_stat, 0) >= 80:
                            # Small shaping reward per relevant boost stage
                            boost_reward += 0.05 * boost_stage
        
        step_reward += boost_reward

        if self.policy_fn is not None:
            order, action_info = self.policy_fn(state_prompt, battle)
        else:
            # Fallback random legal order
            order = self.choose_random_doubles_move(battle)
            action_info = {"log_prob": torch.tensor(0.0), "action_text": order.message}

        # 3. Reward for selecting super-effective moves
        # Check if the chosen order targets an opponent with a super-effective move (multiplier > 1.0)
        supereffective_reward = 0.0
        orders_to_check = []
        if isinstance(order, DoubleBattleOrder):
            if order.first_order:
                orders_to_check.append((order.first_order, 0))
            if order.second_order:
                orders_to_check.append((order.second_order, 1))
        elif isinstance(order, BattleOrder):
            orders_to_check.append((order, 0))

        for bo, slot_idx in orders_to_check:
            if bo and bo.order and hasattr(bo.order, "type") and hasattr(bo.order, "category"):
                # Ensure it's a damaging move
                if bo.order.category and bo.order.category.name != "STATUS":
                    target_pos = bo.move_target
                    target_mon = None
                    if target_pos == 1 and battle.opponent_active_pokemon[0]:
                        target_mon = battle.opponent_active_pokemon[0]
                    elif target_pos == 2 and battle.opponent_active_pokemon[1]:
                        target_mon = battle.opponent_active_pokemon[1]
                    elif target_pos in (-1, -2): # spread or opponent field
                        # Target whichever opponent is active
                        for opp in battle.opponent_active_pokemon:
                            if opp and not opp.fainted:
                                target_mon = opp
                                break

                    if target_mon and not target_mon.fainted:
                        type_mult = self._type_multiplier(bo.order.type, target_mon)
                        if type_mult > 1.0:
                            # +0.10 for super-effective (2x), +0.20 for double super-effective (4x)
                            supereffective_reward += 0.05 * type_mult

        step_reward += supereffective_reward

        self.episode_transitions.append({
            "state_prompt": state_prompt,
            "action_text": action_info.get("action_text", ""),
            "log_prob": action_info.get("log_prob", None),
            "step_reward": step_reward,
            "boost_reward": boost_reward,
            "supereffective_reward": supereffective_reward,
            "turn": battle.turn
        })

        return order

class DoublesRLEnv:
    """
    Simulates VGC Regulation C matches against opponent bots (Stall bot, Heuristics bot)
    and collects full episodes for Policy Gradient training.
    """

    def __init__(
        self,
        opponent_type: str = "stall",
        format_str: str = "gen9vgc2023regc",
        team_1: str = VGC_REG_C_TEAM_1,
        team_2: str = VGC_REG_C_TEAM_2,
    ):
        self.format_str = format_str
        self.team_1 = team_1
        self.team_2 = team_2
        self.opponent_type = opponent_type

    def create_opponent(self, uid: int):
        cfg = AccountConfiguration(f"Opp_{uid}", "")
        if self.opponent_type == "stall":
            opp = StallDoublesPlayer(battle_format=self.format_str, team=self.team_2, account_configuration=cfg, max_concurrent_battles=1)
        elif self.opponent_type == "heuristics":
            opp = SimpleHeuristicsDoublesPlayer(battle_format=self.format_str, team=self.team_2, account_configuration=cfg, max_concurrent_battles=1)
        else:
            opp = RandomDoublesPlayer(battle_format=self.format_str, team=self.team_2, account_configuration=cfg, max_concurrent_battles=1)
        
        opp.teampreview = lambda b: "/team 1234"
        return opp

    async def run_episode(self, policy_fn) -> Dict[str, Any]:
        """Runs a single complete match and returns the full trajectory with discounted returns."""
        uid = int(time.time() * 1000) % 1000000
        agent = RLAgentDoublesPlayer(
            policy_fn=policy_fn,
            battle_format=self.format_str,
            team=self.team_1,
            account_configuration=AccountConfiguration(f"RLAgent_{uid}", ""),
            max_concurrent_battles=1
        )
        opp = self.create_opponent(uid)

        await agent.battle_against(opp, n_battles=1)

        # Compute terminal reward
        won = False
        turns = 0
        for b in agent.battles.values():
            won = bool(b.won)
            turns = b.turn

        terminal_reward = 1.0 if won else -1.0
        
        # Assign discounted returns
        transitions = agent.episode_transitions
        if transitions:
            # Add terminal reward to the final step
            transitions[-1]["step_reward"] += terminal_reward

        return {
            "won": won,
            "turns": turns,
            "transitions": transitions,
            "total_reward": sum(t["step_reward"] for t in transitions) if transitions else terminal_reward
        }
