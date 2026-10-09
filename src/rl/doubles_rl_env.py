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
from src.environment.move_category import MoveCategory
from src.data.static.vgc_teams import VGC_REG_C_TEAM_1, VGC_REG_C_TEAM_2
from src.rl.damage_calc import check_ko_threshold_shift, estimate_damage
from src.rl.board_evaluator import evaluate_board_state
from src.rl.vgc_matrix_game import VGCMatrixGameSolver

class NashMatrixDoublesPlayer(DoublesPlayer):
    """Opponent player that solves the 1-step simultaneous matrix game for mixed strategy Nash equilibrium."""
    def teampreview(self, battle: DoubleBattle) -> str:
        return "/team 1234"

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        order, _ = VGCMatrixGameSolver.solve_turn(battle)
        return order

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

        # 2. Reward for stat boosts on premade sets + (n-1) Hit-KO threshold shift bonus:
        # Since teams use tournament-standard competitive sets, all positive stat boosts are advantageous.
        # Furthermore, if an offensive boost shifts an opposing Pokemon into an (n-1) hit-KO range (e.g. 2HKO -> OHKO),
        # an additional tactical KO threshold reward is granted.
        boost_reward = 0.0
        ko_threshold_reward = 0.0

        for slot_idx in (0, 1):
            mon = battle.active_pokemon[slot_idx]
            if mon and not mon.fainted and hasattr(mon, "boosts"):
                for stat_name, boost_stage in mon.boosts.items():
                    if boost_stage > 0:
                        # Base reward for accumulating positive boosts (+0.04 per stage)
                        boost_reward += 0.04 * boost_stage

                        # If offensive boost, check if it enables (n-1) hit-KO range against active opponents
                        if stat_name.lower() in ("atk", "spa"):
                            available_moves = battle.available_moves[slot_idx] if slot_idx < len(battle.available_moves) else []
                            for move in available_moves:
                                if move.category != MoveCategory.STATUS and move.base_power > 0:
                                    is_physical = (move.category == MoveCategory.PHYSICAL)
                                    if (is_physical and stat_name.lower() == "atk") or (not is_physical and stat_name.lower() == "spa"):
                                        for opp in battle.opponent_active_pokemon:
                                            if opp and not opp.fainted:
                                                is_spread = (move.target in ("allAdjacentFoes", "allAdjacent"))
                                                shift = check_ko_threshold_shift(mon, opp, move, is_spread=is_spread)
                                                if shift["improved"]:
                                                    # Bonus: +0.10 for shifting an opponent to (n-1) hit KO range (+0.15 if shifting directly to OHKO)
                                                    ko_threshold_reward += 0.15 if shift["is_ohko_now"] else 0.10
                                                    break # avoid overcounting multiple moves on the same opponent
        
        step_reward += (boost_reward + ko_threshold_reward)

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

        # 4. Positional Board Advantage Delta (Speed control, HP pressure, Threat advantage)
        current_board_eval = evaluate_board_state(battle)
        if getattr(self, "last_board_eval", None) is not None:
            board_eval_delta = (current_board_eval - self.last_board_eval) * 0.01
        else:
            board_eval_delta = 0.0
        self.last_board_eval = current_board_eval
        step_reward += board_eval_delta

        step_reward += supereffective_reward

        self.episode_transitions.append({
            "state_prompt": state_prompt,
            "compact_prompt": action_info.get("compact_prompt", ""),
            "chosen_action_idx": action_info.get("chosen_action_idx", 0),
            "candidate_token_ids": action_info.get("candidate_token_ids", []),
            "action_text": action_info.get("action_text", ""),
            "step_reward": step_reward,
            "boost_reward": boost_reward,
            "ko_threshold_reward": ko_threshold_reward,
            "board_eval_delta": board_eval_delta,
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
        
        # Maintain persistent player instances on the Showdown server
        uid = int(time.time() * 10) % 10000
        bot_name = f"aquaspaghetti{uid:04d}"
        opp_name = f"Opponent{uid:04d}"

        self.agent = RLAgentDoublesPlayer(
            policy_fn=None,
            battle_format=self.format_str,
            team=self.team_1,
            account_configuration=AccountConfiguration(bot_name, ""),
            max_concurrent_battles=1
        )

        if self.opponent_type == "stall":
            self.opp = StallDoublesPlayer(battle_format=self.format_str, team=self.team_2, account_configuration=AccountConfiguration(opp_name, ""), max_concurrent_battles=1)
        elif self.opponent_type == "heuristics":
            self.opp = SimpleHeuristicsDoublesPlayer(battle_format=self.format_str, team=self.team_2, account_configuration=AccountConfiguration(opp_name, ""), max_concurrent_battles=1)
        elif self.opponent_type == "nash":
            self.opp = NashMatrixDoublesPlayer(battle_format=self.format_str, team=self.team_2, account_configuration=AccountConfiguration(opp_name, ""), max_concurrent_battles=1)
        else:
            self.opp = RandomDoublesPlayer(battle_format=self.format_str, team=self.team_2, account_configuration=AccountConfiguration(opp_name, ""), max_concurrent_battles=1)
        
        self.opp.teampreview = lambda b: "/team 1234"

    async def run_episode(self, policy_fn) -> Dict[str, Any]:
        """Runs a single complete match using persistent connected players and returns trajectory."""
        self.agent.policy_fn = policy_fn
        self.agent.episode_transitions = []
        self.agent.last_opp_fainted_count = 0
        self.agent.last_my_fainted_count = 0
        self.agent.last_board_eval = None

        # Run 1 match against persistent opponent
        await self.agent.battle_against(self.opp, n_battles=1)

        # Get the latest battle outcome
        latest_battle = list(self.agent.battles.values())[-1]
        won = bool(latest_battle.won)
        turns = latest_battle.turn

        terminal_reward = 1.0 if won else -1.0
        
        transitions = self.agent.episode_transitions
        if transitions:
            transitions[-1]["step_reward"] += terminal_reward

        return {
            "won": won,
            "turns": turns,
            "transitions": transitions,
            "total_reward": sum(t["step_reward"] for t in transitions) if transitions else terminal_reward
        }
