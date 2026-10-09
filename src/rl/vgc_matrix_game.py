"""
Game-Theoretic Simultaneous Matrix Game Solver for VGC Doubles.
Solves for the mixed-strategy Nash equilibrium across the top legal joint moves
of both players in a single simultaneous turn.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.environment.double_battle import DoubleBattle
from src.environment.pokemon import Pokemon
from src.environment.move import Move
from src.environment.move_category import MoveCategory
from src.player.battle_order import BattleOrder, DoubleBattleOrder, DefaultBattleOrder
from src.rl.damage_calc import estimate_damage, estimate_double_target_damage
from src.rl.board_evaluator import evaluate_board_state

def solve_zero_sum_matrix(payoff_matrix: np.ndarray, max_iterations: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes mixed strategy Nash Equilibrium for a 2-player zero-sum game
    using Fictitious Play (Brown's algorithm).
    Guarantees convergence to Nash Equilibrium in finite zero-sum normal form games.
    Returns: (row_strategy, col_strategy, game_value)
    """
    m, n = payoff_matrix.shape
    if m == 1 and n == 1:
        return np.array([1.0]), np.array([1.0]), float(payoff_matrix[0, 0])

    row_counts = np.zeros(m)
    col_counts = np.zeros(n)

    # Initial arbitrary actions
    row_action = np.argmax(np.mean(payoff_matrix, axis=1))
    col_action = np.argmin(np.mean(payoff_matrix, axis=0))

    row_counts[row_action] += 1
    col_counts[col_action] += 1

    row_payoff_history = payoff_matrix[:, col_action].copy()
    col_payoff_history = payoff_matrix[row_action, :].copy()

    for _ in range(1, max_iterations):
        # Best responses against empirical frequency of opponent
        row_action = np.argmax(row_payoff_history)
        col_action = np.argmin(col_payoff_history)

        row_counts[row_action] += 1
        col_counts[col_action] += 1

        row_payoff_history += payoff_matrix[:, col_action]
        col_payoff_history += payoff_matrix[row_action, :]

    row_strategy = row_counts / np.sum(row_counts)
    col_strategy = col_counts / np.sum(col_counts)
    game_value = float(row_strategy @ payoff_matrix @ col_strategy)

    return row_strategy, col_strategy, game_value

class VGCMatrixGameSolver:
    """
    Generates candidate joint actions for both players, computes the payoff matrix
    via 1-step tactical transition evaluation, and solves for the Nash optimal mixed strategy.
    """

    @staticmethod
    def get_candidate_slot_orders(mon: Pokemon, slot_idx: int, battle: DoubleBattle, is_ally: bool = True) -> List[BattleOrder]:
        """Generates viable high-priority single orders for a Pokémon slot."""
        if not mon or mon.fainted:
            return [BattleOrder(None)]

        moves = battle.available_moves[slot_idx] if is_ally else list(mon.moves.values())
        switches = battle.available_switches[slot_idx] if is_ally else [
            p for p in battle.opponent_team.values() if not p.fainted and not p.active
        ]

        candidate_orders: List[BattleOrder] = []

        # 1. Protect / Detect moves (essential for scouting and stalling)
        for m in moves:
            if m.id in {"protect", "detect", "spikyshield", "banefulbunker"}:
                candidate_orders.append(BattleOrder(m))

        # 2. Damaging moves with targets
        opponents = battle.opponent_active_pokemon if is_ally else battle.active_pokemon
        for m in moves:
            if m.category != MoveCategory.STATUS:
                # Target opponent 1
                if len(opponents) > 0 and opponents[0] and not opponents[0].fainted:
                    candidate_orders.append(BattleOrder(m, move_target=DoubleBattle.OPPONENT_1_POSITION))
                # Target opponent 2
                if len(opponents) > 1 and opponents[1] and not opponents[1].fainted:
                    candidate_orders.append(BattleOrder(m, move_target=DoubleBattle.OPPONENT_2_POSITION))
                # Spread move
                if m.target in ("allAdjacentFoes", "allAdjacent"):
                    candidate_orders.append(BattleOrder(m, move_target=DoubleBattle.EMPTY_TARGET_POSITION))

        # 3. Tactical switches (e.g., pivot to highest HP reserve)
        if switches:
            best_reserve = max(switches, key=lambda p: p.current_hp_fraction if hasattr(p, "current_hp_fraction") else 1.0)
            candidate_orders.append(BattleOrder(best_reserve))

        # Ensure at least 1 order
        if not candidate_orders:
            candidate_orders.append(BattleOrder(moves[0]) if moves else DefaultBattleOrder())

        return candidate_orders[:6] # Top 6 orders per slot

    @classmethod
    def get_joint_candidates(cls, battle: DoubleBattle, is_ally: bool = True) -> List[DoubleBattleOrder]:
        """Generates combined 2-slot joint battle orders."""
        active_mons = battle.active_pokemon if is_ally else battle.opponent_active_pokemon
        slot_0_orders = cls.get_candidate_slot_orders(active_mons[0] if len(active_mons) > 0 else None, 0, battle, is_ally)
        slot_1_orders = cls.get_candidate_slot_orders(active_mons[1] if len(active_mons) > 1 else None, 1, battle, is_ally)

        joint_orders: List[DoubleBattleOrder] = []
        for o1 in slot_0_orders:
            for o2 in slot_1_orders:
                # Prevent switching to the same reserve mon simultaneously
                if o1.order and o2.order and hasattr(o1.order, "species") and hasattr(o2.order, "species"):
                    if o1.order.species == o2.order.species:
                        continue
                joint_orders.append(DoubleBattleOrder(first_order=o1, second_order=o2))

        # Cap at top 12 joint options to maintain sub-100ms matrix solve time
        return joint_orders[:12] if joint_orders else [DoubleBattleOrder()]

    @classmethod
    def evaluate_joint_action_payoff(
        cls,
        battle: DoubleBattle,
        ally_action: DoubleBattleOrder,
        opp_action: DoubleBattleOrder
    ) -> float:
        """
        Simulates expected payoff of an (ally_action, opp_action) pair.
        Considers damage dealt, damage taken, double-target synergy, and Protect interaction.
        """
        base_score = evaluate_board_state(battle)
        payoff = base_score

        ally_mons = battle.active_pokemon
        opp_mons = battle.opponent_active_pokemon

        # Check if opponent used Protect
        opp_protecting = [False, False]
        if opp_action.first_order and opp_action.first_order.order and hasattr(opp_action.first_order.order, "id"):
            if opp_action.first_order.order.id in {"protect", "detect", "spikyshield"}:
                opp_protecting[0] = True
        if opp_action.second_order and opp_action.second_order.order and hasattr(opp_action.second_order.order, "id"):
            if opp_action.second_order.order.id in {"protect", "detect", "spikyshield"}:
                opp_protecting[1] = True

        # Ally attacks
        for slot_idx, order in enumerate([ally_action.first_order, ally_action.second_order]):
            if order and order.order and isinstance(order.order, Move) and order.order.category != MoveCategory.STATUS:
                target_pos = order.move_target
                target_idx = 0 if target_pos == DoubleBattle.OPPONENT_1_POSITION else (1 if target_pos == DoubleBattle.OPPONENT_2_POSITION else -1)
                
                if target_idx in (0, 1) and target_idx < len(opp_mons) and opp_mons[target_idx]:
                    if opp_protecting[target_idx]:
                        payoff -= 3.0  # Penalize attacking into a protected target
                    else:
                        m_d, M_d, avg_d = estimate_damage(ally_mons[slot_idx], opp_mons[target_idx], order.order)
                        target_hp = opp_mons[target_idx].current_hp or 100
                        payoff += min(15.0, (avg_d / target_hp) * 10.0)
                        if M_d >= target_hp:
                            payoff += 8.0  # High bonus for KO

        # Opponent attacks
        for slot_idx, order in enumerate([opp_action.first_order, opp_action.second_order]):
            if order and order.order and isinstance(order.order, Move) and order.order.category != MoveCategory.STATUS:
                target_pos = order.move_target
                target_idx = 0 if target_pos == 1 else (1 if target_pos == 2 else -1)
                if target_idx in (0, 1) and target_idx < len(ally_mons) and ally_mons[target_idx]:
                    # Check ally protect
                    ally_protect = False
                    if target_idx == 0 and ally_action.first_order and getattr(ally_action.first_order.order, "id", "") in {"protect", "detect"}:
                        ally_protect = True
                    elif target_idx == 1 and ally_action.second_order and getattr(ally_action.second_order.order, "id", "") in {"protect", "detect"}:
                        ally_protect = True

                    if ally_protect:
                        payoff += 4.0 # Successfully protected against attack
                    else:
                        m_d, M_d, avg_d = estimate_damage(opp_mons[slot_idx], ally_mons[target_idx], order.order)
                        ally_hp = ally_mons[target_idx].current_hp or 100
                        payoff -= min(15.0, (avg_d / ally_hp) * 10.0)
                        if M_d >= ally_hp:
                            payoff -= 8.0 # Opponent KO threat

        return float(payoff)

    @classmethod
    def solve_turn(cls, battle: DoubleBattle) -> Tuple[DoubleBattleOrder, Dict[str, Any]]:
        """
        Solves the simultaneous matrix game for the current turn.
        Returns: (chosen_order, metadata)
        """
        ally_candidates = cls.get_joint_candidates(battle, is_ally=True)
        opp_candidates = cls.get_joint_candidates(battle, is_ally=False)

        m = len(ally_candidates)
        n = len(opp_candidates)
        payoff_matrix = np.zeros((m, n), dtype=np.float32)

        for i, a_act in enumerate(ally_candidates):
            for j, o_act in enumerate(opp_candidates):
                payoff_matrix[i, j] = cls.evaluate_joint_action_payoff(battle, a_act, o_act)

        ally_dist, opp_dist, val = solve_zero_sum_matrix(payoff_matrix)

        # Sample or argmax from the equilibrium strategy
        # Using temperature sampling over Nash distribution for robust unpredictability
        if np.max(ally_dist) > 0.95:
            chosen_idx = int(np.argmax(ally_dist))
        else:
            probs = ally_dist / np.sum(ally_dist)
            chosen_idx = int(np.random.choice(m, p=probs))

        chosen_order = ally_candidates[chosen_idx]

        return chosen_order, {
            "chosen_index": chosen_idx,
            "nash_strategy": ally_dist.tolist(),
            "game_value": val,
            "matrix_shape": (m, n),
            "candidates_count": m
        }
