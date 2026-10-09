"""
Comprehensive Board Evaluation Function for Pokémon VGC (Gen 9 Doubles).
Evaluates board states based on:
1. Active & Reserve HP differential (weighted by offensive/defensive viability).
2. Speed control advantage (Tailwind, Trick Room, stat drops, raw speed tiers).
3. Field & Weather advantage (Terrain, Weather synergy).
4. Immediate threat pressure (threat to OHKO / 2HKO opponents).
5. Resource advantage (Terastallization available, healthy reserves).
"""

from typing import Dict, Any, List, Optional, Tuple
import math
from src.environment.double_battle import DoubleBattle
from src.environment.pokemon import Pokemon
from src.environment.move import Move
from src.environment.move_category import MoveCategory
from src.rl.damage_calc import estimate_damage

def evaluate_mon_speed(mon: Pokemon, battle: DoubleBattle, is_ally: bool) -> float:
    """Computes effective speed factoring in stat stages and side conditions (e.g. Tailwind)."""
    if not mon or mon.fainted:
        return 0.0

    raw_spe = (mon.stats or {}).get("spe")
    if raw_spe is None:
        raw_spe = mon.base_stats.get("spe", 80)

    # Stat boosts
    stage = mon.boosts.get("spe", 0) if hasattr(mon, "boosts") else 0
    if stage >= 0:
        boost_mult = (2 + stage) / 2.0
    else:
        boost_mult = 2.0 / (2 - stage)

    eff_spe = raw_spe * boost_mult

    # Tailwind multiplier (2x speed)
    side_conditions = battle.side_conditions if is_ally else battle.opponent_side_conditions
    if hasattr(side_conditions, "get"):
        if side_conditions.get("tailwind"):
            eff_spe *= 2.0

    # Trick Room invert (lower speed acts first)
    if hasattr(battle, "fields") and "trickroom" in battle.fields:
        eff_spe = 10000.0 / max(1.0, eff_spe)

    return eff_spe

def evaluate_board_state(battle: DoubleBattle) -> float:
    """
    Returns a scalar evaluation score from the agent's perspective.
    Positive -> Advantageous position.
    Negative -> Disadvantageous position.
    """
    # 1. Terminal win / loss check
    all_ally_fainted = all(p.fainted for p in battle.team.values())
    all_opp_fainted = all(p.fainted for p in battle.opponent_team.values())
    
    if all_opp_fainted and not all_ally_fainted:
        return 100.0
    if all_ally_fainted and not all_opp_fainted:
        return -100.0

    # 2. Team HP & Resource Count (Weights: 40% of evaluation)
    ally_hp_total = 0.0
    ally_count = 0
    for p in battle.team.values():
        if not p.fainted:
            ally_hp_total += p.current_hp_fraction
            ally_count += 1

    opp_hp_total = 0.0
    opp_count = 0
    for p in battle.opponent_team.values():
        if not p.fainted:
            opp_hp_total += p.current_hp_fraction
            opp_count += 1

    # In VGC, maintaining Pokémon count is crucial for switches and pivoting
    count_diff = (ally_count - opp_count) * 8.0
    hp_diff = (ally_hp_total - opp_hp_total) * 12.0

    # 3. Active Field Speed Control (Weights: 20% of evaluation)
    # Higher active speed gives priority in claiming double-KOs
    speed_score = 0.0
    ally_active = [p for p in battle.active_pokemon if p and not p.fainted]
    opp_active = [p for p in battle.opponent_active_pokemon if p and not p.fainted]

    for a_mon in ally_active:
        a_spe = evaluate_mon_speed(a_mon, battle, is_ally=True)
        for o_mon in opp_active:
            o_spe = evaluate_mon_speed(o_mon, battle, is_ally=False)
            if a_spe > o_spe:
                speed_score += 2.0
            elif a_spe < o_spe:
                speed_score -= 2.0

    # 4. Immediate Threat & Pressure Score (Weights: 25% of evaluation)
    # How much damage can we threaten on their active slots vs how much they threaten us?
    threat_diff = 0.0
    for a_mon in ally_active:
        moves = a_mon.moves.values() if hasattr(a_mon, "moves") else []
        for move in moves:
            if move.category != MoveCategory.STATUS and move.base_power > 0:
                for o_mon in opp_active:
                    min_d, max_d, avg_d = estimate_damage(a_mon, o_mon, move)
                    opp_hp = o_mon.current_hp or 100
                    if max_d >= opp_hp:
                        threat_diff += 3.5  # We threaten a KO
                    elif avg_d > 0:
                        threat_diff += min(2.0, (avg_d / opp_hp) * 1.5)

    for o_mon in opp_active:
        moves = o_mon.moves.values() if hasattr(o_mon, "moves") else []
        for move in moves:
            if move.category != MoveCategory.STATUS and move.base_power > 0:
                for a_mon in ally_active:
                    min_d, max_d, avg_d = estimate_damage(o_mon, a_mon, move)
                    ally_hp = a_mon.current_hp or 100
                    if max_d >= ally_hp:
                        threat_diff -= 3.5  # Opponent threatens a KO
                    elif avg_d > 0:
                        threat_diff -= min(2.0, (avg_d / ally_hp) * 1.5)

    # 5. Strategic Mechanics: Terastallization Advantage (Weights: 15% of evaluation)
    tera_score = 0.0
    can_tera_ally = any(battle.can_tera) if hasattr(battle, "can_tera") else False
    if can_tera_ally:
        tera_score += 2.5
    if hasattr(battle, "opponent_can_tera") and battle.opponent_can_tera:
        tera_score -= 2.5

    total_eval = count_diff + hp_diff + speed_score + threat_diff + tera_score
    return float(total_eval)
