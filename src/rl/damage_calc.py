"""
Competitive Pokemon Damage & KO Probability Calculator for VGC (Gen 9 Doubles).
Computes exact damage ranges, expected hits-to-KO, and threshold changes.
"""

import math
from typing import Dict, Any, Optional, Tuple, List
from src.environment.move import Move
from src.environment.move_category import MoveCategory
from src.environment.pokemon import Pokemon
from src.environment.double_battle import DoubleBattle

def estimate_damage(
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
    is_spread: bool = False,
    boost_override: Optional[Dict[str, int]] = None
) -> Tuple[int, int, float]:
    """
    Computes (min_damage, max_damage, avg_damage) according to Gen 9 formula.
    Supports boost_override to test counterfactual scenarios (e.g. before vs after +1 Atk/SpA).
    """
    if not move or move.category == MoveCategory.STATUS or move.base_power == 0:
        return (0, 0, 0.0)

    level = attacker.level or 50
    base_power = move.base_power

    # Determine offensive & defensive stats
    if move.category == MoveCategory.PHYSICAL:
        stat_name = "atk"
        def_stat_name = "def"
    else:
        stat_name = "spa"
        def_stat_name = "spd"

    # Base actual stats
    atk_val = (attacker.stats or {}).get(stat_name)
    if atk_val is None:
        atk_val = attacker.base_stats.get(stat_name, 100)

    def_val = (defender.stats or {}).get(def_stat_name)
    if def_val is None:
        def_val = defender.base_stats.get(def_stat_name, 100)

    # Boost multiplier
    atk_boosts = dict(attacker.boosts) if hasattr(attacker, "boosts") else {}
    if boost_override:
        atk_boosts.update(boost_override)

    stage = atk_boosts.get(stat_name, 0)
    if stage >= 0:
        atk_mult = (2 + stage) / 2.0
    else:
        atk_mult = 2.0 / (2 - stage)

    def_stage = defender.boosts.get(def_stat_name, 0) if hasattr(defender, "boosts") else 0
    if def_stage >= 0:
        def_mult = (2 + def_stage) / 2.0
    else:
        def_mult = 2.0 / (2 - def_stage)

    a = max(1.0, atk_val * atk_mult)
    d = max(1.0, def_val * def_mult)

    # Core base damage formula
    base_dmg = math.floor(math.floor(math.floor(2 * level / 5 + 2) * base_power * a / d) / 50) + 2

    # Doubles spread move penalty (0.75x in doubles when targeting multiple targets)
    if is_spread:
        base_dmg = math.floor(base_dmg * 0.75)

    # STAB (Same Type Attack Bonus)
    type_chart = defender._data.type_chart if hasattr(defender, "_data") else {}
    stab = 1.5 if (move.type and attacker.types and move.type in attacker.types) else 1.0

    # Type effectiveness
    type_mult = 1.0
    m_type = move.type.name.upper() if hasattr(move.type, "name") else str(move.type).upper()
    if defender.type_1:
        t1 = defender.type_1.name.upper()
        type_mult *= type_chart.get(t1, {}).get(m_type, 1.0)
    if defender.type_2:
        t2 = defender.type_2.name.upper()
        type_mult *= type_chart.get(t2, {}).get(m_type, 1.0)

    if type_mult == 0.0:
        return (0, 0, 0.0)

    # Modifier without random roll: base_dmg * STAB * type_mult
    mod_dmg = math.floor(base_dmg * stab * type_mult)

    # Damage roll is 85% to 100% (integers 85..100 divided by 100)
    min_dmg = math.floor(mod_dmg * 0.85)
    max_dmg = math.floor(mod_dmg * 1.00)
    avg_dmg = (min_dmg + max_dmg) / 2.0

    return (min_dmg, max_dmg, avg_dmg)

def hits_to_ko(min_dmg: int, max_dmg: int, current_hp: int) -> float:
    """
    Estimates the number of hits required to guarantee / realistically achieve a KO.
    Returns float (e.g. 1.0 for OHKO, 2.0 for 2HKO, etc.).
    """
    if max_dmg <= 0 or current_hp <= 0:
        return 99.0
    if min_dmg >= current_hp:
        return 1.0
    # Average hits needed
    avg_dmg = (min_dmg + max_dmg) / 2.0
    return max(1.0, current_hp / avg_dmg)

def check_ko_threshold_shift(
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
    is_spread: bool = False
) -> Dict[str, Any]:
    """
    Checks if a +1 boost on the attacking stat shifts the KO range from n-hits to (n-1)-hits.
    For example: 2HKO -> OHKO, or 3HKO -> 2HKO.
    """
    curr_hp = defender.current_hp or defender.max_hp or 100
    
    stat_name = "atk" if move.category == MoveCategory.PHYSICAL else "spa"
    curr_stage = attacker.boosts.get(stat_name, 0) if hasattr(attacker, "boosts") else 0

    # 1. Damage without extra boost
    min_0, max_0, avg_0 = estimate_damage(attacker, defender, move, is_spread=is_spread)
    hits_before = hits_to_ko(min_0, max_0, curr_hp)

    # 2. Damage with hypothetical +1 boost
    min_1, max_1, avg_1 = estimate_damage(attacker, defender, move, is_spread=is_spread, boost_override={stat_name: curr_stage + 1})
    hits_after = hits_to_ko(min_1, max_1, curr_hp)

    # Check integer threshold cross (e.g., ceil(hits_before) > ceil(hits_after))
    n_before = math.ceil(hits_before)
    n_after = math.ceil(hits_after)

    improved = n_after < n_before

    return {
        "improved": improved,
        "n_before": n_before,
        "n_after": n_after,
        "is_ohko_now": (min_1 >= curr_hp or hits_after <= 1.0),
        "min_dmg_after": min_1,
        "max_dmg_after": max_1,
        "defender_hp": curr_hp
    }

def estimate_double_target_damage(
    attacker_1: Pokemon,
    move_1: Move,
    attacker_2: Pokemon,
    move_2: Move,
    defender: Pokemon,
    defender_protecting: bool = False
) -> Dict[str, Any]:
    """
    Evaluates combined focus-fire from both active Pokémon on a single target.
    Accounts for Protect / Spiky Shield negation and calculates joint KO probability.
    """
    if defender_protecting:
        return {
            "total_min_dmg": 0,
            "total_max_dmg": 0,
            "total_avg_dmg": 0.0,
            "is_ko": False,
            "protected": True
        }

    curr_hp = defender.current_hp or defender.max_hp or 100

    min_1, max_1, avg_1 = estimate_damage(attacker_1, defender, move_1) if attacker_1 and move_1 else (0, 0, 0.0)
    min_2, max_2, avg_2 = estimate_damage(attacker_2, defender, move_2) if attacker_2 and move_2 else (0, 0, 0.0)

    total_min = min_1 + min_2
    total_max = max_1 + max_2
    total_avg = avg_1 + avg_2

    is_ko = (total_min >= curr_hp) or (total_avg >= curr_hp * 0.95)

    return {
        "total_min_dmg": total_min,
        "total_max_dmg": total_max,
        "total_avg_dmg": total_avg,
        "is_ko": is_ko,
        "protected": False,
        "defender_hp": curr_hp
    }

