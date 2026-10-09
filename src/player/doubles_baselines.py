import random
from typing import List, Optional, Tuple
from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.environment.move import Move
from src.environment.move_category import MoveCategory
from src.environment.pokemon import Pokemon
from src.environment.pokemon_type import PokemonType
from src.player.player import Player
from src.player.battle_order import BattleOrder, DefaultBattleOrder, DoubleBattleOrder
from src.data.gen_data import GenData


class RandomDoublesPlayer(Player):
    """A doubles baseline player that chooses uniform random legal orders."""

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        return self.choose_random_singles_move(battle)


class SimpleHeuristicsDoublesPlayer(Player):
    """
    A rule-based heuristic player for doubles battles.
    Considers type matchups, STAB, offensive damage output, targeting high-threat/low-HP
    opponents, and smart switching.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_data = GenData.from_gen(self._format_to_gen(self._format))

    @staticmethod
    def _format_to_gen(battle_format: str) -> int:
        if battle_format.startswith("gen"):
            try:
                return int(battle_format[3])
            except ValueError:
                pass
        return 9

    def _type_multiplier(self, move_type: PokemonType, target: Pokemon) -> float:
        if target is None or target.fainted:
            return 1.0
        chart = self._gen_data.type_chart
        t_name = move_type.name.upper()
        mult = 1.0
        if target.type_1:
            def_t1 = target.type_1.name.upper()
            mult *= chart.get(def_t1, {}).get(t_name, 1.0)
        if target.type_2:
            def_t2 = target.type_2.name.upper()
            mult *= chart.get(def_t2, {}).get(t_name, 1.0)
        return mult

    def _evaluate_move_order(
        self,
        mon: Pokemon,
        move: Move,
        target_pos: int,
        battle: DoubleBattle,
        opponents: List[Optional[Pokemon]],
    ) -> float:
        # Base utility
        if move.category == MoveCategory.STATUS:
            # Protect utility
            if move.id in {"protect", "detect", "spikyshield", "banefulbunker"}:
                # Reward protect if current mon is below 50% HP or threatened
                if mon.current_hp_fraction < 0.5:
                    return 45.0
                return 15.0
            # Tailwind / Trick Room utility
            if move.id == "tailwind":
                return 55.0
            if move.id == "trickroom":
                return 40.0
            return 10.0

        target_mon = None
        if target_pos == DoubleBattle.OPPONENT_1_POSITION:
            target_mon = opponents[0]
        elif target_pos == DoubleBattle.OPPONENT_2_POSITION:
            target_mon = opponents[1]
        elif target_pos == DoubleBattle.EMPTY_TARGET_POSITION:
            # Spread move - targets both foes
            score = 0.0
            for opp in opponents:
                if opp and not opp.fainted:
                    mult = self._type_multiplier(move.type, opp)
                    score += move.base_power * mult * 0.75  # 75% damage for spread
            stab = 1.5 if (move.type in mon.types) else 1.0
            return score * stab

        if target_mon is None or target_mon.fainted:
            return 0.0

        multiplier = self._type_multiplier(move.type, target_mon)
        if multiplier == 0:
            return 0.0

        stab = 1.5 if (move.type in mon.types) else 1.0
        raw_damage_est = move.base_power * multiplier * stab

        # Bonus for targeting low-HP opponents (finishing blow)
        hp_fraction = target_mon.current_hp_fraction
        if hp_fraction < 0.35 and raw_damage_est > 40:
            raw_damage_est *= 1.6
        elif hp_fraction < 0.6:
            raw_damage_est *= 1.2

        return raw_damage_est

    def _best_order_for_mon(
        self,
        slot_idx: int,
        mon: Pokemon,
        battle: DoubleBattle,
        switches: List[Pokemon],
        moves: List[Move],
        can_tera: bool,
    ) -> BattleOrder:
        opponents = battle.opponent_active_pokemon
        best_score = -1.0
        best_order: Optional[BattleOrder] = None

        # Consider switching if heavily threatened
        if mon.current_hp_fraction < 0.2 and switches and random.random() < 0.3:
            best_switch = max(switches, key=lambda p: p.current_hp_fraction)
            return BattleOrder(best_switch)

        targets_map = {
            m: battle.get_possible_showdown_targets(m, mon) for m in moves
        }

        for move in moves:
            possible_targets = targets_map.get(move, [DoubleBattle.EMPTY_TARGET_POSITION])
            for target_pos in possible_targets:
                score = self._evaluate_move_order(mon, move, target_pos, battle, opponents)
                if score > best_score:
                    best_score = score
                    best_order = BattleOrder(move, move_target=target_pos)

        # Fallback if no move scored well
        if best_order is None:
            if moves:
                first_move = moves[0]
                t = targets_map.get(first_move, [DoubleBattle.EMPTY_TARGET_POSITION])[0]
                best_order = BattleOrder(first_move, move_target=t)
            elif switches:
                best_order = BattleOrder(switches[0])
            else:
                best_order = DefaultBattleOrder()

        return best_order

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_singles_move(battle)

        active_mons = battle.active_pokemon
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        force_switch = battle.force_switch

        # Case 1: Forced switch
        if any(force_switch):
            switch_orders = []
            chosen_mons = set()
            for slot_idx, must_switch in enumerate(force_switch):
                if must_switch:
                    valid_switches = [s for s in available_switches[slot_idx] if s not in chosen_mons]
                    if valid_switches:
                        best_sw = max(valid_switches, key=lambda p: p.current_hp_fraction)
                        chosen_mons.add(best_sw)
                        switch_orders.append(BattleOrder(best_sw))
                    elif available_switches[slot_idx]:
                        switch_orders.append(BattleOrder(available_switches[slot_idx][0]))
                    else:
                        switch_orders.append(DefaultBattleOrder())

            if len(switch_orders) == 1:
                return switch_orders[0]
            elif len(switch_orders) == 2:
                return DoubleBattleOrder(first_order=switch_orders[0], second_order=switch_orders[1])

        # Case 2: Standard turn
        orders_per_slot: List[Optional[BattleOrder]] = [None, None]
        for slot_idx in (0, 1):
            mon = active_mons[slot_idx]
            if mon and not mon.fainted:
                moves = available_moves[slot_idx]
                switches = available_switches[slot_idx]
                can_tera = bool(battle.can_tera[slot_idx])
                orders_per_slot[slot_idx] = self._best_order_for_mon(
                    slot_idx, mon, battle, switches, moves, can_tera
                )

        if orders_per_slot[0] and orders_per_slot[1]:
            ord_0 = getattr(orders_per_slot[0], "order", None)
            ord_1 = getattr(orders_per_slot[1], "order", None)
            if (
                isinstance(ord_0, Pokemon)
                and isinstance(ord_1, Pokemon)
                and ord_0.species == ord_1.species
            ):
                alt_switches = [
                    s for s in available_switches[1] if s.species != ord_0.species
                ]
                if alt_switches:
                    orders_per_slot[1] = BattleOrder(alt_switches[0])
                elif available_moves[1]:
                    first_move = available_moves[1][0]
                    t = battle.get_possible_showdown_targets(first_move, active_mons[1])
                    t_idx = t[0] if t else DoubleBattle.EMPTY_TARGET_POSITION
                    orders_per_slot[1] = BattleOrder(first_move, move_target=t_idx)
                else:
                    orders_per_slot[1] = DefaultBattleOrder()

            return DoubleBattleOrder(first_order=orders_per_slot[0], second_order=orders_per_slot[1])
        elif orders_per_slot[0]:
            return DoubleBattleOrder(first_order=orders_per_slot[0])
        elif orders_per_slot[1]:
            return DoubleBattleOrder(second_order=orders_per_slot[1])
        else:
            return self.choose_random_doubles_move(battle)
