import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from src.data.gen_data import GenData
from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.environment.move import Move
from src.environment.move_category import MoveCategory
from src.environment.pokemon import Pokemon
from src.environment.pokemon_type import PokemonType
from src.player.battle_order import BattleOrder, DefaultBattleOrder, DoubleBattleOrder
from src.player.player import Player

DOUBLES_SYSTEM_PROMPT = """You are an expert competitive Pokémon VGC player. Your goal is to win the Doubles battle.
You control 2 Pokémon on the field simultaneously. Each turn, evaluate:
1. Threat priorities and positioning.
2. Speed control (Tailwind, Trick Room) and Protect turns.
3. Offensive type matchups and focus-fire / spread damage.
Always output your decision in valid JSON matching the instructed schema.
"""


class DoublesPlayer(Player):
    """
    Base class for LLM-based Doubles / VGC players.
    Handles Doubles state translation, prompt formatting with Knowledge-Augmentation (KAG),
    and robust JSON action parsing into DoubleBattleOrder.
    """

    def __init__(
        self,
        *args,
        reason_algo: str = "io",
        T: int = 1,
        knowledge: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reason_algo = reason_algo
        self.T = T
        self.knowledge = knowledge
        self._gen_data = GenData.from_format(self._format)
        self.last_thought: str = ""
        self.last_state_prompt: str = ""

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

    def _describe_pokemon(self, mon: Optional[Pokemon], is_opponent: bool = False) -> str:
        if mon is None or mon.fainted:
            return "Empty / Fainted"

        types = mon.type_1.name if mon.type_1 else "Normal"
        if mon.type_2:
            types += f"/{mon.type_2.name}"

        status_str = f", Status: {mon.status.name}" if mon.status else ""
        boosts_str = ""
        if hasattr(mon, "_boosts"):
            active_boosts = [f"{k}:+{v}" if v > 0 else f"{k}:{v}" for k, v in mon._boosts.items() if v != 0]
            if active_boosts:
                boosts_str = f", Boosts: [{', '.join(active_boosts)}]"

        hp_pct = round(mon.current_hp_fraction * 100)
        desc = f"{mon.species} (Type: {types}, HP: {hp_pct}%{status_str}{boosts_str})"

        # Known moves for opponents
        if is_opponent and mon.moves:
            known = [m.id for m in mon.moves.values()]
            desc += f", Known moves: [{', '.join(known)}]"

        return desc

    def _format_field_conditions(self, battle: DoubleBattle) -> str:
        conditions = []
        if battle.weather:
            w_names = [w.name if hasattr(w, "name") else str(w) for w in battle.weather.keys()]
            conditions.append(f"Weather: {', '.join(w_names)}")
        if battle.fields:
            f_names = [f.name if hasattr(f, "name") else str(f) for f in battle.fields.keys()]
            conditions.append(f"Terrain/Field: {', '.join(f_names)}")
        if battle.side_conditions:
            sc_names = [sc.name if hasattr(sc, "name") else str(sc) for sc in battle.side_conditions.keys()]
            conditions.append(f"Ally Side: {', '.join(sc_names)}")
        if battle.opponent_side_conditions:
            osc_names = [osc.name if hasattr(osc, "name") else str(osc) for osc in battle.opponent_side_conditions.keys()]
            conditions.append(f"Opponent Side: {', '.join(osc_names)}")

        return " | ".join(conditions) if conditions else "None"

    def state_translate(self, battle: DoubleBattle) -> str:
        lines = []

        # Turn header
        lines.append(f"=== Turn {battle.turn} (Doubles) ===")

        # Field conditions
        field_summary = self._format_field_conditions(battle)
        lines.append(f"Field Conditions: {field_summary}")

        # Opponent Team
        opp_active = battle.opponent_active_pokemon
        lines.append("\n[Opponent Active Pokemon]")
        lines.append(f"Opponent Slot 1 (target 1): {self._describe_pokemon(opp_active[0], is_opponent=True)}")
        lines.append(f"Opponent Slot 2 (target 2): {self._describe_pokemon(opp_active[1], is_opponent=True)}")

        opp_bench_count = sum(1 for p in battle.opponent_team.values() if not p.active and not p.fainted)
        lines.append(f"Opponent Bench: {opp_bench_count} unfainted Pokemon remaining")

        # Player Team
        my_active = battle.active_pokemon
        lines.append("\n[Your Active Pokemon]")

        force_switch = battle.force_switch
        any_force = any(force_switch)

        for slot_idx in (0, 1):
            mon = my_active[slot_idx]
            slot_num = slot_idx + 1
            slot_header = f"Your Slot {slot_num}: {self._describe_pokemon(mon)}"
            if force_switch[slot_idx]:
                slot_header += " [MUST SWITCH]"
            lines.append(slot_header)

            if mon and not mon.fainted and not force_switch[slot_idx]:
                moves = battle.available_moves[slot_idx]
                targets_map = {m: battle.get_possible_showdown_targets(m, mon) for m in moves}

                move_descriptions = []
                for m in moves:
                    possible_targets = targets_map.get(m, [DoubleBattle.EMPTY_TARGET_POSITION])
                    target_str = []
                    for t in possible_targets:
                        if t == DoubleBattle.OPPONENT_1_POSITION:
                            target_str.append("1 (Opponent Slot 1)")
                        elif t == DoubleBattle.OPPONENT_2_POSITION:
                            target_str.append("2 (Opponent Slot 2)")
                        elif t in (DoubleBattle.POKEMON_1_POSITION, DoubleBattle.POKEMON_2_POSITION):
                            target_str.append(f"{t} (Ally)")
                        elif t == DoubleBattle.EMPTY_TARGET_POSITION:
                            target_str.append("null (Spread / Self / All)")

                    # KAG Matchup insights
                    matchups = []
                    if self.knowledge and m.category != MoveCategory.STATUS:
                        for opp_idx, opp_mon in enumerate(opp_active):
                            if opp_mon and not opp_mon.fainted:
                                mult = self._type_multiplier(m.type, opp_mon)
                                if mult > 1.0:
                                    matchups.append(f"vs Opp{opp_idx+1}: {mult}x Super-Effective")
                                elif mult == 0.0:
                                    matchups.append(f"vs Opp{opp_idx+1}: 0x Immune")

                    matchup_note = f" ({', '.join(matchups)})" if matchups else ""
                    move_descriptions.append(
                        f"  - {m.id} | Type: {m.type.name} | Pow: {m.base_power} | Cat: {m.category.name} | Valid Targets: [{', '.join(target_str)}]{matchup_note}"
                    )

                if move_descriptions:
                    lines.append("  Available Moves:")
                    lines.extend(move_descriptions)

            switches = battle.available_switches[slot_idx]
            if switches:
                sw_desc = [f"{p.species} ({round(p.current_hp_fraction * 100)}% HP)" for p in switches]
                lines.append(f"  Available Switches: [{', '.join(sw_desc)}]")

        # Instructions based on turn type
        lines.append("\n[Instructions]")
        if any_force:
            lines.append("Forced switch required. Output a JSON object specifying the switch for the required slot(s):")
            lines.append(
                '{\n  "thought": "<reason>",\n  "slot_1": {"action": "switch", "name": "<pokemon_name>"}\n}'
            )
        else:
            lines.append(
                "Output your action for both slots as a valid JSON object matching this schema:"
            )
            lines.append(
                '{\n'
                '  "thought": "<reasoning for target choice and strategy>",\n'
                '  "slot_1": {"action": "move", "name": "<move_id>", "target": 1, "terastallize": false},\n'
                '  "slot_2": {"action": "move", "name": "<move_id>", "target": null, "terastallize": false}\n'
                '}'
            )
            lines.append(
                'Note: Use target: 1 for Opponent 1, target: 2 for Opponent 2, or null for spread/self moves. '
                'For switches use: {"action": "switch", "name": "<pokemon_name>"}.'
            )

        return "\n".join(lines)

    def parse(self, llm_output: str, battle: DoubleBattle) -> Tuple[BattleOrder, str]:
        """Parse LLM JSON response into a legal DoubleBattleOrder."""
        thought = ""
        try:
            json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON found in LLM output: {llm_output}")

            data = json.loads(json_match.group(0))
            thought = data.get("thought", "")

            # Forced switch handling
            force_switch = battle.force_switch
            if any(force_switch):
                switch_orders = []
                for slot_idx, must_switch in enumerate(force_switch):
                    if must_switch:
                        slot_key = f"slot_{slot_idx + 1}"
                        chosen_mon_name = None
                        if slot_key in data:
                            chosen_mon_name = data[slot_key].get("name", "").lower()
                        elif "switch" in data:
                            chosen_mon_name = str(data["switch"]).lower()

                        matched_pokemon = None
                        for p in battle.available_switches[slot_idx]:
                            if p.species.lower() == chosen_mon_name:
                                matched_pokemon = p
                                break

                        if matched_pokemon:
                            switch_orders.append(BattleOrder(matched_pokemon))
                        elif battle.available_switches[slot_idx]:
                            switch_orders.append(BattleOrder(battle.available_switches[slot_idx][0]))
                        else:
                            switch_orders.append(DefaultBattleOrder())

                if len(switch_orders) == 1:
                    return switch_orders[0], thought
                elif len(switch_orders) == 2:
                    return DoubleBattleOrder(first_order=switch_orders[0], second_order=switch_orders[1]), thought

            # Standard 2-slot turn
            orders: List[Optional[BattleOrder]] = [None, None]
            my_active = battle.active_pokemon

            for slot_idx in (0, 1):
                mon = my_active[slot_idx]
                if mon is None or mon.fainted:
                    continue

                slot_key = f"slot_{slot_idx + 1}"
                slot_data = data.get(slot_key, {})
                action_type = slot_data.get("action", "move").lower()

                if action_type == "switch":
                    target_species = slot_data.get("name", "").lower()
                    matched_mon = None
                    for p in battle.available_switches[slot_idx]:
                        if p.species.lower() == target_species:
                            matched_mon = p
                            break
                    if matched_mon:
                        orders[slot_idx] = BattleOrder(matched_mon)
                    elif battle.available_switches[slot_idx]:
                        orders[slot_idx] = BattleOrder(battle.available_switches[slot_idx][0])

                if orders[slot_idx] is None:  # Default or chosen move
                    chosen_move_id = slot_data.get("name", "").lower().replace(" ", "").replace("-", "")
                    matched_move = None
                    for m in battle.available_moves[slot_idx]:
                        if m.id.lower() == chosen_move_id:
                            matched_move = m
                            break

                    if matched_move is None and battle.available_moves[slot_idx]:
                        matched_move = battle.available_moves[slot_idx][0]

                    if matched_move:
                        targets = battle.get_possible_showdown_targets(matched_move, mon)
                        req_target = slot_data.get("target")
                        target_pos = DoubleBattle.EMPTY_TARGET_POSITION

                        if req_target in targets:
                            target_pos = req_target
                        elif targets:
                            target_pos = targets[0]

                        tera = bool(slot_data.get("terastallize", False)) and bool(battle.can_tera[slot_idx])
                        orders[slot_idx] = BattleOrder(
                            matched_move, move_target=target_pos, terastallize=tera
                        )

            # Prevent switching both slots to the same bench Pokemon
            if (
                orders[0]
                and orders[1]
                and isinstance(orders[0].order, Pokemon)
                and isinstance(orders[1].order, Pokemon)
                and orders[0].order == orders[1].order
            ):
                alt_switches = [s for s in battle.available_switches[1] if s != orders[0].order]
                if alt_switches:
                    orders[1] = BattleOrder(alt_switches[0])
                elif battle.available_moves[1]:
                    first_move = battle.available_moves[1][0]
                    t = battle.get_possible_showdown_targets(first_move, my_active[1])[0]
                    orders[1] = BattleOrder(first_move, move_target=t)

            final_order = None
            if orders[0] and orders[1]:
                final_order = DoubleBattleOrder(first_order=orders[0], second_order=orders[1])
            elif orders[0]:
                final_order = DoubleBattleOrder(first_order=orders[0])
            elif orders[1]:
                final_order = DoubleBattleOrder(second_order=orders[1])

            if final_order:
                return final_order, thought

        except Exception as e:
            # Fallback gracefully on parsing error
            pass

        # Safe fallback
        fallback = self.choose_random_doubles_move(battle)
        return fallback, thought

    def order_to_json(self, order: BattleOrder, battle: DoubleBattle) -> str:
        """Serialize a DoubleBattleOrder or BattleOrder into JSON matching the schema."""
        action_dict: Dict[str, Any] = {"thought": "Strategic doubles action"}

        if any(battle.force_switch):
            # Forced switch
            if isinstance(order, BattleOrder) and isinstance(order.order, Pokemon):
                action_dict["slot_1"] = {"action": "switch", "name": order.order.species}
            elif isinstance(order, DoubleBattleOrder):
                if order.first_order and isinstance(order.first_order.order, Pokemon):
                    action_dict["slot_1"] = {"action": "switch", "name": order.first_order.order.species}
                if order.second_order and isinstance(order.second_order.order, Pokemon):
                    action_dict["slot_2"] = {"action": "switch", "name": order.second_order.order.species}
            return json.dumps(action_dict)

        def _sub_order_to_dict(sub_order: Optional[BattleOrder]) -> Optional[Dict[str, Any]]:
            if sub_order is None or sub_order.order is None:
                return None
            if isinstance(sub_order.order, Move):
                target_val = sub_order.move_target if sub_order.move_target != DoubleBattle.EMPTY_TARGET_POSITION else None
                return {
                    "action": "move",
                    "name": sub_order.order.id,
                    "target": target_val,
                    "terastallize": sub_order.terastallize,
                }
            elif isinstance(sub_order.order, Pokemon):
                return {
                    "action": "switch",
                    "name": sub_order.order.species,
                }
            return None

        if isinstance(order, DoubleBattleOrder):
            dict_1 = _sub_order_to_dict(order.first_order)
            dict_2 = _sub_order_to_dict(order.second_order)
            if dict_1:
                action_dict["slot_1"] = dict_1
            if dict_2:
                action_dict["slot_2"] = dict_2
        elif isinstance(order, BattleOrder):
            dict_1 = _sub_order_to_dict(order)
            if dict_1:
                action_dict["slot_1"] = dict_1

        return json.dumps(action_dict)
