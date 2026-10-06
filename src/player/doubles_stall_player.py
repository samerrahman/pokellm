import random
from typing import List, Optional, Tuple, Dict, Any
from src.player.doubles_player import DoublesPlayer
from src.player.battle_order import BattleOrder, DoubleBattleOrder
from src.environment.double_battle import DoubleBattle
from src.environment.move import Move
from src.environment.pokemon import Pokemon

class StallDoublesPlayer(DoublesPlayer):
    """
    A defensive heuristic doubles bot designed for VGC Regulation C.
    Strategy:
    - Protect cycling (protecting one or both slots to scout / burn turns).
    - Status & Disruption: prioritize Spore (sleep), Will-O-Wisp (burn physical threats).
    - Recovery & Support: Pollen Puff targeting low-HP allies, Rage Powder redirection.
    - Bulky switching: switch weakened Pokemon into high-defensive walls (Ting-Lu, Iron Hands, Amoonguss).
    - Safe chip damage: snarl (SpA drops), icy wind / speed control, heavy slam.
    """

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        # Handle single forced switch
        if sum(battle.force_switch) == 1:
            slot_idx = 0 if battle.force_switch[0] else 1
            return self._choose_slot_action(battle, slot_idx=slot_idx)

        order_1 = self._choose_slot_action(battle, slot_idx=0)
        order_2 = self._choose_slot_action(battle, slot_idx=1)

        # Handle switch collisions (voluntary or forced)
        if order_1 and order_2 and order_1.order and order_2.order:
            if isinstance(order_1.order, Pokemon) and isinstance(order_2.order, Pokemon):
                if order_1.order.species == order_2.order.species:
                    avail_switches = [m for m in battle.available_switches[1] if m.species != order_1.order.species]
                    if avail_switches:
                        order_2 = BattleOrder(avail_switches[0])
                    elif battle.available_moves[1]:
                        m = battle.available_moves[1][0]
                        order_2 = BattleOrder(m)
                    else:
                        order_2 = BattleOrder(None)

        return DoubleBattleOrder(first_order=order_1, second_order=order_2)

    def _choose_slot_action(self, battle: DoubleBattle, slot_idx: int) -> BattleOrder:
        # If forced switch
        if battle.force_switch[slot_idx]:
            switches = battle.available_switches[slot_idx]
            if switches:
                # Prioritize high HP bulky switches
                best_switch = max(switches, key=lambda m: (m.current_hp_fraction if m.current_hp_fraction else 1.0) * (m.base_stats.get('hp', 80) + m.base_stats.get('def', 80)))
                return BattleOrder(best_switch)
            return BattleOrder(None)

        # Active mon check
        active_mon = battle.active_pokemon[slot_idx]
        if not active_mon or active_mon.fainted:
            return BattleOrder(None)

        moves = battle.available_moves[slot_idx]
        switches = battle.available_switches[slot_idx]
        ally_slot_idx = 1 - slot_idx
        ally_mon = battle.active_pokemon[ally_slot_idx]

        # 1. Check for Protect/Spiky Shield/Baneful Bunker
        protect_moves = [m for m in moves if "protect" in m.id or "spikyshield" in m.id or "detect" in m.id]
        if protect_moves and random.random() < 0.45:
            # 45% chance to protect if available
            return BattleOrder(protect_moves[0])

        # 2. Check for Disruption: Spore / Sleep Powder
        sleep_moves = [m for m in moves if m.id in ["spore", "sleeppowder", "hypnosis"]]
        if sleep_moves:
            for opp_idx, opp in enumerate(battle.opponent_active_pokemon):
                if opp and not opp.fainted and opp.status is None:
                    # Check grass immunity for spore
                    if "grass" not in [t.name.lower() for t in opp.types if t]:
                        target_id = opp_idx + 1
                        return BattleOrder(sleep_moves[0], move_target=target_id)

        # 3. Check for Will-O-Wisp (burn physical attackers like Dragonite / Ting-Lu / Iron Hands)
        burn_moves = [m for m in moves if m.id == "willowisp"]
        if burn_moves:
            for opp_idx, opp in enumerate(battle.opponent_active_pokemon):
                if opp and not opp.fainted and opp.status is None:
                    if "fire" not in [t.name.lower() for t in opp.types if t]:
                        target_id = opp_idx + 1
                        return BattleOrder(burn_moves[0], move_target=target_id)

        # 4. Check for Ally Healing (Pollen Puff on damaged ally)
        pollen_moves = [m for m in moves if m.id == "pollenpuff"]
        if pollen_moves and ally_mon and not ally_mon.fainted and ally_mon.current_hp_fraction and ally_mon.current_hp_fraction < 0.70:
            # Target ally: ally slot is -1 or -2
            # In Showdown: if slot_idx == 0, ally is -2; if slot_idx == 1, ally is -1
            target_id = -2 if slot_idx == 0 else -1
            return BattleOrder(pollen_moves[0], move_target=target_id)

        # 5. Check for Redirection: Rage Powder / Follow Me
        redirect_moves = [m for m in moves if m.id in ["ragepowder", "followme"]]
        if redirect_moves and ally_mon and not ally_mon.fainted and ally_mon.current_hp_fraction and ally_mon.current_hp_fraction < 0.50:
            return BattleOrder(redirect_moves[0])

        # 6. Check for defensive switch if heavily damaged (< 25% HP) and regenerator or healthy bench available
        if active_mon.current_hp_fraction and active_mon.current_hp_fraction < 0.25 and switches and random.random() < 0.5:
            healthy_switches = [s for s in switches if s.current_hp_fraction and s.current_hp_fraction > 0.6]
            if healthy_switches:
                return BattleOrder(healthy_switches[0])

        # 7. Otherwise, select highest utility or chip damage move
        if moves:
            scored_moves = []
            for m in moves:
                targets = battle.get_possible_showdown_targets(m, active_mon)
                score = m.base_power
                if m.id in ["snarl", "electroweb", "icywind"]:
                    score += 40  # prioritize stat-lowering spread moves
                scored_moves.append((score, m, targets))
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            best_score, best_move, targets = scored_moves[0]
            target_id = targets[0] if targets else None
            # If target can be selected, hit the lowest HP opponent
            if targets and any(t in [1, 2] for t in targets):
                valid_opp_targets = [t for t in targets if t in [1, 2]]
                # Pick target with lowest HP
                def get_opp_hp(t_idx):
                    opp = battle.opponent_active_pokemon[t_idx - 1]
                    return opp.current_hp_fraction if (opp and opp.current_hp_fraction) else 1.0
                target_id = min(valid_opp_targets, key=get_opp_hp)

            return BattleOrder(best_move, move_target=target_id)

        return BattleOrder(None)
