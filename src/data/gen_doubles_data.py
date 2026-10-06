import argparse
import asyncio
import json
import os
import random
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.client.account_configuration import AccountConfiguration
from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.player.battle_order import BattleOrder
from src.player.doubles_baselines import SimpleHeuristicsDoublesPlayer, RandomDoublesPlayer
from src.player.doubles_player import DOUBLES_SYSTEM_PROMPT, DoublesPlayer


class CollectingHeuristicsDoublesPlayer(SimpleHeuristicsDoublesPlayer, DoublesPlayer):
    """
    A heuristic doubles player that records state prompts and selected actions
    for SFT and preference optimization datasets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Map: battle_tag -> list of (turn, prompt, json_output)
        self.history: Dict[str, List[Tuple[int, str, str]]] = {}

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, DoubleBattle):
            return super().choose_move(battle)

        # 1. Generate text prompt using state_translate
        state_prompt = self.state_translate(battle)
        full_prompt = (
            f"[SYSTEM]\n{DOUBLES_SYSTEM_PROMPT}\n\n"
            f"[BATTLE STATE]\n{state_prompt}\n\n"
            f"[ACTION JSON]\nOutput: "
        )

        # 2. Get heuristic decision
        order = super().choose_move(battle)

        # 3. Convert decision to JSON
        json_output = self.order_to_json(order, battle)

        # 4. Record turn
        if battle.battle_tag not in self.history:
            self.history[battle.battle_tag] = []
        self.history[battle.battle_tag].append((battle.turn, full_prompt, json_output))

        return order


async def run_data_generation(
    n_battles: int = 15,
    battle_format: str = "gen9randomdoublesbattle",
    output_dir: str = "battle_data/doubles",
):
    os.makedirs(output_dir, exist_ok=True)
    rand_id = str(random.randint(1000, 9999))

    p1_name = f"HeurCollector_{rand_id}"
    p2_name = f"RandOpponent_{rand_id}"

    player1 = CollectingHeuristicsDoublesPlayer(
        battle_format=battle_format,
        account_configuration=AccountConfiguration(p1_name, ""),
        max_concurrent_battles=1,
    )
    player2 = RandomDoublesPlayer(
        battle_format=battle_format,
        account_configuration=AccountConfiguration(p2_name, ""),
        max_concurrent_battles=1,
    )

    sft_records = []
    dpo_records = []

    print(f"Generating data across {n_battles} battles in format '{battle_format}'...")
    pbar = tqdm(total=n_battles, desc="Generating Battles")

    for i in range(n_battles):
        if i % 2 == 0:
            await player1.battle_against(player2, n_battles=1)
        else:
            await player2.battle_against(player1, n_battles=1)
        pbar.update(1)

    pbar.close()

    # Process battle results
    total_turns_collected = 0
    for battle_tag, turns in player1.history.items():
        battle = player1.battles.get(battle_tag)
        if battle is None:
            continue

        won = bool(battle.won)
        for turn, prompt, json_out in turns:
            sft_records.append({
                "battle_tag": battle_tag,
                "turn": turn,
                "prompt": prompt,
                "output": json_out,
                "won": won,
            })
            total_turns_collected += 1

            if won:
                # Add to DPO as a positive example
                dpo_records.append({
                    "prompt": prompt,
                    "chosen": json_out,
                    "rejected": '{"thought": "pass", "slot_1": {"action": "move", "name": "splash"}, "slot_2": {"action": "move", "name": "splash"}}',
                })

    sft_file = os.path.join(output_dir, "doubles_sft.jsonl")
    with open(sft_file, "w") as f:
        for r in sft_records:
            f.write(json.dumps(r) + "\n")

    dpo_file = os.path.join(output_dir, "doubles_dpo.jsonl")
    with open(dpo_file, "w") as f:
        for r in dpo_records:
            f.write(json.dumps(r) + "\n")

    win_count = sum(1 for b in player1.battles.values() if b.won)
    print(f"Data generation complete!")
    print(f"Player 1 Win Rate: {win_count}/{len(player1.battles)} ({win_count / max(1, len(player1.battles)) * 100:.1f}%)")
    print(f"Total turns collected: {total_turns_collected}")
    print(f"Saved SFT dataset to: {sft_file}")
    print(f"Saved DPO dataset to: {dpo_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate VGC / Doubles training data from self-play.")
    parser.add_argument("--n_battles", type=int, default=15, help="Number of games to simulate")
    parser.add_argument("--format", type=str, default="gen9randomdoublesbattle", help="Pokemon Showdown format")
    parser.add_argument("--output_dir", type=str, default="battle_data/doubles", help="Output directory")
    args = parser.parse_args()

    asyncio.run(
        run_data_generation(
            n_battles=args.n_battles,
            battle_format=args.format,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
