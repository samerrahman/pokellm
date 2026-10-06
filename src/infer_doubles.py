import argparse
import asyncio
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.client.account_configuration import AccountConfiguration
from src.player.doubles_baselines import RandomDoublesPlayer, SimpleHeuristicsDoublesPlayer
from src.player.doubles_llm_player import DoublesLLMPlayer


async def evaluate_doubles_bot(
    model_dir: str,
    opponent_type: str = "random",
    n_battles: int = 5,
    battle_format: str = "gen9randomdoublesbattle",
    temperature: float = 0.5,
):
    print(f"Loading trained doubles model from: {model_dir}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    rand_suffix = str(random.randint(1000, 9999))
    bot_name = f"PokeLLM_Doubles_{rand_suffix}"
    opp_name = f"Opponent_{rand_suffix}"

    llm_player = DoublesLLMPlayer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        battle_format=battle_format,
        temperature=temperature,
        account_configuration=AccountConfiguration(bot_name, ""),
        save_trajectory_path="battle_log/eval_doubles_bot.jsonl",
        max_concurrent_battles=1,
    )

    if opponent_type == "random":
        opponent = RandomDoublesPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(opp_name, ""),
            max_concurrent_battles=1,
        )
    else:
        opponent = SimpleHeuristicsDoublesPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(opp_name, ""),
            max_concurrent_battles=1,
        )

    print(f"\n--- Starting Evaluation: {bot_name} vs {opponent_type} opponent ({n_battles} games) ---")
    pbar = tqdm(total=n_battles, desc="Evaluating Battles")

    for i in range(n_battles):
        if i % 2 == 0:
            await llm_player.battle_against(opponent, n_battles=1)
        else:
            await opponent.battle_against(llm_player, n_battles=1)
        pbar.update(1)

    pbar.close()

    wins = sum(1 for b in llm_player.battles.values() if b.won)
    total = len(llm_player.battles)
    win_rate = (wins / max(1, total)) * 100

    print("\n" + "=" * 50)
    print(f"Evaluation Results for {bot_name}:")
    print(f"Total Battles: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {total - wins}")
    print(f"Win Rate: {win_rate:.1f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Doubles LLM bot on Showdown.")
    parser.add_argument("--model_dir", type=str, default="saved_models/doubles_bot", help="Path to saved model")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "heuristic"], help="Opponent type")
    parser.add_argument("--n_battles", type=int, default=5, help="Number of evaluation games")
    parser.add_argument("--format", type=str, default="gen9randomdoublesbattle", help="Battle format")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    args = parser.parse_args()

    asyncio.run(
        evaluate_doubles_bot(
            model_dir=args.model_dir,
            opponent_type=args.opponent,
            n_battles=args.n_battles,
            battle_format=args.format,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
