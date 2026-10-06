"""
Online Reinforcement Learning (Policy Gradient / REINFORCE) training script
for PokéLLMon VGC Doubles bot against Heuristic Stall & Aggro opponents.
"""

import os
import time
import asyncio
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.rl.doubles_rl_env import DoublesRLEnv
from src.player.doubles_player import DoublesPlayer
from src.player.battle_order import BattleOrder, DoubleBattleOrder, DefaultBattleOrder
from src.environment.double_battle import DoubleBattle

# CPU execution is exceptionally fast for lightweight models (gpt2 forward pass ~10ms)
# and avoids the known MPS shared/private memory pool fragmentation leak across recurring turns.
device = torch.device("cpu")

def build_discrete_policy(model_name: str = "gpt2"):
    """
    Initializes a lightweight local model for action probability estimation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.train()
    return model, tokenizer

def select_action_with_policy(model, tokenizer, state_prompt: str, battle: DoubleBattle):
    """
    Evaluates candidate legal actions and samples with temperature using the policy LM.
    Returns (chosen_order, {'log_prob': log_prob, 'action_text': text})
    """
    # 0. Check forced switch
    if any(battle.force_switch):
        # We need to send switches only for slots that must switch
        must_switches = battle.force_switch
        if sum(must_switches) == 1:
            slot_idx = 0 if must_switches[0] else 1
            available = battle.available_switches[slot_idx]
            if available:
                candidate_orders = [BattleOrder(p) for p in available]
            else:
                candidate_orders = [DefaultBattleOrder()]
        else:
            # Both slots must switch
            switches_0 = battle.available_switches[0]
            switches_1 = battle.available_switches[1]
            pair_orders = []
            for s0 in switches_0:
                for s1 in switches_1:
                    if s0.species != s1.species:
                        pair_orders.append(DoubleBattleOrder(first_order=BattleOrder(s0), second_order=BattleOrder(s1)))
            candidate_orders = pair_orders if pair_orders else [DefaultBattleOrder()]
    else:
        # Standard double battle turn
        candidate_orders = []
        slot_orders = [[], []]
        for slot_idx in (0, 1):
            mon = battle.active_pokemon[slot_idx]
            if not mon or mon.fainted:
                slot_orders[slot_idx].append(BattleOrder(None))
                continue
                
            moves = battle.available_moves[slot_idx]
            switches = battle.available_switches[slot_idx]
            for m in moves:
                targets = battle.get_possible_showdown_targets(m, mon)
                if not targets:
                    slot_orders[slot_idx].append(BattleOrder(m))
                else:
                    for t in targets:
                        slot_orders[slot_idx].append(BattleOrder(m, move_target=t))
            for s in switches:
                slot_orders[slot_idx].append(BattleOrder(s))

        # Pair candidates into DoubleBattleOrder
        if slot_orders[0] and slot_orders[1]:
            for o1 in slot_orders[0][:4]:
                for o2 in slot_orders[1][:4]:
                    # Avoid duplicate switch
                    if o1.order and o2.order and hasattr(o1.order, "species") and hasattr(o2.order, "species"):
                        if o1.order.species == o2.order.species:
                            continue
                    candidate_orders.append(DoubleBattleOrder(first_order=o1, second_order=o2))
        elif slot_orders[0]:
            candidate_orders = [DoubleBattleOrder(first_order=o) for o in slot_orders[0][:6]]
        elif slot_orders[1]:
            candidate_orders = [DoubleBattleOrder(second_order=o) for o in slot_orders[1][:6]]

        if not candidate_orders:
            candidate_orders = [DoubleBattleOrder()]

    # Truncate prompt context for fast forward pass
    compact_prompt = state_prompt[-400:] + "\nAction: "
    input_ids = tokenizer.encode(compact_prompt, return_tensors="pt").to(device)

    # Score each candidate action
    action_texts = [o.message for o in candidate_orders]
    with torch.set_grad_enabled(model.training):
        # Forward pass on prefix
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :] # [1, vocab_size]
        
        # Action token scoring
        candidate_scores = []
        for text in action_texts:
            act_ids = tokenizer.encode(text, add_special_tokens=False)
            if act_ids:
                first_tok = act_ids[0]
                candidate_scores.append(logits[0, first_tok])
            else:
                candidate_scores.append(torch.tensor(0.0, device=device))

        scores_tensor = torch.stack(candidate_scores)
        probs = torch.softmax(scores_tensor, dim=0)
        dist = torch.distributions.Categorical(probs)
        chosen_idx = dist.sample()
        log_prob = dist.log_prob(chosen_idx)

    chosen_order = candidate_orders[chosen_idx.item()]
    return chosen_order, {"log_prob": log_prob, "action_text": chosen_order.message}

async def train_rl(
    num_iterations: int = 10,
    episodes_per_iter: int = 4,
    gamma: float = 0.95,
    lr: float = 1e-4,
    opponent_type: str = "stall"
):
    print(f"=== Starting Online RL Training against {opponent_type.upper()} bot ===")
    print(f"Device: {device} | Iterations: {num_iterations} | Batch size: {episodes_per_iter}")
    
    model, tokenizer = build_discrete_policy("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = DoublesRLEnv(opponent_type=opponent_type)

    running_win_rate = 0.0

    for it in range(1, num_iterations + 1):
        t_iter_start = time.time()
        policy_fn = lambda s, b: select_action_with_policy(model, tokenizer, s, b)
        
        batch_log_probs = []
        batch_returns = []
        batch_wins = 0
        total_turns = 0

        # Collect rollouts
        for ep in range(episodes_per_iter):
            episode = await env.run_episode(policy_fn)
            if episode["won"]:
                batch_wins += 1
            total_turns += episode["turns"]

            # Compute discounted returns G_t
            transitions = episode["transitions"]
            g = 0.0
            returns = []
            for t in reversed(transitions):
                g = t["step_reward"] + gamma * g
                returns.insert(0, g)

            for t, ret in zip(transitions, returns):
                if t["log_prob"] is not None:
                    batch_log_probs.append(t["log_prob"])
                    batch_returns.append(ret)

        win_rate = (batch_wins / episodes_per_iter) * 100
        running_win_rate = 0.7 * running_win_rate + 0.3 * win_rate if it > 1 else win_rate

        # Compute Policy Gradient Loss: - E [ log_prob * (Return - baseline) ]
        if batch_log_probs:
            returns_tensor = torch.tensor(batch_returns, device=device, dtype=torch.float32)
            # Normalize returns baseline
            if len(returns_tensor) > 1 and returns_tensor.std() > 1e-6:
                advantages = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            else:
                advantages = returns_tensor

            log_probs_tensor = torch.stack(batch_log_probs)
            loss = -(log_probs_tensor * advantages).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_val = loss.item()
            
            # Explicitly clean up tensors and cache
            del log_probs_tensor, returns_tensor, advantages, loss
        else:
            loss_val = 0.0

        del batch_log_probs, batch_returns
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        dur = time.time() - t_iter_start
        print(
            f"Iter {it:02d}/{num_iterations:02d} | "
            f"WinRate: {win_rate:5.1f}% (Moving: {running_win_rate:5.1f}%) | "
            f"Loss: {loss_val:+.4f} | Turns: {total_turns} | Time: {dur:.2f}s"
        )

    # Save model
    save_dir = "saved_models/doubles_rl_bot"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nTraining completed! Model checkpoint saved to: {save_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5, help="Number of RL iterations")
    parser.add_argument("--batch_size", type=int, default=4, help="Episodes per iteration")
    parser.add_argument("--opponent", type=str, default="stall", choices=["stall", "heuristics", "random"])
    args = parser.parse_args()

    asyncio.run(train_rl(
        num_iterations=args.iterations,
        episodes_per_iter=args.batch_size,
        opponent_type=args.opponent
    ))
