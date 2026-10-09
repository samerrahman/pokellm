"""
Online Reinforcement Learning (Policy Gradient / REINFORCE) training script
for PokéLLM VGC Doubles bot against Heuristic Stall & Aggro opponents.
Integrates Weights & Biases (wandb) for real-time loss, win rate, and reward tracking.
"""

import os
import gc
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
from src.rl.vgc_matrix_game import VGCMatrixGameSolver

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

device = torch.device("cpu")

def build_discrete_policy(model_name: str = "gpt2"):
    """
    Initializes a lightweight local model for action probability estimation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer

def select_action_with_policy(model, tokenizer, state_prompt: str, battle: DoubleBattle):
    """
    Evaluates candidate legal actions under no_grad during rollouts to prevent graph memory accumulation.
    Returns (chosen_order, {'action_text': text, 'chosen_action_id': int, 'candidate_action_ids': list, 'compact_prompt': str})
    """
    # 0. Check forced switch
    if any(battle.force_switch):
        must_switches = battle.force_switch
        if sum(must_switches) == 1:
            slot_idx = 0 if must_switches[0] else 1
            available = battle.available_switches[slot_idx]
            if available:
                candidate_orders = [BattleOrder(p) for p in available]
            else:
                candidate_orders = [DefaultBattleOrder()]
        else:
            switches_0 = battle.available_switches[0]
            switches_1 = battle.available_switches[1]
            pair_orders = []
            for s0 in switches_0:
                for s1 in switches_1:
                    if s0.species != s1.species:
                        pair_orders.append(DoubleBattleOrder(first_order=BattleOrder(s0), second_order=BattleOrder(s1)))
            candidate_orders = pair_orders if pair_orders else [DefaultBattleOrder()]
    else:
        # Standard double battle turn: use game-theoretic tactical candidate generation
        # prioritizes Protect, high-damage moves against active foes, and pivot switches
        candidate_orders = VGCMatrixGameSolver.get_joint_candidates(battle, is_ally=True)
        if not candidate_orders:
            candidate_orders = [DoubleBattleOrder()]

    compact_prompt = state_prompt[-400:] + "\nAction: "
    input_ids = tokenizer.encode(compact_prompt, return_tensors="pt").to(device)

    action_texts = [o.message for o in candidate_orders]
    
    # Fast evaluation without retaining backward graphs during rollout
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :] # [1, vocab_size]
        
        candidate_token_ids = []
        candidate_scores = []
        for text in action_texts:
            act_ids = tokenizer.encode(text, add_special_tokens=False)
            tok = act_ids[0] if act_ids else tokenizer.eos_token_id
            candidate_token_ids.append(tok)
            candidate_scores.append(logits[0, tok].item())

        scores_t = torch.tensor(candidate_scores, dtype=torch.float32)
        probs = torch.softmax(scores_t, dim=0)
        chosen_idx = torch.distributions.Categorical(probs).sample().item()

    chosen_order = candidate_orders[chosen_idx]
    
    return chosen_order, {
        "action_text": chosen_order.message,
        "chosen_action_idx": chosen_idx,
        "candidate_token_ids": candidate_token_ids,
        "compact_prompt": compact_prompt
    }

async def train_rl(
    num_iterations: int = 20,
    episodes_per_iter: int = 4,
    gamma: float = 0.95,
    lr: float = 1e-4,
    opponent_type: str = "stall",
    use_wandb: bool = True,
    project_name: str = "pokellm-vgc"
):
    print(f"=== Starting Online RL Training against {opponent_type.upper()} bot ===")
    print(f"Device: {device} | Iterations: {num_iterations} | Batch size: {episodes_per_iter}")

    if use_wandb and HAS_WANDB:
        wandb.init(
            project=project_name,
            config={
                "model": "gpt2",
                "format": "gen9vgc2023regc",
                "bot_name": "aquaspaghetti",
                "opponent": opponent_type,
                "iterations": num_iterations,
                "batch_size": episodes_per_iter,
                "learning_rate": lr,
                "gamma": gamma
            }
        )
        print("WandB run initialized successfully.")
    
    model, tokenizer = build_discrete_policy("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = DoublesRLEnv(opponent_type=opponent_type)

    running_win_rate = 0.0

    for it in range(1, num_iterations + 1):
        t_iter_start = time.time()
        
        # Policy wrapper ensures no grad accumulation during matches
        policy_fn = lambda s, b: select_action_with_policy(model, tokenizer, s, b)
        
        batch_prompts = []
        batch_candidate_tokens = []
        batch_chosen_indices = []
        batch_returns = []
        batch_wins = 0
        total_turns = 0
        total_dense_rewards = 0.0

        # 1. Rollout collection
        model.eval()
        for ep in range(episodes_per_iter):
            episode = await env.run_episode(policy_fn)
            if episode["won"]:
                batch_wins += 1
            total_turns += episode["turns"]
            total_dense_rewards += episode.get("total_reward", 0.0)

            transitions = episode["transitions"]
            g = 0.0
            ep_returns = []
            for t in reversed(transitions):
                g = t["step_reward"] + gamma * g
                ep_returns.insert(0, g)

            for t, ret in zip(transitions, ep_returns):
                if "compact_prompt" in t and "candidate_token_ids" in t:
                    batch_prompts.append(t["compact_prompt"])
                    batch_candidate_tokens.append(t["candidate_token_ids"])
                    batch_chosen_indices.append(t["chosen_action_idx"])
                    batch_returns.append(ret)

            del episode
            gc.collect()

        win_rate = (batch_wins / episodes_per_iter) * 100
        running_win_rate = 0.7 * running_win_rate + 0.3 * win_rate if it > 1 else win_rate
        avg_turns = total_turns / episodes_per_iter
        avg_reward = total_dense_rewards / episodes_per_iter

        # 2. Optimization step: single forward pass with gradients on collected steps
        if batch_prompts:
            model.train()
            optimizer.zero_grad()
            
            # Normalize returns to get advantages
            returns_tensor = torch.tensor(batch_returns, device=device, dtype=torch.float32)
            if len(returns_tensor) > 1 and returns_tensor.std() > 1e-6:
                advantages = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            else:
                advantages = returns_tensor

            # Compute log probs and accumulate gradients per transition to keep memory strictly constant
            total_loss = 0.0
            n_steps = len(batch_prompts)
            for idx, (prompt, cand_tokens, chosen_idx) in enumerate(zip(batch_prompts, batch_candidate_tokens, batch_chosen_indices)):
                inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
                out = model(inp)
                last_logits = out.logits[0, -1, :] # [vocab_size]
                cand_logits = torch.stack([last_logits[tok] for tok in cand_tokens])
                probs = torch.softmax(cand_logits, dim=0)
                log_p = torch.log(probs[chosen_idx] + 1e-10)
                adv = advantages[idx]
                step_loss = -(log_p * adv) / n_steps
                step_loss.backward()
                total_loss += step_loss.item()
                del out, inp, last_logits, cand_logits, probs, log_p, step_loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_val = total_loss

            del returns_tensor, advantages
            gc.collect()
        else:
            loss_val = 0.0

        dur = time.time() - t_iter_start
        print(
            f"Iter {it:02d}/{num_iterations:02d} | "
            f"WinRate: {win_rate:5.1f}% (Moving: {running_win_rate:5.1f}%) | "
            f"Loss: {loss_val:+.4f} | AvgTurns: {avg_turns:.1f} | AvgReward: {avg_reward:+.2f} | Time: {dur:.2f}s"
        )

        if use_wandb and HAS_WANDB:
            wandb.log({
                "iteration": it,
                "win_rate": win_rate,
                "running_win_rate": running_win_rate,
                "policy_loss": loss_val,
                "avg_turns": avg_turns,
                "avg_reward": avg_reward,
                "iter_time_sec": dur
            })

    # Save trained checkpoint
    save_dir = "saved_models/doubles_rl_bot"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nTraining completed! Model checkpoint saved to: {save_dir}")

    if use_wandb and HAS_WANDB:
        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10, help="Number of RL iterations")
    parser.add_argument("--batch_size", type=int, default=2, help="Episodes per iteration")
    parser.add_argument("--opponent", type=str, default="stall", choices=["stall", "heuristics", "random"])
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    asyncio.run(train_rl(
        num_iterations=args.iterations,
        episodes_per_iter=args.batch_size,
        opponent_type=args.opponent,
        use_wandb=not args.no_wandb
    ))
