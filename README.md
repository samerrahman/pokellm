# PokéLLM: Competitive Pokémon VGC Reinforcement Learning Bot

An end-to-end autonomous reinforcement learning and LLM agent architecture designed for official competitive **Pokémon VGC (Video Game Championships, Regulation C)** on Pokémon Showdown.

Unlike standard single-battle bots (Gen 8 Random Singles), PokéLLM addresses the combinatorial complexity of **VGC Doubles**:
- **Simultaneous Dual-Slot Action Space:** Managing 2 active Pokémon on the field concurrently ($4 \text{ moves} \times \text{targets} + \text{switches}$ per slot, resulting in dozens of joint action permutations per turn).
- **Bring 6, Pick 4 Team Preview:** Strategic selection and lead ordering under Open Team Sheet (OTS) conditions.
- **VGC Tactical Mechanics:** Speed control (Tailwind, Trick Room), Protect cycling, spread move damage decay, redirection (Rage Powder, Follow Me), and Terastallization timing.
- **Bot Persona:** Dedicated competitive VGC bot moniker **`aquaspaghetti`** running standard high-tier tournament rosters.

---

## Online Reinforcement Learning (RL) Framework

Because offline Gen 9 VGC doubles datasets are scarce, PokéLLM trains directly in an online simulation loop against diverse heuristic baselines using **Policy Gradient (REINFORCE) with Advantage Baselines**.

### Reward Shaping Architecture
The dense reward signal $R_t$ incentivizes sound tactical doubles play without overfitting:

1. **Knockout Deltas:**
   - $+0.2$ per opponent fainted Pokémon.
   - $-0.2$ per allied fainted Pokémon.

2. **Super-Effective Damage Multiplier:**
   - Detects move type against target defensive typings in real time.
   - Awards $+0.05 \times \text{Multiplier}$ for advantageous targeting ($+0.10$ for $2\times$, $+0.20$ for $4\times$ double super-effective hits).

3. **High-Value Stat Buffs (Post-EV Thresholding):**
   - Evaluates active allied stat boosts (`atk`, `spa`, `spe`, `def`, `spd`).
   - Rather than relying on static uninvested base stats, checks the actual **after-EVs stat value** calculated from the Showdown engine at Level 50.
   - Boosts on stats exceeding the competitive threshold ($\ge 130$ post-EV) earn $+0.05$ per boost stage (e.g. Flutter Mane Speed/SpA, Chi-Yu SpA, Iron Hands Attack).

4. **Terminal Outcome:**
   - $+1.0$ for winning the match, $-1.0$ for losing.

---

## Opponents & Baselines

To ensure robust policy generalization, the training environment pits `aquaspaghetti` against varied tactical opponents:
- **`StallDoublesPlayer`:** Bulky attrition bot executing Protect prediction cycles, Amoonguss Spore/Rage Powder disruption, ally Pollen Puff healing, and defensive pivot switching into Assault Vest Iron Hands and Ting-Lu.
- **`SimpleHeuristicsDoublesPlayer`:** Aggressive offensive baseline prioritizing STAB focus-fire, type coverage, and target sniping.
- **`RandomDoublesPlayer`:** High-entropy stochastic baseline for exploratory coverage.

---

## Getting Started

### 1. Requirements
- Python >= 3.10
- PyTorch >= 2.0
- Transformers, Peft, orjson
- Node.js >= 16 (for local Pokémon Showdown engine)

### 2. Setting Up the Local Pokémon Showdown Server
```sh
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```
The server will listen at `localhost:8000`.

### 3. Training the Bot
Run the online policy gradient trainer:
```sh
python train_doubles_rl.py --iterations 20 --batch_size 4 --opponent stall
```

To run offline preference/SFT training:
```sh
python train_doubles.py --model_name qwen_0_5b --stage sft
```
