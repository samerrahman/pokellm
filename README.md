# PokéLLM

An experimental project exploring reinforcement learning and language model policies for competitive **Pokémon VGC (Regulation C Doubles)** on Pokémon Showdown.

### Background

Most existing work in Pokémon battle AI focuses on singles formats (such as Gen 8 Random Singles). This project experiments with the doubles format, which presents a few different dynamics:
- **Two Active Pokémon:** Each turn requires choosing actions for both slots simultaneously, expanding the branching factor of candidate moves and switches.
- **Team Preview:** Players bring 6 Pokémon and select 4, factoring in team synergy and lead positioning.
- **Doubles Mechanics:** Moves like Protect, spread attacks, redirection, and speed control play a larger role in game flow.
- **Test Bot:** The agent runs under the username **`aquaspaghetti`** using a standard Regulation C team.

---

### Reinforcement Learning Setup

Given the limited availability of high-quality doubles replays, the current approach explores online training against scripted baselines using a basic **Policy Gradient (REINFORCE)** loop.

#### Reward Signals
The step reward incorporates a few heuristic signals alongside the final match outcome:

1. **Faints:**
   - Small positive reward when an opponent Pokémon faints.
   - Small negative penalty when an allied Pokémon faints.

2. **Type Effectiveness:**
   - Small bonus when selecting damaging moves that hit for super-effective damage based on current board state.

3. **Stat Stage Boosts:**
   - Encourages boosting stats that have significant EV investment (checking post-EV Level 50 stats $\ge 130$, like Speed or offensive stats on dedicated attackers).

4. **Match Outcome:**
   - Win / loss terminal reward at the end of the battle.

---

### Opponents & Baselines

To give the policy something consistent to practice against, the environment includes a few baseline bots:
- **`StallDoublesPlayer`:** A defensive bot using Protect cycles, sleep disruption (Spore), and bulky pivots.
- **`SimpleHeuristicsDoublesPlayer`:** A straightforward offensive bot that prioritizes STAB and high-damage coverage.
- **`RandomDoublesPlayer`:** A random legal move baseline used for testing and sanity checks.

---

### Getting Started

#### 1. Requirements
- Python >= 3.10
- PyTorch >= 2.0
- Transformers, Peft, orjson
- Node.js >= 16 (for Pokémon Showdown)

#### 2. Setting Up the Local Battle Engine
```sh
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```
Showdown will run locally on `localhost:8000`.

#### 3. Running Training
To run the online RL loop:
```sh
python train_doubles_rl.py --iterations 10 --batch_size 2 --opponent stall
```

To run supervised / preference experiments:
```sh
python train_doubles.py --model_name qwen_0_5b --stage sft
```
