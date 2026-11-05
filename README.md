# PokéLLM

A bot that plays competitive Pokémon VGC (Regulation C doubles) on Pokémon Showdown. I'm using it to mess around with reinforcement learning and language models as battle policies.

Most Pokémon AI projects do singles. I wanted to try doubles because it's harder:
- you pick moves for two Pokémon at once, so there are a lot more options each turn
- team preview, where you bring 6 and pick 4
- doubles stuff like Protect, spread moves, redirection, and speed control matters a lot more

The bot plays as `aquaspaghetti` with a normal Reg C team.

## How it learns

There aren't many good doubles replays to learn from, so right now it trains online with plain REINFORCE (policy gradient) against some scripted bots.

The reward is a mix of a few things:
- a small bonus when an opponent faints, a small penalty when one of mine does
- a bonus when a boost pushes an opponent into fewer hits to KO (like 2HKO to OHKO), using a Gen 9 damage calc
- a bit of reward for stacking stat boosts
- how much the board position improved (speed, field conditions, HP, threats)
- win or loss at the end

## Opponents

- `NashMatrixDoublesPlayer` builds a payoff matrix of both players' moves each turn and plays a mixed Nash equilibrium
- `StallDoublesPlayer` plays defensively: Protect, Spore, bulky pivots
- `SimpleHeuristicsDoublesPlayer` just goes for STAB and big damage
- `RandomDoublesPlayer` picks random legal moves, mostly for testing

## Running it

You need Python 3.10+, PyTorch 2+, transformers, peft, orjson, and Node.js 16+ for Showdown.

Start a local Showdown server (it runs on `localhost:8000`):
```sh
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

Then train with RL (`--opponent` can be `stall`, `heuristics`, or `random`):
```sh
python train_doubles_rl.py --iterations 10 --batch_size 2 --opponent stall
```

Or do supervised fine-tuning on collected battle data:
```sh
python train_doubles.py --model_name gpt2 --data_path battle_data/doubles/doubles_sft.jsonl
```
