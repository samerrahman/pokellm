import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request


def is_port_in_use(port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_server(port: int = 8000, timeout: int = 15) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def start_showdown_server():
    if is_port_in_use(8000):
        print("Pokemon Showdown server is already running on port 8000.")
        return None

    showdown_dir = os.path.join(os.getcwd(), "pokemon-showdown")
    if not os.path.exists(showdown_dir):
        # Check parent or other known location
        alt = "/Users/Samer/Projects/Pokellm/pokemon-showdown"
        if os.path.exists(alt):
            showdown_dir = alt

    print(f"Starting local Pokemon Showdown server from {showdown_dir}...")
    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=showdown_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if wait_for_server(8000):
        print("Pokemon Showdown server started successfully on port 8000.")
        return proc
    else:
        print("Warning: Timed out waiting for Showdown server to open port 8000.")
        return proc


def run_command(cmd, desc):
    print(f"\n{'=' * 60}")
    print(f"STEP: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"Error: Step '{desc}' failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)


def main():
    parser = argparse.ArgumentParser(description="End-to-End VGC Doubles Training Pipeline for PokéLLMon")
    parser.add_argument("--battles", type=int, default=10, help="Number of self-play battles to collect")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model for fine-tuning")
    parser.add_argument("--eval_battles", type=int, default=5, help="Number of evaluation battles")
    parser.add_argument("--skip_data", action="store_true", help="Skip data collection if dataset exists")
    parser.add_argument("--skip_train", action="store_true", help="Skip training if model checkpoint exists")
    args = parser.parse_args()

    python_bin = "/Users/Samer/Projects/Pokellm/venv/bin/python"
    if not os.path.exists(python_bin):
        python_bin = sys.executable

    # 1. Start Showdown Server
    server_proc = start_showdown_server()

    try:
        # 2. Generate Doubles Data
        dataset_path = "battle_data/doubles/doubles_sft.jsonl"
        if not args.skip_data or not os.path.exists(dataset_path):
            run_command(
                [python_bin, "-m", "src.data.gen_doubles_data", "--n_battles", str(args.battles)],
                f"Generating {args.battles} Doubles Battles for Training",
            )
        else:
            print(f"Using existing data at: {dataset_path}")

        # 3. Train Model
        output_model_dir = "saved_models/doubles_bot"
        if not args.skip_train or not os.path.exists(output_model_dir):
            run_command(
                [
                    python_bin,
                    "train_doubles.py",
                    "--model_name", args.model,
                    "--data_path", dataset_path,
                    "--output_dir", output_model_dir,
                    "--epochs", str(args.epochs),
                    "--batch_size", "2",
                ],
                f"Fine-Tuning {args.model} for VGC Doubles",
            )
        else:
            print(f"Using existing model at: {output_model_dir}")

        # 4. Evaluate Model on Live Doubles Games
        run_command(
            [
                python_bin,
                "-m", "src.infer_doubles",
                "--model_dir", output_model_dir,
                "--n_battles", str(args.eval_battles),
                "--opponent", "random",
            ],
            f"Evaluating Trained Bot across {args.eval_battles} Live Games",
        )

        print("\n" + "=" * 60)
        print("VGC DOUBLES PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        if server_proc:
            print("Stopping local Showdown server...")
            server_proc.terminate()


if __name__ == "__main__":
    main()
