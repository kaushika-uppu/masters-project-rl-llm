import yaml
import argparse

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    validate_config(config)
    return config

def validate_config(config: dict) -> None:
    training_modes = ["sft", "rl"]
    active_modes = [mode for mode in training_modes if mode in config]

    if len(active_modes) == 0:
        raise ValueError("No training mode specified. Available: sft, rl")
    

def main():
    parser = argparse.ArgumentParser(description="Run project training pipeline from config.")
    parser.add_argument("config_path", type=str, help="Path to the config YAML file.")
    args = parser.parse_args()

    config = load_config(args.config_path)

    # 1. SFT
    # 2. RL
    # 3. Evaluation

if __name__ == "__main__":
    main()