# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import argparse
import subprocess
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()

    for c in args.configs:
        subprocess.run(["python", "-m", "declearn.cli.train", "--config", c], check=True)

if __name__ == "__main__":
    main()