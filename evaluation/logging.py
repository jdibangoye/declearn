# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import json
import csv
import os
from typing import Any, Dict
import hashlib
import subprocess

class ExperimentLogger:
    def __init__(self, out_dir: str, config: Dict[str, Any]):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.config = config
        self.config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.csv_path = os.path.join(out_dir, "results.csv")
        self.jsonl_path = os.path.join(out_dir, "results.jsonl")
        self.git_hash = self._get_git_hash()
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "mean", "ci", "config_hash", "git_hash"])
        self.step = 0

    def _get_git_hash(self) -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"])
            return out.decode().strip()
        except Exception:
            return "nogit"

    def log_eval(self, **kwargs):
        self.step += 1
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.step, kwargs.get("mean", ""), kwargs.get("ci", ""), self.config_hash, self.git_hash])
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({"step": self.step, **kwargs, "config_hash": self.config_hash,
                                "git_hash": self.git_hash}) + "\n")