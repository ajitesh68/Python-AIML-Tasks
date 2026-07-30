import json
import logging 
import os
import datetime 
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    def __init__(self, log_path: str="experiments/run_logs.json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_path = os.path.join(base_dir, log_path)
        folder = os.path.dirname(self.log_path)
        if folder:
            os.makedirs(folder,exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start_run(self, experiment_name: str) ->None:
        self.current_run = {
            "experiment_name": experiment_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "params": {},
            "metrics": {}
        }
        self.logger.info(f"Run started: {experiment_name}")
        

    def log_params(self, params: Dict[str,Any]) ->None:
        self.current_run["params"].update(params)
        self.logger.info(f"Params logged:{params}")

    def log_metrics(self, metrics: Dict[str,Any]) ->None:
        self.current_run["metrics"].update(metrics)
        self.logger.info(f"Metrics logged:{metrics}")

    def end_run(self, save: bool=True) ->None:
        if save:
            runs = []
            if os.path.exists(self.log_path):
                with open(self.log_path,'r') as f:
                    runs = json.load(f)
            
            runs.append(self.current_run)

            with open(self.log_path,'w') as f:
                json.dump(runs,f,indent=2)
            
            self.logger.info(f"Run saved:{self.current_run['experiment_name']}")
        else:
            self.logger.info("Run discarded(save = False)")
        


        