import sys
from datetime import datetime
import pandas as pd
import os


class TrainLogger(object):
    def __init__(self, output):
        self.terminal = sys.stdout

        os.makedirs(f"logs/{output}", exist_ok=True)
        self.log_config_file = open(f"logs/{output}/config.txt", 'w+')
        self.log_results_csv = open(f"logs/{output}/results.csv", 'w+')
        self.log = open(f"logs/{output}/log.txt", 'w+')

        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def log_to_results(self, message):
        self.log_results_csv.write(message)

    def log_config(self, agent, config):
        data = [[k, v] for k, v in config.items()]
        self.log_config_file.write(f"######## {type(agent).__name__} Config #########")
        self.log_config_file.write(pd.DataFrame(data).to_string() + "\n")
        self.log_config_file.write("#############################################\n\n\n")
        self.log_to_results("reward,loss,epsilon\n")

    def get_results_df(self) -> pd.DataFrame:
        self.log_results_csv.close()
        return pd.read_csv(self.log_results_csv.name)
