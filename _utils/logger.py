import sys
import pandas as pd
import os
from copy import copy
from _utils.utils import Bcolors



def create_logger(algo, agent, config, experiment_number, trial_number=None, is_trial=False):
    # Create Train Logger
    prob = config['problem']
    output = f"{algo}/logs/{prob}"
    os.makedirs(output, exist_ok=True)
    output = f"{output}/experiment_{experiment_number}"
    terminal_log_output=None

    if is_trial:
        terminal_log_output = copy(output)
        output = output + f"/trial_{trial_number}"

    logger = TrainLogger(output=output, terminal_log_output=terminal_log_output)
    logger.log_config(agent, dict(config, **agent.config))

    return logger


class TrainLogger(object):
    def __init__(self, output, terminal_log_output):
        self.terminal = sys.stdout

        os.makedirs(f"{output}", exist_ok=True)
        self.log_config_file = open(f"{output}/config.txt", 'w+')
        self.log_results_csv = open(f"{output}/results.csv", 'w+')
        if terminal_log_output is None:
            self.log = open(f"{output}/log.txt", 'w+')
        else:
            self.log = open(f"{terminal_log_output}/log.txt", "w+")

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

    def get_results_df(self) -> pd.DataFrame:
        print(f'{Bcolors.OKGREEN}File {self.log_results_csv.name} closed{Bcolors.ENDC}')
        self.log_results_csv.close()
        return pd.read_csv(self.log_results_csv.name)
