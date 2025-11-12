# main.py - Entry point script to run the component via subprocess

import subprocess

if __name__ == "__main__":
    # Call the my_component.py script using a subprocess
    subprocess.run(["mlflow", "run", "data_preparation"])

    subprocess.run(["mlflow", "run", "model_training"])

    subprocess.run(["mlflow", "run", "model_evaluation"])
