import datetime
import os
import yaml
import glob
from utils import save_yaml
from data_collection.data_collector import run_data_collector
from neural_network.scripts import train_model
import configparser

def main():
    def log_vi(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"

        print(line, end="")  # still print to terminal

        with open(vi_log_path, "a") as f:
            f.write(line)

    VI_config_path = "value_iteration/VI_config.yaml"

    with open(VI_config_path, "r") as f:
        VI_config = yaml.safe_load(f)

    model_name = VI_config["model_name"]
    data_config_path = VI_config["data_config_path"]
    train_config_path = VI_config["train_config_path"]
    value_iteration_loops = VI_config["VI_loops"]

    # Load train config
    train_config = configparser.ConfigParser()
    train_config.read(train_config_path)

    # Load model config
    with open(f"configs/{model_name}config.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    base_dir = "value_iteration/output"
    main_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Main output folder
    main_output_dir = os.path.join(base_dir, f"{main_timestamp}_{model_name}_VI")
    os.makedirs(main_output_dir, exist_ok=True)

    # Save the VI config copy in main folder
    yaml_save_path = os.path.join(main_output_dir, "VI_config.yaml")
    save_yaml(VI_config, yaml_save_path)

    # Create VI log file
    vi_log_path = os.path.join(main_output_dir, "VI.log")

    with open(vi_log_path, "w") as f:
        f.write("=== VALUE ITERATION LOG ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Started at: {main_timestamp}\n")
        f.write(f"VI loops: {value_iteration_loops}\n")
        f.write("=" * 40 + "\n\n")

    # -----------------------------
    # Value Iteration Loops
    # -----------------------------
    try:
        for loop in range(value_iteration_loops):
            log_vi(f"=== Starting Value Iteration Loop {loop+1}/{value_iteration_loops} ===")

            loop_dir = os.path.join(main_output_dir, f"loop_{loop+1}")
            os.makedirs(loop_dir, exist_ok=True)

            data_collection_dir = os.path.join(loop_dir, "data_collection")
            training_dir = os.path.join(loop_dir, "training")
            os.makedirs(data_collection_dir, exist_ok=True)
            os.makedirs(training_dir, exist_ok=True)

            if loop == 0:
                model_config["mpc"]["controller_name"] = VI_config["controller_name"]
                model_config["mpc"]["terminal_cost"] = False
                train_config.set("MODEL", "load_checkpoint", "False")

            else:
                # Load from previous loop's best model
                prev_training_dir = os.path.join(main_output_dir, f"loop_{loop}", "training")
                pt_files = glob.glob(os.path.join(prev_training_dir, "*.pt"))

                if len(pt_files) != 1:
                    raise RuntimeError(
                        f"Expected exactly one .pt file in {prev_training_dir}, "
                        f"found {len(pt_files)}"
                    )

                best_model_path = pt_files[0]

                train_config.set("MODEL", "load_checkpoint", "True")
                train_config.set("MODEL", "checkpoint_path", best_model_path)
                model_config["mpc"]["controller_name"] = VI_config["NN_controller_name"]
                model_config["mpc"]["terminal_cost"] = True
                model_config["NN"]["checkpoint_path"] = best_model_path

            # -----------------
            # Data Collection
            # -----------------
            log_vi("Starting data collection")
            run_data_collector(
                model_name,
                data_config_path=data_config_path,
                run_dir=data_collection_dir,
                config=model_config,
            )
            log_vi(f"Data collection completed (saved in {data_collection_dir})")

            # Find the npz file produced by data collection
            npz_files = glob.glob(os.path.join(data_collection_dir, "*.npz"))

            if len(npz_files) != 1:
                raise RuntimeError(
                    f"Expected exactly one .npz file in {data_collection_dir}, "
                    f"found {len(npz_files)}"
                )

            data_npz_path = npz_files[0]
            # -----------------
            # Training
            # -----------------
            log_vi("Starting model training")

            # Get last checkpoint info
            if loop == 0:
                # First loop, no checkpoint
                train_config.set("MODEL", "load_checkpoint", "False")

            train_model(
                train_config,
                run_dir=training_dir,
                data_path=data_npz_path,
            )
            log_vi(f"Model training completed (saved in {training_dir})")
            prev_training_dir = training_dir
            log_vi(f"=== Finished Value Iteration Loop {loop+1} ===\n")

    except Exception as e:
        import traceback
        traceback.print_exc()
        log_vi(f"ERROR: {e}")

if __name__ == "__main__":
    main()