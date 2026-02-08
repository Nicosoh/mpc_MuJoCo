import os
import sys
import yaml
import json
import shutil
import datetime
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from utils import save_yaml

def main():
    # Helper function to append to log file with timestamp
    def log_vi(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        print(line, end="")
        with open(vi_log_path, "a") as f:
            f.write(line)
    
    def update_plot(x, ground_truth, controller, MSE):
        """Update and save plot with current metrics."""
        ax1.clear()
        ax2.clear()
        
        # Top subplot: ground truth vs controller
        ax1.plot(x, ground_truth, label="Ground Truth", linewidth = 0.8)
        ax1.plot(x, controller, label="Controller", linewidth = 0.8)
        ax1.set_ylabel("Mean Cost")
        ax1.grid(True, which="both", linestyle=":", alpha=0.4)
        ax1.legend()
        
        # Bottom subplot: MSE
        ax2.plot(x, MSE, label="MSE", color='r', linewidth = 0.8)
        ax2.set_xlabel("Value Iteration Loop")
        ax2.set_ylabel("Mean Squared Error")
        ax2.grid(True, which="both", linestyle=":", alpha=0.4)
        ax2.legend()
        ax2.set_yscale('log')
        
        fig.tight_layout()
        plt.savefig(plt_save_path, dpi=400, bbox_inches='tight')
    
    # Load VI config
    VI_config_path = "value_iteration/VI_config.yaml"
    with open(VI_config_path, "r") as f:
        VI_config = yaml.safe_load(f)
    
    # Extract config from VI
    model_name = VI_config["model_name"]
    data_config_path = VI_config["data_config_path"]
    train_config_path = VI_config["train_config_path"]
    value_iteration_loops = VI_config["VI_loops"]
    resume_training = VI_config["resume_training"]
    
    # Check if resume training
    start_loop = 0
    main_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if resume_training:
        start_loop = VI_config["loop_to_resume_from"]
        main_output_dir = VI_config["VI_dir"]
        suffix = f"_resume_from_{start_loop}"
        
        # Delete the directory for the loop we are resuming from
        loop_dir_to_delete = os.path.join(main_output_dir, f"loop_{start_loop}")

        if os.path.exists(loop_dir_to_delete):
            print(f"Deleting existing directory: {loop_dir_to_delete}")
            shutil.rmtree(loop_dir_to_delete)
    
    else:
        base_dir = "value_iteration/output"
        main_output_dir = os.path.join(base_dir, f"{main_timestamp}_{model_name}_VI")
        os.makedirs(main_output_dir, exist_ok=True)
        suffix = ""

    # Paths to save files to
    plt_save_path = os.path.join(main_output_dir, f"VI_plot{suffix}.png")
    vi_log_path   = os.path.join(main_output_dir, f"VI{suffix}.log")
    yaml_save_path = os.path.join(main_output_dir, f"VI_config{suffix}.yaml")

    save_yaml(VI_config, yaml_save_path)
    
    with open(vi_log_path, "w") as f:
        f.write("=== VALUE ITERATION LOG (SUBPROCESS MODE) ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Started at: {main_timestamp}\n")
        f.write(f"VI loops: {value_iteration_loops}\n")
        f.write("=" * 50 + "\n\n")
    
    # Setup plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    x = []
    ground_truth = []
    controller = []
    MSE = []
    
    # Absolute path to the worker script
    vi_loop_worker_path = os.path.join(os.path.dirname(__file__), "vi_loop_worker.py")
        
    # VI Loop, Zero indexed, so minus one from the folder number
    for loop in range(start_loop - 1, value_iteration_loops):
        log_vi(f"=== Starting Value Iteration Loop {loop+1}/{value_iteration_loops} ===")
        
        # Spawn subprocess for this loop
        cmd = [
            sys.executable,
            vi_loop_worker_path,
            str(loop),
            main_output_dir,
            model_name,
            data_config_path,
            train_config_path,
            VI_config_path,
        ]
        
        try: # Try to run the subprocess
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            
            # If result is not successful
            if result.returncode != 0:
                log_vi(f"Loop worker failed with return code {result.returncode}")
                if result.stdout:
                    log_vi(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    log_vi(f"STDERR:\n{result.stderr}")
                log_vi(f"ERROR: Loop {loop+1} failed")
                continue
            
            # Path to metrics
            loop_dir = os.path.join(main_output_dir, f"loop_{loop+1}")
            metrics_path = os.path.join(loop_dir, "metrics.json")
            
            # Read metrics
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                
                if metrics.get("success"):
                    x.append(metrics["loop"])
                    ground_truth.append(metrics["gt_cost"])
                    controller.append(metrics["ctrl_cost"])
                    MSE.append(metrics["mse"])
                    
                    log_vi(f"Metrics: GT={metrics['gt_cost']:.4f}, CTRL={metrics['ctrl_cost']:.4f}, MSE={metrics['mse']:.4e}")
                else:
                    log_vi(f"ERROR: {metrics.get('error', 'Unknown error')}")
            else:
                log_vi(f"ERROR: No metrics file found at {metrics_path}")
            
            # Update plot after each loop
            update_plot(x, ground_truth, controller, MSE)
            
        except Exception as e:
            log_vi(f"ERROR spawning/running loop worker: {e}")
            import traceback
            log_vi(traceback.format_exc())
        
        log_vi(f"=== Finished Value Iteration Loop {loop+1} ===\n")
    
    log_vi(f"\n=== VI COMPLETE ===")
    log_vi(f"Total loops completed: {len(x)}/{value_iteration_loops}")
    plt.savefig(plt_save_path, dpi=400, bbox_inches='tight')

if __name__ == "__main__":
    main()