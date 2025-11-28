import time
import yaml
import argparse
import os
from datetime import datetime
from utils import *
from simulator import MuJoCoSimulator
from controller import BaseMPCController
from IK import generate_reference_trajectory

def main(model_name, data_collection=False, output_dir=None, timestamp=None, data_config=None):
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[model_name]
    
    # Base output directory
    output_dir = output_dir or "data"
    os.makedirs(output_dir, exist_ok=True)

    # Replace simulation time with max steps and set full_traj to true
    if data_collection:
        config["mujoco"]["sim_duration"] = config["mpc"]["mpc_timestep"]*data_config["max_steps"]
        config["mpc"]["full_traj"] = True
        config["mpc"]["solve_ocp"] = False
        config["mujoco"]["render"] = False

    # Create a subfolder with date, time and model name
    if timestamp is None: #takes timestamp from data collector if given
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{model_name}"
    run_dir = os.path.join(output_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving current run data to: {run_dir}")

    try:
        # Prerequisities 
        config = load_x0(config=config)                                                                 # Load x0 (Starting position)
        yref = load_yref(model_name=config["model"]["name"])                                            # Load yref

        if config["mpc"]["IK_required"]:                                                                # If dealing with manipulators
            collision_config = load_collision_config(model_name=config["model"]["name"])                # Load obstacles
            config["collision_config"] = collision_config                                               # Add to config for summary saving purpose

            yref, config = generate_reference_trajectory(yref, collision_config["obstacles"], config)   # Run IK to generate trajectory
            config["yref_end"] = yref[-1]                                                               # Add to config for summary saving purpose
            np.save(os.path.join(run_dir, "yref.npy"), yref)                                            # Save yref for reference

        controller = BaseMPCController(config, yref, collision_config)                                  # Create MPCController 

        # Save run summary
        save_summary(config=config, output_dir=run_dir)                                                 # Save summary of all 
    
        # Record start time
        start_time = time.time()

        # Create simulator object
        simulator = MuJoCoSimulator(config, controller.yref, controller, collision_config)
            
        # Run simulation
        simulator.run()

        # Record end time and print elapsed time
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\nTotal execution time: {elapsed:.2f} seconds")
        
        if data_collection: #quit here if data collection
            return simulator.logs
        elif config["mpc"]["solve_ocp"]: #quit here if only solving for single OCP
             return ocp_plot(simulator, run_dir)

    except Exception as e:
        import traceback
        print(f"\nRun failed: {e}")

        # Save the error info in the run folder
        error_log_path = os.path.join(run_dir, "error.log")
        with open(error_log_path, "w") as f:
            f.write("=== Simulation Run Failed ===\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            f.write(f"Error: {e}\n\n")
            f.write("=== Traceback ===\n")
            traceback.print_exc(file=f)
        raise

    finally:
        if not data_collection:
            try:
                # Save video if frames were recorded
                if simulator.frames and not config["mpc"]["solve_ocp"]:
                    save_video(simulator.frames, output_dir=run_dir, fps=simulator.config["mujoco"]["sim_framerate"])

                # Attempt to plot logs if available
                if "logs" in simulator.__dict__ and "time" in simulator.logs and not config["mpc"]["solve_ocp"]:
                    plot_signals(
                        time=simulator.logs["time"],
                        logs=simulator.logs,
                        model=simulator.model,
                        plots_config=simulator.config["plots"],
                        yref=simulator.yref, #convert this to use the logs instead and add the case where if traj is used, then plot the traj.
                        output_dir=run_dir,
                    )
            except Exception as plot_err:
                print(f"Could not generate summary or plots: {plot_err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)