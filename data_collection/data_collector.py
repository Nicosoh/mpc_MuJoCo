import argparse
import sys
import os
from datetime import datetime
from main import main

def worker(worker_id, run_indices, model_name, output_dir, data_config, config):
    import os
    import sys
    import gc
    import traceback
    from datetime import datetime
    from main import main

    os.environ["OMP_NUM_THREADS"] = "1"

    worker_output_dir = os.path.join(output_dir, f"worker_{worker_id}")
    os.makedirs(worker_output_dir, exist_ok=True)

    log_file_path = os.path.join(worker_output_dir, f"worker_{worker_id}.log")
    error_file_path = os.path.join(worker_output_dir, f"worker_{worker_id}_error.log")

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    try:
        with open(log_file_path, "a") as log_file:
            sys.stdout = log_file
            sys.stderr = log_file

            print(f"[Worker {worker_id}] starting with {len(run_indices)} runs")
            print(f"[Worker {worker_id}] PID: {os.getpid()}")
            sys.stdout.flush()

            all_logs = {}
            for run_index in run_indices:
                success = False
                retry_count = 0
                while not success:  # keep retrying until it succeeds
                    run_timestamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}_run{run_index}"
                    try:
                        logs = main(
                            model_name,
                            data_collection=True,
                            output_dir=worker_output_dir,
                            timestamp=run_timestamp,
                            data_config=data_config,
                            config=config,
                            worker_id=worker_id,
                        )
                        all_logs[run_timestamp] = logs
                        success = True
                        print(f"[Worker {worker_id}] Finished run {run_index}")
                        sys.stdout.flush()
                    except Exception as e:
                        retry_count += 1
                        error_msg = f"[Worker {worker_id}] Run {run_index} attempt {retry_count} failed: {type(e).__name__}: {e}"
                        print(error_msg)
                        print(traceback.format_exc())
                        sys.stdout.flush()
                        
                        # Save error to error log for later inspection
                        with open(error_file_path, "a") as ef:
                            ef.write(f"\n=== Run {run_index} Attempt {retry_count} ===\n")
                            ef.write(f"Timestamp: {datetime.now()}\n")
                            ef.write(error_msg + "\n")
                            ef.write(traceback.format_exc())
                        
                        if retry_count >= 3:
                            raise RuntimeError(f"Run {run_index} failed after 3 attempts: {e}")
                
                # Force garbage collection after each run to prevent memory accumulation
                gc.collect()
                print(f"[Worker {worker_id}] Garbage collected after run {run_index}")
                sys.stdout.flush()

            print(f"[Worker {worker_id}] completed all assigned runs")
            sys.stdout.flush()
    except Exception as e:
        error_msg = f"[Worker {worker_id}] FATAL ERROR: {type(e).__name__}: {e}"
        print(error_msg)
        print(traceback.format_exc())
        sys.stdout.flush()
        
        # Save fatal error to error log
        with open(error_file_path, "a") as ef:
            ef.write(f"\n=== FATAL ERROR ===\n")
            ef.write(f"Timestamp: {datetime.now()}\n")
            ef.write(error_msg + "\n")
            ef.write(traceback.format_exc())
        
        raise  # Re-raise so ProcessPoolExecutor can see it

    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    return all_logs

def run_data_collector(model_name, data_config_path="data_collection/data_config.yaml", run_dir=None, config=None):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import yaml
    import time
    import os
    from utils import save_yaml
    from data_collection import save_npz

    # Load config
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)["data_collector"]

    n_runs = data_config["runs"]
    n_workers = min(mp.cpu_count(), data_config.get("workers", 8))  # number of workers
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Output dir
    if run_dir is None:
        output_dir = os.path.join("data", f"{timestamp}_{model_name}_data_collection")
    else:
        output_dir = run_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save config used
    save_yaml(data_config, os.path.join(output_dir, "data_config.yaml"))

    # Split n_runs roughly evenly across workers
    run_indices_per_worker = [[] for _ in range(n_workers)]
    for i in range(n_runs):
        run_indices_per_worker[i % n_workers].append(i)

    start_time = time.time()
    all_logs = {}

    print(f"Launching {n_workers} workers for {n_runs} runs")

    ### === PHASE 1: First iteration (1 job per worker) === ###
    print("\n--- Phase 1: Running first iteration for each worker sequentially ---")
    first_runs_per_worker = [runs[0] for runs in run_indices_per_worker if runs]  # take the first run
    for worker_id, run_index in enumerate(first_runs_per_worker):
        logs = worker(
            worker_id=worker_id,
            run_indices=[run_index],  # single run
            model_name=model_name,
            output_dir=output_dir,
            data_config=data_config,
            config=config,
        )
        all_logs.update(logs)

    ### === PHASE 2: Remaining runs asynchronously === ###
    print("\n--- Phase 2: Running remaining jobs asynchronously ---")
    remaining_indices_per_worker = [
        runs[1:] if len(runs) > 1 else [] for runs in run_indices_per_worker
    ]

    # Only launch workers that have remaining runs
    workers_with_remaining = [
        (w_id, runs) for w_id, runs in enumerate(remaining_indices_per_worker) if runs
    ]

    if workers_with_remaining:
        with ProcessPoolExecutor(max_workers=len(workers_with_remaining)) as executor:
            futures = [
                executor.submit(
                    worker,
                    worker_id=w_id,
                    run_indices=runs,
                    model_name=model_name,
                    output_dir=output_dir,
                    data_config=data_config,
                    config=config,
                )
                for w_id, runs in workers_with_remaining
            ]

            for future in as_completed(futures):
                try:
                    logs = future.result()  # No timeout
                    all_logs.update(logs)
                    print(f"Collected logs. Total runs so far: {len(all_logs)}/{n_runs}")
                except Exception as e:
                    error_msg = f"ERROR: A worker failed with {type(e).__name__}: {e}"
                    print(error_msg)
                    print(f"Full traceback:")
                    import traceback
                    traceback.print_exc()
                    print(f"Continuing to collect results from other workers...")

    elapsed = time.time() - start_time

    save_npz(f"{timestamp}_{model_name}_logs.npz", data=all_logs, output_dir=output_dir)

    print("\n=== Data collection finished ===")
    print(f"Total elapsed time: {elapsed:.2f} seconds")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model by name")
    parser.add_argument("model", type=str, help="Name of the model to run")
    args = parser.parse_args()

    print(f"\nStarting data collection for model: {args.model}")
    run_data_collector(args.model)
    print("\nDone.")