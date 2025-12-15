import numpy as np
import os

def save_npz(
    filename,
    data,
    output_dir,
    sep="/",
    
):
    """
    Save a nested dictionary of NumPy arrays to a .npz file.

    Parameters:
    - filename (str): Base filename without number and extension (e.g., 'logs')
    - data (dict): Nested dictionary like {run_id: {var_name: array}}
    - sep (str): Separator for flattened keys
    - output_dir (str or None): Directory to save the file. If None, save in current directory.
    """

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Strip extension if user passed it
    base_name = filename
    if filename.endswith('.npz'):
        base_name = filename[:-4]
    
    # Build full output path
    full_path = os.path.join(output_dir, f"{base_name}.npz")

    # Flatten the nested dictionary keys
    flat_data = {}
    for outer_key, inner_dict in data.items():
        for inner_key, value in inner_dict.items():
            flat_key = f"{outer_key}{sep}{inner_key}"
            flat_data[flat_key] = value

    # Save the file
    np.savez(full_path, **flat_data)

    print(f"Saved file: {full_path}")

def load_npz(filename, sep="/", input_dir=None):
    """
    Load arrays from a .npz file and reconstruct nested dict structure.

    Parameters:
    - filename (str): Name of the .npz file
    - sep (str): Separator used in flattened keys (default is "/")
    - input_dir (str or None): Directory where the file is located. If None, current directory is used.

    Returns:
    - dict: Nested dictionary of arrays {outer_key: {inner_key: array}}
    """
    if input_dir is not None:
        filename = os.path.join(input_dir, filename)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist.")

    if not filename.endswith('.npz'):
        raise ValueError("Only .npz files are supported.")

    data = np.load(filename, allow_pickle=True)
    output = {}

    for flat_key in data.files:
        array = data[flat_key]

        if sep in flat_key:
            outer_key, inner_key = flat_key.split(sep, 1)
        else:
            outer_key, inner_key = "default", flat_key

        if outer_key not in output:
            output[outer_key] = {}
        output[outer_key][inner_key] = array

    data.close()

    # Print summary
    run_keys = list(output.keys())
    print(f"\nLoaded file: {filename}")
    print(f"Total runs loaded: {len(run_keys)}")
    print(f"Runs: {run_keys}")

    # Print unique inner keys (assuming same keys for all runs)
    if run_keys:
        inner_keys = list(output[run_keys[0]].keys())
        print(f"Variables per run: {inner_keys}")

    return output