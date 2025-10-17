import numpy as np
import os

def save_npz(
    filename,
    compressed=False,
    **arrays
):
    """
    Save NumPy arrays to an .npz file with options.

    Parameters:
    - filename (str): Output file path (should end with .npz)
    - compressed (bool): If True, use np.savez_compressed
    - **arrays: Named arrays to save (e.g., X=X_data, y=y_data)
    """

    # Check file extension
    if not filename.endswith('.npz'):
        raise ValueError("Filename must end with '.npz'")

    # Handle existing file
    if os.path.exists(filename):
        base, ext = os.path.splitext(filename)
        i = 1
        new_filename = f"{base}_{i}{ext}"
        while os.path.exists(new_filename):
            i += 1
            new_filename = f"{base}_{i}{ext}"
        filename = new_filename
        print(f"File exists. Saving as '{filename}' instead.")

    # Save the file
    if compressed:
        np.savez_compressed(filename, **arrays)
    else:
        np.savez(filename, **arrays)

    print(f"Saved file: {filename}")

def load_npz(filename):
    """
    Load arrays from an .npz file and print detailed info.

    Parameters:
    - filename (str): Path to the .npz file

    Returns:
    - dict: Dictionary of NumPy arrays
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist.")

    if not filename.endswith('.npz'):
        raise ValueError("Only .npz files are supported.")

    data = np.load(filename)
    output = {}

    print(f"\n Loaded file: {filename}")
    print(f" Contains {len(data.files)} array(s):\n")

    for key in data.files:
        array = data[key]
        print(f"   Name   : {key}")
        print(f"   Shape  : {array.shape}")
        print(f"   Dtype  : {array.dtype}")
        print()

        output[key] = array

    data.close()
    return output
