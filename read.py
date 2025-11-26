import numpy as np
import sys

# 1. Define filenames
input_file = "data/data_07_3cl_ez.npz"
output_file = "data_dump.txt"

# 2. Load the .npz file
try:
    loaded_data = np.load(input_file)
    print(f"Successfully loaded {input_file}")
except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
    sys.exit()

# 3. Open a text file to write the results
with open(output_file, "w") as f:
    # Force numpy to print the entire array, not just [1, 2, ..., 99]
    # If the data is massive, this creates a large text file.
    # You can remove this line to keep the default truncation.
    np.set_printoptions(threshold=np.inf)

    f.write(f"--- Content of {input_file} ---\n\n")

    # 4. Iterate through each array in the file
    for key in loaded_data.files:
        array = loaded_data[key]

        # Write Header for the array
        f.write(f"NAME: '{key}'\n")
        f.write(f"SHAPE: {array.shape}\n")
        f.write(f"DATA TYPE: {array.dtype}\n")
        f.write("-" * 20 + "\n")

        # Write the actual data
        f.write(str(array))

        # Add spacing between arrays
        f.write("\n\n" + "=" * 40 + "\n\n")

print(f"Done! Data has been written to {output_file}")
