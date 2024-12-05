import torch
import os

def convert_pt_to_text(pt_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all .pt files in the folder
    for pt_file in os.listdir(pt_folder):
        if pt_file.endswith('.pt'):
            # Load the .pt file
            pt_path = os.path.join(pt_folder, pt_file)
            data = torch.load(pt_path)
            
            # Convert data to a readable format
            text_data = []
            text_data.append(f"Nodes (x):\n{data.x.tolist()}\n")
            text_data.append(f"Edge Index:\n{data.edge_index.tolist()}\n")
            text_data.append(f"Edge Attributes:\n{data.edge_attr.tolist()}\n")
            
            # Write to a text file
            output_file = os.path.join(output_folder, pt_file.replace('.pt', '.txt'))
            with open(output_file, 'w') as f:
                f.writelines(text_data)

            print(f"Converted {pt_file} to {output_file}")

# Example usage
pt_folder = 'pt_new_dataset_test'  # Input folder containing .pt files
output_folder = 'text_new_dataset_test'     # Output folder to save text files
convert_pt_to_text(pt_folder, output_folder)