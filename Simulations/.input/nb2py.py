import os
import sys
import json

def convert_notebook_to_script(notebook_dir):
    # Remove the .ipynb extension if it is present
    if notebook_dir.endswith(".ipynb"):
        notebook_dir = os.path.splitext(notebook_dir)[0]
    
    # Check if the notebook file exists
    notebook_file = notebook_dir + ".ipynb"
    if not os.path.isfile(notebook_file):
        print(f"Notebook file {notebook_file} does not exist.")
        return
    
    # Load the notebook as a JSON object
    with open(notebook_file, "r") as f:
        notebook = json.load(f)
    
    # Extract the code cells
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    
    # Create a Python script file with the same name as the notebook
    script_file = notebook_dir + ".py"
    
    # Write the code cells to the Python script file
    with open(script_file, "w") as f:
        for cell in code_cells:
            # Concatenate all the strings in the cell["source"] list
            source = "".join(cell["source"])
            if source.strip() == "":
                continue
            f.write(source)
            f.write("\n\n")
    
    #print(f"Notebook {notebook_file} converted to Python script {script_file}.")

if __name__ == "__main__":
    # Get the directory of the notebook file from the command line argument
    if len(sys.argv) != 2:
        print("Usage: python notebook_to_script.py <notebook_directory>")
        sys.exit(1)
    notebook_dir = sys.argv[1]
    
    # Convert the notebook to a Python script
    convert_notebook_to_script(notebook_dir)
