import subprocess

# List of Python files to run
files = ['DistillBERT-model-pro.py', 'DistillBERT-model-res.py','DistillBERT-model-IMDB.py']

# Open a file to store the results
with open('results-DistillBERT-model.txt', 'w') as f:
    # Loop through each file
    for file in files:
        try:
            # Run the Python file and capture the output
            output = subprocess.check_output(['python3', file], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            # Handle any subprocess errors
            output = e.output
        # Write the output to the results file
        f.write(f'Results for {file}:\n\n{output.decode()}\n\n')

