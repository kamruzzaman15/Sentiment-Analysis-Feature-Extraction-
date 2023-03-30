import subprocess

# List of Python files to run
files = ['RoBERTa-base-model-pro.py', 'RoBERTa-base-model-res.py','RoBERTa-base-model-IMDB.py']

# Open a file to store the results
with open('results-RoBERTa-base-model.txt', 'w') as f:
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

