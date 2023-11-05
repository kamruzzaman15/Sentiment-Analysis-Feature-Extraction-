import subprocess

# List of Python files to run
files = ['FR-feature-svm-pro.py', 'FR-feature-vot-pro.py', 'FR-feature-boost-pro.py', 'FR-feature-svm-res.py','FR-feature-vot-res.py','FR-feature-boost-res.py','FR-feature-svm-IMDB.py','FR-feature-vot-IMDB.py','FR-feature-boost-IMDB.py']

# Open a file to store the results
with open('results-FR-feature.txt', 'w') as f:
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

