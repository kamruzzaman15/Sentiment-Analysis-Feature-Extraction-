import subprocess

# List of Python files to run
files = ['DWE-feature-svm-pro.py', 'DWE-feature-vot-pro.py', 'DWE-feature-boost-pro.py', 'DWE-feature-svm-res.py','DWE-feature-vot-res.py','DWE-feature-boost-res.py','DWE-feature-svm-IMDB.py','DWE-feature-vot-IMDB.py','DWE-feature-boost-IMDB.py']

# Open a file to store the results
with open('results-DWE.txt', 'w') as f:
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

