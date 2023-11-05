import subprocess

# List of Python files to run
files = ['BoW-feature-svm-pro.py', 'BoW-feature-vot-pro.py', 'BoW-feature-boost-pro.py', 'BoW-feature-svm-res.py','BoW-feature-vot-res.py','BoW-feature-boost-res.py','BoW-feature-svm-IMDB.py','BoW-feature-vot-IMDB.py','BoW-feature-boost-IMDB.py']

# Open a file to store the results
with open('results-BoW.txt', 'w') as f:
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

