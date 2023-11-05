import subprocess

# List of Python files to run
files = ['CBOW-feature-svm-pro.py', 'CBOW-feature-vot-pro.py', 'CBOW-feature-boost-pro.py', 'CBOW-feature-svm-res.py','CBOW-feature-vot-res.py','CBOW-feature-boost-res.py','CBOW-feature-svm-IMDB.py','CBOW-feature-vot-IMDB.py','CBOW-feature-boost-IMDB.py']

# Open a file to store the results
with open('results-CBOW.txt', 'w') as f:
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

