import subprocess

# List of Python files to run
files = ['FastText-feature-svm-pro.py', 'FastText-feature-vot-pro.py', 'FastText-feature-boost-pro.py', 'FastText-feature-svm-res.py','FastText-feature-vot-res.py','FastText-feature-boost-res.py','FastText-feature-svm-IMDB.py','FastText-feature-vot-IMDB.py','FastText-feature-boost-IMDB.py']

# Open a file to store the results
with open('results-FastText.txt', 'w') as f:
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

