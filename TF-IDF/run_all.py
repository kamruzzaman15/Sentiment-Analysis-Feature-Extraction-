import subprocess

# List of Python files to run
files = ['TF-IDF-feature-svm-pro.py', 'TF-IDF-feature-vot-pro.py', 'TF-IDF-feature-boost-pro.py', 'TF-IDF-feature-svm-res.py','TF-IDF-feature-vot-res.py','TF-IDF-feature-boost-res.py','TF-IDF-feature-svm-IMDB.py','TF-IDF-feature-vot-IMDB.py','TF-IDF-feature-boost-IMDB.py']

# Open a file to store the results
with open('results-TF-IDF.txt', 'w') as f:
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

