import subprocess

def test():
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest`
    """
    subprocess.run(
        ['python', '-u', '-m', 'unittest']
    )
