import subprocess

def test_influx():
    """
    Run unittests for a specific package. Equivalent to:
    `poetry run python -u -m unittest <package_name>`
    """
    subprocess.run(
        ['python', '-m', 'unittest', 'testcases']
    )

def test_navgreen_hist_data():
    """
    Run unittests for a specific package. Equivalent to:
    `poetry run python -u -m unittest <package_name>`
    """
    subprocess.run(
        ['python', '-m', 'unittest', 'hist_data_navgreen_analysis']
    )