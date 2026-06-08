# Linear & Logistic Regression Examples

This repository contains simple Python examples for linear regression and logistic regression using a small medical dataset.

## Contents

- `linear_regression.py`: Example implementation of linear regression.
- `logistic_reg.py`: Example implementation of logistic regression.
- `medical_data_logi.csv`: Sample dataset used by the scripts.
- `requirement.txt`: Python dependencies for the project.

## Requirements

- Python 3.8+
- A virtual environment is recommended

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirement.txt
```

Note: the repository's dependency file is named `requirement.txt`.

## Usage

- Run the linear regression example:

```powershell
python linear_regression.py
```

- Run the logistic regression example:

```powershell
python logistic_reg.py
```

Both scripts load `medical_data_logi.csv` from the repository root; ensure the file is present.

## Files Description

- `linear_regression.py` — demonstrates fitting and evaluating a linear regression model and prints results/metrics.
- `logistic_reg.py` — demonstrates fitting and evaluating a logistic regression model for binary classification.
- `medical_data_logi.csv` — CSV dataset used by the examples. Inspect the file to understand available columns.
- `requirement.txt` — list of Python packages used by the examples.

## Notes

- These examples are educational and intended for small datasets and demonstrations. They are not production-ready.
- If your CSV location differs, update the scripts or run them from the repository root.

## Contributing

Feel free to open issues or submit pull requests to improve examples, add tests, or expand datasets.

## License

No license specified. Add a `LICENSE` file if you wish to set terms for reuse.
