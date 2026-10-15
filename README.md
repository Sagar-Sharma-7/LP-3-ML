# Machine Learning Assignments

Each assignment has its own self-contained folder and Python program. Install the dependencies once, then run the commands below from this folder:

```powershell
py -m pip install -r requirements.txt
py assignment_01_uber_regression/run_assignment.py --download
py assignment_02_email_classification/run_assignment.py --download
py assignment_03_bank_churn_ann/run_assignment.py --download
py assignment_04_diabetes_knn/run_assignment.py --download
py assignment_05_sales_clustering/run_assignment.py --download
```

If Kaggle blocks the automatic download, download the CSV from the link in the relevant assignment document and copy it into that assignment's `data` folder. Then run the same command **without** `--download`. Each program saves charts, metric tables, and a paste-ready `outputs/result.md` file.

Use screenshots of the generated PNG files and copy the generated result text into the Result section of the matching lab manual.
