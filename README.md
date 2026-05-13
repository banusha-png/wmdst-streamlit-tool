# WMDST - Waste Mapping Decision Support Tool

This is a Streamlit web app version of the Excel-based Waste Mapping Decision Support Tool.

## What this tool does

The tool allows a company user to enter production data through forms and automatically generate:

- KPI waste calculations
- Waiting time
- Inventory days
- Value-added ratio
- Defect loss
- Scrap and rework rates
- WAM waste type ranking
- AHP department priority ranking
- Improvement recommendations
- Executive dashboard
- Excel/CSV export

## Folder files

- `app.py` - Main Streamlit application
- `requirements.txt` - Required Python packages
- `.streamlit/config.toml` - Basic app theme settings

## How to run locally

### Step 1 - Install Python

Install Python 3.10 or above.

### Step 2 - Create project folder

Create a folder called:

```bash
wmdst_tool
```

Place `app.py` and `requirements.txt` inside that folder.

### Step 3 - Open terminal in the folder

In Windows, open the folder, click the address bar, type `cmd`, and press Enter.

### Step 4 - Install packages

```bash
pip install -r requirements.txt
```

### Step 5 - Run the app

```bash
streamlit run app.py
```

The app will open in your browser.

## How a company user will use it

1. Open the app link.
2. Go to Production Data Form.
3. Enter department-wise production data.
4. Go to WAM Questionnaire and enter waste scores.
5. Go to AHP Input and select pairwise comparison values.
6. Open Dashboard.
7. Download results from Recommendations & Export.

## Deployment option

For public sharing, upload the files to GitHub and deploy through Streamlit Community Cloud.

## Important note

The current formula weights in the app are adjustable. If your supervisor/company wants the exact same weighting as the Excel workbook, update the weight values inside the `calculate_kpis()` function.