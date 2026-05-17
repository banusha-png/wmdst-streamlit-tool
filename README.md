# WMDST Improved Streamlit Tool

This is the improved version of the Waste Mapping Decision Support Tool.

## Main improvements added

- More visually appealing Streamlit interface
- Guided single-department input form
- Full table editing option
- CSV template download and CSV upload
- Production data validation warnings
- Better AHP input using guided comparison
- Improved dashboard layout
- Management interpretation section
- Excel export with project information and validation notes
- User guide page

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required files

- app.py
- requirements.txt

## Notes

The current app uses session state. For real company use, a database such as SQLite, PostgreSQL, or Google Sheets can be added later to save all previous analyses permanently.