RetailIQ algorithm workflow
===========================

Files in this package:
- app.py
- requirements.txt
- templates/index.html
- templates/choose_algorithm.html
- templates/results.html
- static/style.css

How to use:
1. Back up your current app.py, templates/index.html, and static/style.css.
2. Copy these files into your existing retail_analytics folder, preserving the templates and static folders.
3. Install dependencies in the VS Code terminal:
   python -m pip install -r requirements.txt
4. Run:
   python app.py
5. Open http://127.0.0.1:5000

Upload a retail sales CSV file with columns similar to:
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

Supported algorithms:
- Decision Tree, KNN, Naive Bayes, Random Forest (High/Low sales category classification)
- K-Means (customer segmentation)
- Apriori (product associations)

The app accepts CSV files only. This package does not use heart-disease data.
This is an academic demonstration, not a production deployment.
