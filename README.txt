RetailIQ upgraded project
========================

Included features:
1. Sales dashboard with KPI cards, top-product chart, and monthly trend chart when a date column is available.
2. Sales estimate using a RandomForestRegressor trained on the uploaded CSV.
3. CSV result export and PDF report.
4. Product sales analysis and optional stock alerts when CSV includes StockQuantity, Stock, Inventory, or similar.
5. Login and registration with password hashing and a local SQLite user database.
6. K-Means customer clusters with a selectable cluster count and chart.
7. Existing Decision Tree, KNN, Naive Bayes, Random Forest, K-Means and Apriori algorithms are retained.

Run locally (Windows / VS Code):
1. Extract the ZIP to a new folder. Keep your original project as a backup.
2. Open the extracted folder in VS Code.
3. In Terminal run: python -m pip install -r requirements.txt
4. Run: python app.py
5. Open http://127.0.0.1:5000
6. Create an account, log in, and upload your retail CSV.

Typical CSV columns:
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
Optional inventory column: StockQuantity (or Stock / Inventory).

Important:
- Sales value is Quantity x UnitPrice.
- Sales prediction is an academic estimate from the supplied CSV, not a guaranteed future forecast.
- The low-stock alert uses a simple 10-unit threshold when a stock column exists.
- User accounts are stored in retailiq_users.db locally. For public deployment, set a strong SECRET_KEY and use persistent database storage. Uploaded files and the SQLite DB may not persist on ephemeral hosting unless a persistent disk/database is configured.
- For a public deployment, also configure secure cookies/HTTPS and consider separating uploads by user and implementing upload limits.
- Chart.js is loaded from a CDN, so dashboard charts require internet access in the browser.
