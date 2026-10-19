from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd
import numpy as np
import os
import uuid

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
app.secret_key = "retailiq-local-demo-change-this"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"csv"}

ALGORITHMS = {
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors (KNN)",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "kmeans": "K-Means Customer Clustering",
    "apriori": "Apriori Product Associations",
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def read_retail_csv(path):
    df = pd.read_csv(path)
    # Normalize common column names without changing original values.
    normalized = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    aliases = {
        "invoiceno": ["invoiceno", "invoice", "orderid", "transactionid"],
        "stockcode": ["stockcode", "productid", "itemid", "sku"],
        "description": ["description", "product", "productname", "itemname"],
        "quantity": ["quantity", "qty", "units"],
        "invoicedate": ["invoicedate", "date", "orderdate", "transactiondate"],
        "unitprice": ["unitprice", "price", "sellingprice", "salesprice"],
        "customerid": ["customerid", "customer", "clientid"],
        "country": ["country", "region", "location"],
    }
    rename = {}
    for target, choices in aliases.items():
        for choice in choices:
            if choice in normalized:
                rename[normalized[choice]] = target
                break
    df = df.rename(columns=rename)
    if "quantity" not in df.columns:
        df["quantity"] = 1
    if "unitprice" not in df.columns:
        # If the file already has a total amount column, use it as unitprice for analysis.
        amount_col = next((c for c in df.columns if str(c).strip().lower() in ("sales", "revenue", "total", "amount", "totalamount")), None)
        if amount_col:
            df["unitprice"] = pd.to_numeric(df[amount_col], errors="coerce")
            df["quantity"] = 1
        else:
            raise ValueError("CSV needs a UnitPrice/Price/Sales/Revenue column.")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unitprice"] = pd.to_numeric(df["unitprice"], errors="coerce")
    df = df.dropna(subset=["quantity", "unitprice"])
    df = df[(df["quantity"] > 0) & (df["unitprice"] >= 0)].copy()
    if df.empty:
        raise ValueError("No valid rows found. Check that Quantity and UnitPrice contain positive numeric values.")
    df["sales_value"] = df["quantity"] * df["unitprice"]
    if "description" not in df.columns:
        if "stockcode" in df.columns:
            df["description"] = df["stockcode"].astype(str)
        else:
            df["description"] = [f"Product {i+1}" for i in range(len(df))]
    if "invoiceno" not in df.columns:
        df["invoiceno"] = [f"INV{i+1:05d}" for i in range(len(df))]
    if "customerid" not in df.columns:
        df["customerid"] = [f"C{i % max(1, min(20, len(df))):04d}" for i in range(len(df))]
    return df

def classification_result(df, algorithm):
    # Predict whether an individual row has high or low sales value.
    data = df.copy()
    median = data["sales_value"].median()
    data["sales_category"] = (data["sales_value"] > median).astype(int)
    data["price_per_unit"] = data["unitprice"]
    data["quantity_feature"] = data["quantity"]
    features = data[["quantity_feature", "price_per_unit"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    target = data["sales_category"]
    if target.nunique() < 2:
        return {"title": ALGORITHMS[algorithm], "summary": "The dataset needs a wider range of sales values to create both high and low sales categories.", "metrics": [], "rows": []}
    if len(data) < 8:
        return {"title": ALGORITHMS[algorithm], "summary": "Upload at least 8 valid sales rows to run this algorithm.", "metrics": [], "rows": []}
    stratify = target if target.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=stratify
    )
    if algorithm == "decision_tree":
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif algorithm == "knn":
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, len(X_train))))
    elif algorithm == "naive_bayes":
        model = GaussianNB()
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    result_rows = []
    preview = data.loc[X_test.index, ["quantity", "unitprice", "sales_value"]].copy()
    preview["Actual"] = np.where(y_test.to_numpy() == 1, "High", "Low")
    preview["Predicted"] = np.where(pred == 1, "High", "Low")
    for _, row in preview.head(10).iterrows():
        result_rows.append({
            "quantity": f"{row['quantity']:g}",
            "price": f"{row['unitprice']:.2f}",
            "sales": f"{row['sales_value']:.2f}",
            "actual": row["Actual"],
            "predicted": row["Predicted"]
        })
    accuracy = accuracy_score(y_test, pred) * 100
    return {
        "title": ALGORITHMS[algorithm],
        "summary": "The model classifies sales records into High and Low sales-value categories. These are demonstration predictions, not business guarantees.",
        "metrics": [("Model accuracy", f"{accuracy:.1f}%"), ("Training records", str(len(X_train))), ("Test records", str(len(X_test)))],
        "rows": result_rows
    }

def kmeans_result(df):
    customers = df.groupby("customerid", dropna=False).agg(
        total_spend=("sales_value", "sum"),
        total_quantity=("quantity", "sum"),
        order_count=("invoiceno", "nunique")
    ).reset_index()
    if len(customers) < 3:
        return {"title": ALGORITHMS["kmeans"], "summary": "At least 3 distinct customers are needed for customer clustering.", "metrics": [], "rows": []}
    X = customers[["total_spend", "total_quantity", "order_count"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    k = min(3, len(customers))
    customers["segment"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(X))
    labels = {i: f"Segment {i+1}" for i in sorted(customers["segment"].unique())}
    rows = []
    for _, r in customers.head(15).iterrows():
        rows.append({"customer": str(r["customerid"]), "segment": labels[r["segment"]], "spend": f"{r['total_spend']:.2f}", "quantity": f"{r['total_quantity']:g}", "orders": str(int(r["order_count"]))})
    counts = customers["segment"].value_counts()
    return {
        "title": ALGORITHMS["kmeans"],
        "summary": "Customers are grouped by total spend, quantity purchased, and number of invoices. Segment numbers are labels, not rankings.",
        "metrics": [("Customers analyzed", str(len(customers))), ("Segments created", str(k))],
        "rows": rows,
        "table_type": "customers",
        "segments": [{"name": labels[i], "count": int(counts.get(i, 0))} for i in sorted(labels)]
    }

def apriori_result(df):
    basket = df.pivot_table(index="invoiceno", columns="description", values="quantity", aggfunc="sum", fill_value=0)
    basket = basket.gt(0)
    if basket.shape[0] < 2 or basket.shape[1] < 2:
        return {"title": ALGORITHMS["apriori"], "summary": "Apriori needs at least 2 invoices and 2 different products.", "metrics": [], "rows": []}
    try:
        frequent = apriori(basket, min_support=max(0.05, 1 / max(1, len(basket))), use_colnames=True)
        if frequent.empty:
            return {"title": ALGORITHMS["apriori"], "summary": "No frequent product combinations were found. Try a dataset with more repeated purchases.", "metrics": [], "rows": []}
        rules = association_rules(frequent, metric="confidence", min_threshold=0.1)
        if rules.empty:
            return {"title": ALGORITHMS["apriori"], "summary": "Frequent itemsets were found, but no association rules met the minimum confidence.", "metrics": [("Frequent itemsets", str(len(frequent)))], "rows": []}
        rules = rules.sort_values(["confidence", "lift"], ascending=False).head(15)
        rows = []
        for _, r in rules.iterrows():
            rows.append({"antecedent": ", ".join(map(str, sorted(r["antecedents"]))), "consequent": ", ".join(map(str, sorted(r["consequents"]))), "support": f"{r['support']:.3f}", "confidence": f"{r['confidence']:.3f}", "lift": f"{r['lift']:.2f}"})
        return {"title": ALGORITHMS["apriori"], "summary": "Rules show product combinations observed in invoices. Confidence and lift describe associations, not cause and effect.", "metrics": [("Invoices analyzed", str(len(basket))), ("Rules displayed", str(len(rows)))], "rows": rows, "table_type": "rules"}
    except Exception as exc:
        return {"title": ALGORITHMS["apriori"], "summary": f"Could not calculate association rules: {exc}", "metrics": [], "rows": []}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Please choose a CSV file first.", "error")
            return redirect(url_for("home"))
        if not allowed_file(file.filename):
            flash("Please upload a CSV file. This retail version accepts CSV, not ARFF.", "error")
            return redirect(url_for("home"))
        filename = secure_filename(file.filename)
        saved = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
        file.save(saved)
        try:
            df = read_retail_csv(saved)
            session["dataset_path"] = str(saved.resolve())
            session["dataset_name"] = filename
            session["row_count"] = len(df)
            session["columns"] = list(map(str, df.columns))
            flash(f"Uploaded successfully: {filename} ({len(df)} valid rows).", "success")
            return redirect(url_for("choose_algorithm"))
        except Exception as exc:
            saved.unlink(missing_ok=True)
            flash(f"Could not read this CSV: {exc}", "error")
            return redirect(url_for("home"))
    return render_template("index.html")

@app.route("/choose-algorithm", methods=["GET", "POST"])
def choose_algorithm():
    path = session.get("dataset_path")
    if not path or not Path(path).exists():
        flash("Upload a retail CSV dataset first.", "error")
        return redirect(url_for("home"))
    try:
        df = read_retail_csv(path)
    except Exception as exc:
        flash(f"Could not load dataset: {exc}", "error")
        return redirect(url_for("home"))
    if request.method == "POST":
        algorithm = request.form.get("algorithm", "")
        if algorithm not in ALGORITHMS:
            flash("Please choose an algorithm.", "error")
            return redirect(url_for("choose_algorithm"))
        session["algorithm"] = algorithm
        return redirect(url_for("run_algorithm"))
    return render_template("choose_algorithm.html", dataset_name=session.get("dataset_name", "retail.csv"), row_count=len(df), columns=list(df.columns), algorithms=ALGORITHMS)

@app.route("/run-algorithm")
def run_algorithm():
    path = session.get("dataset_path")
    algorithm = session.get("algorithm")
    if not path or not Path(path).exists() or algorithm not in ALGORITHMS:
        flash("Upload a dataset and choose an algorithm first.", "error")
        return redirect(url_for("home"))
    try:
        df = read_retail_csv(path)
        if algorithm in {"decision_tree", "knn", "naive_bayes", "random_forest"}:
            result = classification_result(df, algorithm)
        elif algorithm == "kmeans":
            result = kmeans_result(df)
        else:
            result = apriori_result(df)
        return render_template("results.html", result=result, dataset_name=session.get("dataset_name", "retail.csv"))
    except Exception as exc:
        flash(f"Algorithm could not run: {exc}", "error")
        return redirect(url_for("choose_algorithm"))

if __name__ == "__main__":
    app.run(debug=True)
