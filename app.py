from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, make_response
from werkzeug.utils import secure_filename
from pathlib import Path
from functools import wraps
import pandas as pd
import numpy as np
import os, uuid, io, sqlite3, csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-retailiq-secret-before-production")
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "retailiq_users.db"
ALLOWED_EXTENSIONS = {"csv"}

ALGORITHMS = {
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors (KNN)",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "kmeans": "K-Means Customer Clustering",
    "apriori": "Apriori Product Associations",
}

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
init_db()

def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access your dashboard.", "error")
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapped

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def read_retail_csv(path):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The CSV file is empty.")
    normalized = {str(c).strip().lower().replace(" ", "").replace("_", "").replace("-", ""): c for c in df.columns}
    aliases = {
        "invoiceno": ["invoiceno", "invoice", "orderid", "transactionid", "transaction"],
        "stockcode": ["stockcode", "productid", "itemid", "sku"],
        "description": ["description", "product", "productname", "itemname", "item"],
        "quantity": ["quantity", "qty", "units", "unitsold"],
        "invoicedate": ["invoicedate", "date", "orderdate", "transactiondate", "saledate"],
        "unitprice": ["unitprice", "price", "sellingprice", "salesprice", "unitcost"],
        "customerid": ["customerid", "customer", "clientid", "buyerid"],
        "country": ["country", "region", "location"],
        "stockquantity": ["stockquantity", "stock", "inventory", "inventoryquantity", "currentstock"],
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
        amount_col = next((c for c in df.columns if str(c).strip().lower() in ("sales", "revenue", "total", "amount", "totalamount", "salesvalue")), None)
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
        raise ValueError("No valid rows found. Check Quantity and UnitPrice contain positive numeric values.")
    df["sales_value"] = df["quantity"] * df["unitprice"]
    if "description" not in df.columns:
        df["description"] = df["stockcode"].astype(str) if "stockcode" in df.columns else [f"Product {i+1}" for i in range(len(df))]
    if "invoiceno" not in df.columns:
        df["invoiceno"] = [f"INV{i+1:05d}" for i in range(len(df))]
    if "customerid" not in df.columns:
        df["customerid"] = [f"C{i % max(1, min(20, len(df))):04d}" for i in range(len(df))]
    if "invoicedate" in df.columns:
        df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    if "stockquantity" in df.columns:
        df["stockquantity"] = pd.to_numeric(df["stockquantity"], errors="coerce")
    return df

def dataset_path():
    path = session.get("dataset_path")
    if path and Path(path).exists():
        return Path(path)
    return None

def get_df():
    path = dataset_path()
    if path is None:
        return None
    return read_retail_csv(path)

def classification_result(df, algorithm):
    data = df.copy()
    median = data["sales_value"].median()
    data["sales_category"] = (data["sales_value"] > median).astype(int)
    features = data[["quantity", "unitprice"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    target = data["sales_category"]
    if target.nunique() < 2:
        return {"title": ALGORITHMS[algorithm], "summary": "The dataset needs a wider range of sales values to create both high and low sales categories.", "metrics": [], "rows": []}
    if len(data) < 8:
        return {"title": ALGORITHMS[algorithm], "summary": "Upload at least 8 valid sales rows to run this algorithm.", "metrics": [], "rows": []}
    stratify = target if target.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42, stratify=stratify)
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
    preview = data.loc[X_test.index, ["quantity", "unitprice", "sales_value"]].copy()
    preview["Actual"] = np.where(y_test.to_numpy() == 1, "High", "Low")
    preview["Predicted"] = np.where(pred == 1, "High", "Low")
    rows = [{"quantity": f"{r.quantity:g}", "price": f"{r.unitprice:.2f}", "sales": f"{r.sales_value:.2f}", "actual": r.Actual, "predicted": r.Predicted} for r in preview.head(30).itertuples()]
    return {"title": ALGORITHMS[algorithm], "summary": "Classifies records into high/low sales-value categories based on quantity and unit price. This is a demonstration task, not a business guarantee.",
            "metrics": [("Model accuracy", f"{accuracy_score(y_test, pred)*100:.1f}%"), ("Training records", str(len(X_train)),), ("Test records", str(len(X_test)))], "rows": rows}

def customer_clusters(df, n_clusters=3):
    customers = df.groupby("customerid", dropna=False).agg(
        total_spend=("sales_value", "sum"), total_quantity=("quantity", "sum"),
        order_count=("invoiceno", "nunique")).reset_index()
    if len(customers) < 2:
        return customers.assign(segment=0), 1
    k = max(2, min(int(n_clusters), len(customers)))
    features = customers[["total_spend", "total_quantity", "order_count"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    customers["segment"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    return customers, k

def kmeans_result(df):
    customers, k = customer_clusters(df, 3)
    if len(customers) < 2:
        return {"title": ALGORITHMS["kmeans"], "summary": "At least 2 distinct customers are needed for customer clustering.", "metrics": [], "rows": []}
    labels = {i: f"Cluster {i+1}" for i in sorted(customers["segment"].unique())}
    counts = customers["segment"].value_counts()
    rows = [{"customer": str(r.customerid), "segment": labels[r.segment], "spend": f"{r.total_spend:.2f}", "quantity": f"{r.total_quantity:g}", "orders": str(int(r.order_count))} for r in customers.head(100).itertuples()]
    return {"title": ALGORITHMS["kmeans"], "summary": "K-Means groups customers using total spend, quantity purchased, and distinct invoices. Cluster numbers are labels, not rankings.",
            "metrics": [("Customers analyzed", str(len(customers))), ("Clusters created", str(k))], "rows": rows, "table_type": "customers",
            "segments": [{"name": labels[i], "count": int(counts.get(i, 0))} for i in sorted(labels)],
            "chart_labels": [labels[i] for i in sorted(labels)], "chart_values": [int(counts.get(i, 0)) for i in sorted(labels)]}

def apriori_result(df):
    basket = df.pivot_table(index="invoiceno", columns="description", values="quantity", aggfunc="sum", fill_value=0).gt(0)
    if basket.shape[0] < 2 or basket.shape[1] < 2:
        return {"title": ALGORITHMS["apriori"], "summary": "Apriori needs at least 2 invoices and 2 different products.", "metrics": [], "rows": []}
    try:
        frequent = apriori(basket, min_support=max(0.05, 1/max(1, len(basket))), use_colnames=True)
        if frequent.empty:
            return {"title": ALGORITHMS["apriori"], "summary": "No frequent product combinations found. Try a dataset with repeated purchases.", "metrics": [], "rows": []}
        rules = association_rules(frequent, metric="confidence", min_threshold=0.1)
        if rules.empty:
            return {"title": ALGORITHMS["apriori"], "summary": "Frequent itemsets found, but no rules met the minimum confidence.", "metrics": [("Frequent itemsets", str(len(frequent)))], "rows": []}
        rules = rules.sort_values(["confidence", "lift"], ascending=False).head(30)
        rows = [{"antecedent": ", ".join(map(str, sorted(r.antecedents))), "consequent": ", ".join(map(str, sorted(r.consequents))), "support": f"{r.support:.3f}", "confidence": f"{r.confidence:.3f}", "lift": f"{r.lift:.2f}"} for r in rules.itertuples()]
        return {"title": ALGORITHMS["apriori"], "summary": "Rules show product combinations observed in invoices. Association does not prove cause and effect.", "metrics": [("Invoices analyzed", str(len(basket))), ("Rules displayed", str(len(rows)))], "rows": rows, "table_type": "rules"}
    except Exception as exc:
        return {"title": ALGORITHMS["apriori"], "summary": f"Could not calculate association rules: {exc}", "metrics": [], "rows": []}

def dashboard_data(df):
    products = df.groupby("description", as_index=False).agg(total_sales=("sales_value", "sum"), units_sold=("quantity", "sum")).sort_values("total_sales", ascending=False)
    monthly = []
    if "invoicedate" in df.columns and df["invoicedate"].notna().any():
        tmp = df.dropna(subset=["invoicedate"]).copy()
        tmp["month"] = tmp["invoicedate"].dt.to_period("M").astype(str)
        monthly = tmp.groupby("month")["sales_value"].sum().sort_index()
    customers, _ = customer_clusters(df, 3)
    return {
        "total_sales": float(df["sales_value"].sum()),
        "total_units": float(df["quantity"].sum()),
        "orders": int(df["invoiceno"].nunique()),
        "customers": int(df["customerid"].nunique()),
        "product_names": products.head(10)["description"].astype(str).tolist(),
        "product_sales": [round(float(x), 2) for x in products.head(10)["total_sales"]],
        "months": monthly.index.tolist() if len(monthly) else [],
        "monthly_sales": [round(float(x), 2) for x in monthly.tolist()] if len(monthly) else [],
        "top_products": [{"name": str(r.description), "sales": f"{r.total_sales:.2f}", "units": f"{r.units_sold:g}"} for r in products.head(10).itertuples()],
        "inventory_available": "stockquantity" in df.columns and df["stockquantity"].notna().any()
    }

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 3 or len(password) < 8:
            flash("Username must be at least 3 characters and password at least 8 characters.", "error")
            return render_template("auth.html", mode="register")
        from werkzeug.security import generate_password_hash
        try:
            with db() as conn:
                conn.execute("INSERT INTO users(username,password_hash) VALUES (?,?)", (username, generate_password_hash(password)))
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("That username is already registered.", "error")
    return render_template("auth.html", mode="register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        from werkzeug.security import check_password_hash
        with db() as conn:
            user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            nxt = request.args.get("next")
            return redirect(nxt if nxt and nxt.startswith("/") else url_for("home"))
        flash("Incorrect username or password.", "error")
    return render_template("auth.html", mode="login")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Please choose a CSV file first.", "error")
            return redirect(url_for("home"))
        if not allowed_file(file.filename):
            flash("Please upload a CSV file. This retail version accepts CSV.", "error")
            return redirect(url_for("home"))
        filename = secure_filename(file.filename)
        saved = UPLOAD_DIR / f"{session['user_id']}_{uuid.uuid4().hex}_{filename}"
        file.save(saved)
        try:
            df = read_retail_csv(saved)
            session["dataset_path"] = str(saved.resolve())
            session["dataset_name"] = filename
            session["row_count"] = len(df)
            flash(f"Uploaded successfully: {filename} ({len(df)} valid rows).", "success")
            return redirect(url_for("dashboard"))
        except Exception as exc:
            saved.unlink(missing_ok=True)
            flash(f"Could not read this CSV: {exc}", "error")
            return redirect(url_for("home"))
    return render_template("index.html")

@app.route("/dashboard")
@login_required
def dashboard():
    df = get_df()
    if df is None:
        flash("Upload a retail CSV dataset first.", "error")
        return redirect(url_for("home"))
    return render_template("dashboard.html", data=dashboard_data(df), dataset_name=session.get("dataset_name", "retail.csv"))

@app.route("/choose-algorithm", methods=["GET", "POST"])
@login_required
def choose_algorithm():
    try:
        df = get_df()
    except Exception as exc:
        flash(f"Could not load dataset: {exc}", "error")
        return redirect(url_for("home"))
    if df is None:
        flash("Upload a retail CSV dataset first.", "error")
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
@login_required
def run_algorithm():
    df = get_df()
    algorithm = session.get("algorithm")
    if df is None or algorithm not in ALGORITHMS:
        flash("Upload a dataset and choose an algorithm first.", "error")
        return redirect(url_for("home"))
    try:
        if algorithm in {"decision_tree", "knn", "naive_bayes", "random_forest"}:
            result = classification_result(df, algorithm)
        elif algorithm == "kmeans":
            result = kmeans_result(df)
        else:
            result = apriori_result(df)
        session["last_result"] = result
        return render_template("results.html", result=result, dataset_name=session.get("dataset_name", "retail.csv"))
    except Exception as exc:
        flash(f"Algorithm could not run: {exc}", "error")
        return redirect(url_for("choose_algorithm"))

@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    df = get_df()
    if df is None:
        flash("Upload a dataset first.", "error")
        return redirect(url_for("home"))
    if request.method == "POST":
        try:
            quantity = float(request.form.get("quantity", "1"))
            price = float(request.form.get("price", "0"))
            if quantity <= 0 or price < 0:
                raise ValueError("Quantity must be positive and price cannot be negative.")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            X = df[["quantity", "unitprice"]].replace([np.inf, -np.inf], np.nan).fillna(0)
            model.fit(X, df["sales_value"])
            estimate = float(model.predict(pd.DataFrame([{"quantity": quantity, "unitprice": price}]))[0])
            return render_template("prediction.html", dataset_name=session.get("dataset_name"), prediction=f"{estimate:.2f}", quantity=quantity, price=price)
        except Exception as exc:
            flash(f"Prediction could not be calculated: {exc}", "error")
    return render_template("prediction.html", dataset_name=session.get("dataset_name"), prediction=None, quantity=1, price=0)

@app.route("/inventory")
@login_required
def inventory():
    df = get_df()
    if df is None:
        flash("Upload a dataset first.", "error")
        return redirect(url_for("home"))
    products = df.groupby("description", as_index=False).agg(units_sold=("quantity", "sum"), sales=("sales_value", "sum"))
    if "stockquantity" in df.columns and df["stockquantity"].notna().any():
        stock = df.groupby("description")["stockquantity"].last()
        products["stock"] = products["description"].map(stock)
        products["status"] = np.where(products["stock"].fillna(0) <= 10, "Low stock", "Stock available")
    else:
        products["stock"] = np.nan
        products["status"] = "Stock column not provided"
    products = products.sort_values("sales", ascending=False).head(100)
    rows = [{"name": str(r.description), "units": f"{r.units_sold:g}", "sales": f"{r.sales:.2f}", "stock": "—" if pd.isna(r.stock) else f"{r.stock:g}", "status": r.status} for r in products.itertuples()]
    return render_template("inventory.html", rows=rows, has_stock="stockquantity" in df.columns and df["stockquantity"].notna().any())

@app.route("/clusters")
@login_required
def clusters():
    df = get_df()
    if df is None:
        flash("Upload a dataset first.", "error")
        return redirect(url_for("home"))
    try:
        requested_k = max(2, min(int(request.args.get("k", 3)), 8))
    except ValueError:
        requested_k = 3
    customers, k = customer_clusters(df, requested_k)
    labels = {i: f"Cluster {i+1}" for i in sorted(customers["segment"].unique())}
    counts = customers["segment"].value_counts()
    rows = [{"customer": str(r.customerid), "cluster": labels[r.segment], "spend": f"{r.total_spend:.2f}", "quantity": f"{r.total_quantity:g}", "orders": int(r.order_count)} for r in customers.head(200).itertuples()]
    return render_template("clusters.html", rows=rows, k=k, labels=[labels[i] for i in sorted(labels)], values=[int(counts.get(i,0)) for i in sorted(labels)], total=len(customers))

@app.route("/download/csv")
@login_required
def download_csv():
    result = session.get("last_result")
    if not result:
        flash("Run an algorithm first to download its results.", "error")
        return redirect(url_for("choose_algorithm"))
    rows = result.get("rows", [])
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    else:
        output.write("message\nNo result rows available\n")
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=retailiq_results.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response

@app.route("/download/pdf")
@login_required
def download_pdf():
    result = session.get("last_result")
    if not result:
        flash("Run an algorithm first to download its results.", "error")
        return redirect(url_for("choose_algorithm"))
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), title="RetailIQ Analysis Report")
        styles = getSampleStyleSheet()
        story = [Paragraph("RetailIQ Analysis Report", styles["Title"]), Spacer(1, 10),
                 Paragraph(str(result.get("title", "Algorithm Results")), styles["Heading2"]),
                 Paragraph(str(result.get("summary", ""))[:1000], styles["BodyText"]), Spacer(1, 12)]
        metrics = result.get("metrics", [])
        if metrics:
            story.append(Table([["Metric", "Value"]] + [[str(x[0]), str(x[1])] for x in metrics], hAlign="LEFT"))
            story.append(Spacer(1, 12))
        rows = result.get("rows", [])
        if rows:
            headers = list(rows[0].keys())
            table_data = [[str(h).replace("_", " ").title() for h in headers]]
            table_data += [[str(r.get(h, ""))[:70] for h in headers] for r in rows[:100]]
            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#087f73")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.lightgrey),
                ("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
            story.append(table)
        doc.build(story)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="retailiq_report.pdf", mimetype="application/pdf")
    except ImportError:
        flash("PDF dependency is missing. Run: python -m pip install reportlab", "error")
        return redirect(url_for("run_algorithm"))

if __name__ == "__main__":
    app.run(debug=True)
