# dashboard_ponorogo_autoupdate_fixed_v4.py
from dash import Dash, html, dcc, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from datetime import datetime
import logging

# ------------- Basic app & logging -------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Dashboard Interaktif Kabupaten Ponorogo"

# ------------- Paths & Folder Setup -------------
data_folder = "Database"
os.makedirs(data_folder, exist_ok=True)

# ----------------- 🔄 Auto Convert Excel → CSV -----------------
def convert_excel_to_csv(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    excel_files = [f for f in os.listdir(folder_path) if f.endswith((".xlsx", ".xls"))]

    if excel_files:
        logging.info("Ditemukan file Excel, mulai konversi ke CSV...")
        for excel_file in excel_files:
            excel_path = os.path.join(folder_path, excel_file)
            try:
                df = pd.read_excel(excel_path)
                output_name = os.path.splitext(excel_file)[0] + "_konversi.csv"
                output_path = os.path.join(folder_path, output_name)
                df.to_csv(output_path, sep=",", index=False, encoding="utf-8-sig")
                logging.info(f" - {excel_file} dikonversi menjadi {output_name}")
            except Exception as e:
                logging.error(f"Gagal mengonversi {excel_file}: {e}")
        logging.info("Konversi selesai.")
    else:
        logging.info("Tidak ada file Excel ditemukan, semua data sudah berformat CSV.")

    # Perbarui daftar file CSV
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    return csv_files

# Jalankan konversi otomatis sebelum load data
csv_files = convert_excel_to_csv(data_folder)

# ----------------- GeoJSON -----------------
geojson_path = "35.02_kecamatan.geojson"
if os.path.exists(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)
else:
    geojson = None
geojson_key = "nm_kecamatan"

# ----------------- Utility Functions -----------------
def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return None

def generate_data_quality_report(df):
    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_per_col": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }
    report["numeric_summary"] = df.select_dtypes(include="number").describe().T.reset_index()
    return report

def impute_missing(df, strategy="mean"):
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    for c in num_cols:
        if df[c].isna().sum() > 0:
            if strategy == "mean":
                df[c].fillna(df[c].mean(), inplace=True)
            elif strategy == "median":
                df[c].fillna(df[c].median(), inplace=True)
            elif strategy == "zero":
                df[c].fillna(0, inplace=True)
    for c in df.select_dtypes(exclude="number").columns:
        df[c].fillna("Unknown", inplace=True)
    return df

# ----------------- Data Processing -----------------
def process_data(csv_name, n_clusters, contamination, impute_strategy=None, mask_identifiers=False):
    csv_path = os.path.join(data_folder, csv_name)
    df = safe_read_csv(csv_path)
    if df is None or "Nilai" not in df.columns or "Kecamatan" not in df.columns:
        logging.error("Kolom 'Nilai' atau 'Kecamatan' tidak ditemukan.")
        return None

    # Bersihkan nilai
    df["Nilai"] = df["Nilai"].replace(["-", "", " "], pd.NA)
    df["Nilai"] = pd.to_numeric(df["Nilai"], errors="coerce")

    if df["Nilai"].isna().all():
        logging.error("Semua nilai di kolom 'Nilai' kosong atau tidak valid.")
        return None

    # Imputasi
    if impute_strategy:
        df = impute_missing(df, impute_strategy)
    else:
        df = df.dropna(subset=["Nilai"])

    # Masking identitas
    if mask_identifiers:
        for c in ["Nama", "No_KTP", "NIK", "Alamat"]:
            if c in df.columns:
                df[c] = df[c].astype(str).apply(lambda x: x[:1] + "***" if x and x.lower() != "nan" else x)

    # Analisis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["Nilai"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    cluster_order = df.groupby("Cluster")["Nilai"].mean().sort_values().index.tolist()
    label_map = {i: f"Klaster {rank+1}" for rank, i in enumerate(cluster_order)}
    df["Cluster_Label"] = df["Cluster"].map(label_map)

    iso = IsolationForest(contamination=contamination, random_state=42)
    df["Anomali"] = iso.fit_predict(df[["Nilai"]])
    df["Anomali"] = df["Anomali"].map({1: "Normal", -1: "Anomali"})

    logging.info(f"Dataset '{csv_name}' berhasil diproses ({len(df)} baris).")
    return df.sort_values(by="Nilai", ascending=False).reset_index(drop=True)

# ----------------- Layout -----------------
app.layout = html.Div([
    html.H1("🗺️ Dashboard Interaktif Kabupaten Ponorogo — Auto Update", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Br(),
            html.H3("Pengaturan Analisis"),
            html.Label("Pilih Dataset"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[{"label": f, "value": f} for f in csv_files],
                value=csv_files[0] if csv_files else None,
                style={"width": "100%"}
            ),
            html.Br(),
            html.Label("Jumlah Klaster (KMeans)"),
            dcc.Slider(id="cluster-slider", min=2, max=8, step=1, value=3,
                       marks={i: str(i) for i in range(2, 9)}),
            html.Br(),
            html.Label("Persentase Anomali (IsolationForest)"),
            dcc.Slider(id="contamination-slider", min=0.01, max=0.2, step=0.01, value=0.05,
                       marks={i/100: f"{i}%" for i in range(1, 21, 2)}),
            html.Br(),
            html.Label("Imputasi (Data Wrangling)"),
            dcc.RadioItems(id="impute-strategy", options=[
                {"label": "Tidak", "value": ""},
                {"label": "Mean", "value": "mean"},
                {"label": "Median", "value": "median"},
                {"label": "Zero", "value": "zero"},
            ], value="mean"),
            html.Br(),
            html.Label("Masking Identifiers (Etika Profesi)"),
            dcc.RadioItems(id="mask-ident", options=[
                {"label": "Tidak", "value": False},
                {"label": "Ya (mask kolom sensitif bila ada)", "value": True},
            ], value=False),
            html.Br(),
            html.Div(id="run-timestamp", style={"fontSize": "12px", "color": "#666"})
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

        html.Div([
            html.Div(id="kpi-cards", style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),

            html.H3("🌍 Peta Sebaran Nilai per Kecamatan"),
            dcc.Dropdown(id="map-style", options=[
                {"label": "Carto Positron", "value": "carto-positron"},
                {"label": "Open Street Map", "value": "open-street-map"},
                {"label": "Carto Darkmatter", "value": "carto-darkmatter"}
            ], value="carto-positron", style={"width": "40%"}),
            dcc.Graph(id="map-plot", style={"height": "520px"}),

            dcc.Tabs(id="eda-tabs", value="tab-bar", children=[
                dcc.Tab(label="Perbandingan (Bar)", value="tab-bar"),
                dcc.Tab(label="Distribusi (Histogram)", value="tab-hist"),
                dcc.Tab(label="Scatter / Korelasi", value="tab-scatter"),
                dcc.Tab(label="Ringkasan Statistik", value="tab-summary"),
                dcc.Tab(label="Laporan Kualitas Data", value="tab-quality")
            ]),
            html.Div(id="eda-content", style={"marginTop": "10px"})
        ], style={"width": "68%", "display": "inline-block", "padding": "10px"})
    ], style={"display": "flex", "gap": "2%"}),

    html.Hr(),
    html.H3("📋 Tabel Data (Dengan Klaster & Anomali)"),
    dash_table.DataTable(id="data-table", page_size=12, style_table={'overflowX': 'auto'}),
    html.Div(id="insight-output", style={"marginTop": "20px"}),

    dcc.Store(id="eda-store", storage_type="memory")
], style={"fontFamily": "Arial, sans-serif", "margin": "20px"})

# ----------------- Callback: update dashboard -----------------
@app.callback(
    [Output("run-timestamp", "children"),
     Output("kpi-cards", "children"),
     Output("map-plot", "figure"),
     Output("data-table", "data"),
     Output("data-table", "columns"),
     Output("insight-output", "children"),
     Output("eda-store", "data")],
    [Input("dataset-dropdown", "value"),
     Input("map-style", "value"),
     Input("cluster-slider", "value"),
     Input("contamination-slider", "value"),
     Input("impute-strategy", "value"),
     Input("mask-ident", "value")]
)
def update_dashboard(selected_csv, map_style, n_clusters, contamination, impute_strategy, mask_ident):
    if not selected_csv:
        return "❌ Tidak ada dataset terpilih.", [], {}, [], [], html.Div("❌ Dataset tidak tersedia."), {}

    df = process_data(selected_csv, n_clusters, contamination,
                      impute_strategy if impute_strategy else None, mask_ident)
    if df is None:
        return "❌ Gagal memproses dataset.", [], {}, [], [], html.Div("❌ Dataset tidak valid."), {}

    total_kec = df["Kecamatan"].nunique()
    mean_val = round(df["Nilai"].mean(), 2)
    std_val = round(df["Nilai"].std(), 2)
    anomaly_count = int((df["Anomali"] == "Anomali").sum())

    kpis = [
        html.Div([html.H4("Total Kecamatan"), html.H2(f"{total_kec}")],
                 style={"flex": "1", "padding": "10px", "border": "1px solid #ddd", "borderRadius": "6px", "textAlign": "center"}),
        html.Div([html.H4("Rata-rata Nilai"), html.H2(f"{mean_val}")],
                 style={"flex": "1", "padding": "10px", "border": "1px solid #ddd", "borderRadius": "6px", "textAlign": "center"}),
        html.Div([html.H4("STD Nilai"), html.H2(f"{std_val}")],
                 style={"flex": "1", "padding": "10px", "border": "1px solid #ddd", "borderRadius": "6px", "textAlign": "center"}),
        html.Div([html.H4("Jumlah Anomali"), html.H2(f"{anomaly_count}")],
                 style={"flex": "1", "padding": "10px", "border": "1px solid #ddd", "borderRadius": "6px", "textAlign": "center"})
    ]

    if geojson:
        fig_map = px.choropleth_mapbox(
            df, geojson=geojson, locations="Kecamatan",
            featureidkey=f"properties.{geojson_key}",
            color="Cluster_Label", hover_data=["Kecamatan", "Nilai", "Cluster_Label", "Anomali"],
            mapbox_style=map_style, center={"lat": -7.97, "lon": 111.52}, zoom=9, height=520
        )
    else:
        fig_map = px.bar(df, x="Kecamatan", y="Nilai", color="Cluster_Label", title="GeoJSON tidak ditemukan")

    df_final = df[["Kecamatan", "Nilai", "Cluster_Label", "Anomali"]]
    table_data = df_final.to_dict("records")
    table_columns = [{"name": i, "id": i} for i in df_final.columns]

    highest = df.iloc[0]
    lowest = df.iloc[-1]
    summary = df.groupby("Cluster_Label")["Nilai"].agg(["mean", "max", "min"]).reset_index()
    insight_div = html.Div([
        html.H4("💡 Insight Otomatis:"),
        html.P(f"🏆 Kecamatan nilai tertinggi: {highest['Kecamatan']} ({highest['Nilai']}) — {highest['Cluster_Label']}"),
        html.P(f"📉 Kecamatan nilai terendah: {lowest['Kecamatan']} ({lowest['Nilai']}) — {lowest['Cluster_Label']}"),
        dash_table.DataTable(
            data=summary.round(2).to_dict("records"),
            columns=[{"name": i, "id": i} for i in summary.columns],
            style_cell={'textAlign': 'center', 'padding': '5px'},
            style_header={'backgroundColor': '#f5f5f5', 'fontWeight': 'bold'}
        )
    ])

    dq = generate_data_quality_report(df)
    eda_store = {
        "bar": px.bar(df, x="Kecamatan", y="Nilai", color="Cluster_Label", title="Perbandingan Nilai per Kecamatan").to_json(),
        "hist": px.histogram(df, x="Nilai", nbins=20, title="Distribusi Nilai").to_json(),
        "scatter": px.scatter(df.reset_index(), x="index", y="Nilai", color="Cluster_Label",
                              hover_data=["Kecamatan"], title="Scatter Nilai (index vs nilai)").to_json(),
        "summary": summary.round(2).to_dict("records"),
        "dq": pd.DataFrame(list(dq["missing_per_col"].items()), columns=["Kolom", "Missing"]).to_dict("records")
    }

    timestamp = f"Terakhir diperbarui: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return timestamp, kpis, fig_map, table_data, table_columns, insight_div, eda_store

# ----------------- Callback kedua: update isi tab -----------------
@app.callback(
    Output("eda-content", "children"),
    [Input("eda-tabs", "value"),
     Input("eda-store", "data")]
)
def update_eda_tab(selected_tab, eda_store):
    if not eda_store:
        return html.Div("Tidak ada data EDA.")
    import plotly.io as pio
    try:
        if selected_tab == "tab-bar":
            return dcc.Graph(figure=pio.from_json(eda_store["bar"]))
        elif selected_tab == "tab-hist":
            return dcc.Graph(figure=pio.from_json(eda_store["hist"]))
        elif selected_tab == "tab-scatter":
            return dcc.Graph(figure=pio.from_json(eda_store["scatter"]))
        elif selected_tab == "tab-summary":
            return html.Div([
                html.H4("Ringkasan Statistik per Klaster"),
                dash_table.DataTable(
                    data=eda_store["summary"],
                    columns=[{"name": i, "id": i} for i in pd.DataFrame(eda_store["summary"]).columns],
                    style_table={'overflowX': 'auto'}
                )
            ])
        elif selected_tab == "tab-quality":
            return html.Div([
                html.H4("Laporan Kualitas Data"),
                dash_table.DataTable(
                    data=eda_store["dq"],
                    columns=[{"name": i, "id": i} for i in pd.DataFrame(eda_store["dq"]).columns],
                    style_table={'overflowX': 'auto'}
                )
            ])
    except Exception as e:
        logging.error(f"Error saat memuat EDA: {e}")
        return html.Div("Terjadi kesalahan saat memuat EDA.")
    return html.Div("Pilih tab untuk menampilkan analisis.")

# ----------------- Run -----------------
if __name__ == "__main__":
    app.run(debug=True)

server = app.server

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
