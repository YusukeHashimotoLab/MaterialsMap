import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

# CSVファイルの読み込み
file_path = '../data/data_cleaned.csv'
df = pd.read_csv(file_path)

print(df.columns)

# 散布図を作成
fig = px.scatter(
    df,
    y="tSNE_1",
    x="tSNE_2",
    color="predicted_zT",
    labels={"tSNE_1": "tSNE 1", "tSNE_2": "tSNE 2"},
    color_continuous_scale="Viridis",
    hover_name="es_source_calc_id",
    hover_data=['formula_pretty', 'nsites', 'nelements', 'volume', 'density', 'density_atomic',
       'energy_per_atom', 'band_gap'],
    height=700,
    width=700
)

# HTMLとして保存
html_file_path = "../result/interactive_map.html"
pio.write_html(fig, html_file_path)

print(f"HTMLファイルが {html_file_path} に保存されました。")
