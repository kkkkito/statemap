import matplotlib
matplotlib.use('Agg')

import pandas as pd
chronos = pd.read_csv("./data/CRISPR_gene_effect.csv",index_col=0)
chronos.columns = [c.split()[0].upper() + '_Chronos' for c in chronos.columns]
expression = pd.read_csv("./data/CCLE_expression.csv", index_col=0)
expression.columns = [c.split()[0].upper()+'_Expression' for c in expression.columns]

sample_info = pd.read_csv("./data/sample_info.csv", index_col=0)
id_to_name = {i:n for i, n in zip(sample_info.index, sample_info["CCLE_Name"])}
name_to_id = {n:i for i, n in zip(sample_info.index, sample_info["CCLE_Name"])}
name_to_lineage = {n:l for n, l in zip(sample_info["CCLE_Name"], sample_info["lineage"])}

prism = pd.read_csv("./data/PRISM.csv", index_col=0)
prism.index = [id_to_name.get(i,'') for i in prism.index]

mutation = pd.read_csv('./data/25Q2_damaging_mutations.csv',index_col=0)
mutation.columns = [c.split()[0].upper()+'_Mutation' for c in mutation.columns]
mutation.index = [id_to_name.get(i,"") for i in mutation.index]

ch_ex = chronos.join(expression,how='inner')
ch_ex.index = [id_to_name[i] for i in ch_ex.index]
ch_ex = ch_ex.dropna(axis=1)
ch_ex = ch_ex[~ch_ex.index.isna()]


#Imports and Utils

from flask import Flask, request, render_template
import matplotlib.pyplot as plt
from io import BytesIO
import trimap
import base64
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import collections
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from sklearn.cluster import KMeans
import plotly.express as px
import seaborn as sns
from flask import session
import time
import os
import random
import secrets
from flask import Flask, render_template, request, redirect, url_for


def reorder_clusters_by_x(embedding, cls):
    unique_labels = np.unique(cls)
    centroids = []
    for label in unique_labels:
        mask = cls == label
        center_x = embedding[mask, 0].mean()
        centroids.append((label, center_x))
    sorted_labels = [l for l, _ in sorted(centroids, key=lambda x: x[1])]
    label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    new_cls = np.array([label_map[label] for label in cls])
    return new_cls

import plotly.express as px
import seaborn as sns

def rgb_to_hex(rgb_str):
    if rgb_str.startswith("rgb"):
        nums = [int(s) for s in rgb_str[4:-1].split(",")]
        return '#{:02x}{:02x}{:02x}'.format(*nums)
    return rgb_str

chr_genes_default = """AMOTL2
KIRREL1
NF2
PDCD10
RNF146
TAOK1
PTPN14
MAP4K4
FRYL
LATS2
NRP1"""
exp_genes_default = """YAP1
WWTR1
CCN1
CCN2"""

import os
import json

app = Flask(__name__)
app.secret_key = "secretsecret"

@app.route("/", methods=["GET"])
def index():
    uid = request.args.get('uid')

    if not uid:
        new_uid = secrets.token_urlsafe(6)
        os.makedirs(f'tmp/{new_uid}', exist_ok=True)
        return redirect(url_for('index', uid=new_uid))

    return render_template("index.html",
        step1_done=False,
        step2_done=False,
        chr_genes=chr_genes_default,
        exp_genes=exp_genes_default,
        n_clusters=None,
        cluster_labels=None,
        current_feature=None,
        uid = uid
    )

@app.route("/reduce", methods=["POST"])
def reduce():
    uid = request.args.get('uid')
    chr_genes = request.form.get("chr_genes", "").strip()
    exp_genes = request.form.get("exp_genes", "").strip()
    chr_gene_list = [g.strip().upper() for g in chr_genes.splitlines() if g.strip()]
    exp_gene_list = [g.strip().upper() for g in exp_genes.splitlines() if g.strip()]
    columns_of_interest = []
    missing_genes = []

    for g in chr_gene_list:
        col = g + "_Chronos"
        if col in ch_ex.columns:
            columns_of_interest.append(col)
        else:
            missing_genes.append(g + " (Chronos)")
    for g in exp_gene_list:
        col = g + "_Expression"
        if col in ch_ex.columns:
            columns_of_interest.append(col)
        else:
            missing_genes.append(g + " (Expression)")

    # 選択肢がゼロならエラーで返す
    if not columns_of_interest:
        return render_template(
            "index.html",
            plot_div=None,
            step1_done=False,
            step2_done=False,
            chr_genes=chr_genes,
            exp_genes=exp_genes,
            n_clusters=None,
            cluster_labels=None,
            current_feature=None,
            uid=uid,
            error_message="The specified genes were not found in the dataset.<br>Please enter valid gene symbols following DepMap"
        )

    # それ以外は通常処理
    data = ch_ex[columns_of_interest]

    scaled_data = StandardScaler().fit_transform(data)
    trimap_reducer = trimap.TRIMAP(n_dims=2)

    embedding = trimap_reducer.fit_transform(scaled_data)
    embedding = StandardScaler().fit_transform(embedding)

    npy_path = f"./tmp/{uid}/embedding.npy"
    np.save(npy_path, embedding)

    # Get cell line names for hover
    cell_line_names = ch_ex.index.astype(str).tolist()  # or adjust if your cell line names are elsewhere

    # Create Plotly figure
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hover_name=cell_line_names,  # This will show cell line name on hover
        title="TriMap Embedding",
        labels={"x": "Component 1", "y": "Component 2"},
        width=500,
        height=500
    )
    fig.update_traces(marker=dict(size=6, color='gray'))  # adjust size/color

    # Convert Plotly figure to HTML div string
    plot_div = fig.to_html(full_html=False)

    # メッセージ生成: missing_genesがあれば
    error_message = None
    if missing_genes:
        error_message = (
            "The following genes were not found in the dataset and were skipped:<br>"
            + ", ".join(missing_genes) +
            "<br>Please use official DepMap gene symbols."
            )

    return render_template(
        "index.html",
        plot_div=plot_div,
        step1_done=True,
        step2_done=False,
        chr_genes=chr_genes,
        exp_genes=exp_genes,
        n_clusters=5,
        cluster_labels=None,
        current_feature=None,
        current_gene="YAP1",
        current_type="Expression",
        error_message=error_message,
        uid=uid
    )

@app.route("/cluster", methods=["POST"])
def cluster():
    uid = request.args.get('uid')
    embedding = np.load(f"./tmp/{uid}/embedding.npy")

    n_clusters_str = request.form.get("n_clusters", "5").strip()
    n_clusters = int(n_clusters_str)
    chr_genes = request.form.get("chr_genes", "").strip()
    exp_genes = request.form.get("exp_genes", "").strip()

    spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=70, gamma = 5)
    #spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=70, n_neighbors=int(250/n_clusters))
    cls = spectral.fit_predict(embedding)  # クラスタラベルを取得
    cls = reorder_clusters_by_x(embedding, cls)
    cls_path = f"./tmp/{uid}/cls.npy"
    np.save(cls_path, cls)

    # 昇順クラスタID・ラベル
    counts = collections.Counter(cls)
    sorted_labels = sorted(counts.keys())
    cluster_labels = [f"{label}: {counts[label]}" for label in sorted_labels]
    cell_line_names = ch_ex.index.astype(str).tolist()

    # 色パレット（Set1/Set3、hex変換）
    color_palette = px.colors.qualitative.Set1
    if len(sorted_labels) > len(color_palette):
        color_palette = px.colors.qualitative.Set3
    hex_colors = [rgb_to_hex(c) for c in color_palette]
    palette_list = hex_colors[:len(sorted_labels)]  # ここを忘れず定義

    df_bar = pd.DataFrame({
        "Name": ch_ex.index,
        "Cluster": cls,
    })

    df_bar["Lineage"] = df_bar["Name"].map(name_to_lineage)
    ct = pd.crosstab(df_bar["Lineage"], df_bar["Cluster"])
    ct_ratio = ct.div(ct.sum(axis=1), axis=0)

    # Ensure columns in ascending cluster order (to match palette)
    ct_ratio = ct_ratio.reindex(sorted_labels, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ct_ratio.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=palette_list  # key line: colors in cluster order
    )
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Lineage")
    ax.set_title("Cluster composition per lineage (proportion)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc='upper left') # axから呼ぶ
    buf = BytesIO()
    fig.tight_layout() # figから呼ぶ
    fig.savefig(buf, format="png") 
    buf.seek(0)
    image_data2 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    
    fig = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=[str(i) for i in cls],
        color_discrete_sequence=palette_list,
        category_orders={"color": [str(x) for x in sorted_labels]},
        hover_name=cell_line_names,
        title=f"Clustering (k={n_clusters})",
        labels={"color": "Cluster", "x": "Component 1", "y": "Component 2"},
        width=500,
        height=500
    )
    fig.update_traces(marker=dict(size=6),
                      type='scattergl')
    
    plot_div = fig.to_html(full_html=False)
    with open(f"./tmp/{uid}/plot.html", "w") as f:
        f.write(plot_div)
    with open(f"./tmp/{uid}/param.json", "w") as f:
        json.dump({
            "n_clusters": n_clusters,
            "chr_genes": chr_genes,
            "exp_genes": exp_genes
        }, f)

    return render_template("index.html",
        plot_div=plot_div,
        step1_done=True,
        step2_done=True,
        chr_genes=chr_genes,
        exp_genes=exp_genes,
        n_clusters=n_clusters,
        cluster_labels=cluster_labels,
        current_feature=None,
        current_gene=exp_genes.split('\n')[0],
        current_type="Expression",
        error_message=None,
        image_data2=image_data2,
        uid=uid                   
    )


@app.route("/feature", methods=["POST"])
def feature():
    uid = request.args.get('uid')
    embedding = np.load(f"./tmp/{uid}/embedding.npy")
    plot = f"./tmp/{uid}/plot.html"

    feature_gene = request.form.get("feature_gene", "").strip().upper()
    feature_type = request.form.get("feature_type", "Expression")
    n_clusters = request.form.get("n_clusters", "5")

    cls_global = np.load(f"./tmp/{uid}/cls.npy")
    chr_genes = request.form.get("chr_genes", "").strip()
    exp_genes = request.form.get("exp_genes", "").strip()
    error_message = None
    data = ch_ex.join(mutation, how='left')
    data = data.join(prism,how='left')

    #feature_name = feature_gene + "_Expression" if feature_type == "Expression" else feature_gene + "_Chronos"
    feature_name = f"{feature_gene}_{feature_type}"

    if feature_name not in data.columns:
        counts = collections.Counter(cls_global) if cls_global is not None else {}
        cluster_labels = [f"{label}: {count}" for label, count in sorted(counts.items())]
        n_clusters = len(cluster_labels)
        error_message = f"The \"{feature_name}\" was not found."
        return render_template("index.html",
            step1_done=True,
            step2_done=True,
            chr_genes=chr_genes,
            exp_genes=exp_genes,
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
            current_feature=None,
            current_gene=feature_gene,
            current_type=feature_type,
            error_message=error_message,
            uid=uid
        )

    feature_values = data[feature_name].values
    cell_line_names = data.index.astype(str).tolist()

    # クラスタIDを必ず昇順で使う
    cls = np.load(f"./tmp/{uid}/cls.npy")
    unique_cls = np.unique(cls)
    sorted_cls = sorted(unique_cls)

    # ----- カラーパレット(Plotly, matplotlib両対応) -----
    color_palette = px.colors.qualitative.Set1
    if len(sorted_cls) > len(color_palette):
        color_palette = px.colors.qualitative.Set3
    hex_colors = [rgb_to_hex(c) for c in color_palette]
    # 必ずクラスタID昇順で同じ順番を使う
    palette_list = hex_colors[:len(sorted_cls)]
    palette = {cid: palette_list[i] for i, cid in enumerate(sorted_cls)}
    vmin = np.percentile(feature_values, 5)
    vmax = np.percentile(feature_values, 95)

    fig_cluster = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=[str(i) for i in cls_global],
        color_discrete_sequence=palette_list,
        category_orders={"color": [str(x) for x in sorted_cls]},
        hover_name=cell_line_names,
        title="Cluster colored scatter",
        labels={"color": "Cluster", "x": "Component 1", "y": "Component 2"},
        width=380,
        height=380
    )
    fig_cluster.update_traces(marker=dict(size=6),
                              type='scattergl')
    plot_div_cluster = fig_cluster.to_html(full_html=False)
    mask = ~np.isnan(feature_values)
    fig_feat = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=np.where(mask, feature_values, np.nan),
        range_color=[vmin, vmax],
        hover_name=cell_line_names,
        color_continuous_scale="RdBu",
        labels={"color": feature_name, "x": "Component 1", "y": "Component 2"},
        title=f"Feature: {feature_name}",
        width=380,
        height=380
    )
    # NaN 部分の marker を透明化
    fig_feat.update_traces(
        marker=dict(
            opacity=np.where(mask, 1.0, 0.0)  # 値が NaN の点は透明
        ),
        selector=dict(mode="markers")
    )

    fig_feat.update_traces(marker=dict(size=6),
                           type='scattergl')
    fig_feat.update_coloraxes(colorbar_title='')
    plot_div_feat = fig_feat.to_html(full_html=False)


    # ----------- violin plot (matplotlib/seaborn) -----------
    fig2, ax2 = plt.subplots(figsize=(5, 3))

    if feature_type == 'Expression': y_label = 'log2TPM'
    if feature_type == 'Chronos' : y_label = 'Chronos score'
    if feature_type == 'Mutation' : y_label = 'Damaging mutation'
    if feature_type == 'PRISM' : y_label = 'PRISM score'

    df_vln = pd.DataFrame({y_label: feature_values, 'cluster': cls_global.astype(int)})
    # seaborn: palette（昇順ID順）だけ渡す（dict可）

    sns.violinplot(
        x='cluster', y=y_label, hue='cluster',
        data=df_vln, ax=ax2,
        palette=palette, legend=False, cut=0, order=sorted_cls, hue_order=sorted_cls
    )
    ax2.set_title(f"{feature_name} distribution by cluster")
    buf2 = BytesIO()
    fig2.tight_layout() # (推奨) ラベルなどのレイアウトを自動調整
    fig2.savefig(buf2, format="png") # fig2オブジェクトから保存
    buf2.seek(0)
    plt.close(fig2)
    image_data2 = base64.b64encode(buf2.read()).decode('utf-8')

    # クラスタラベルも昇順
    counts = collections.Counter(cls_global)
    cluster_labels = [f"{label}: {counts[label]}" for label in sorted_cls]
    n_clusters = len(cluster_labels)

    return render_template("index.html",
        plot_div_cluster=plot_div_cluster,
        plot_div_feat=plot_div_feat,
        image_data2=image_data2,
        step1_done=True,
        step2_done=True,
        chr_genes=chr_genes,
        exp_genes=exp_genes,
        n_clusters=n_clusters,
        cluster_labels=cluster_labels,
        current_feature=feature_name,
        current_gene=feature_gene,
        current_type=feature_type,
        error_message=None,
        uid=uid
    )
from flask import send_file

@app.route("/download_csv", methods=["POST"])
def download_csv():
    uid = request.args.get('uid')
    embedding_global = np.load(f"./tmp/{uid}/embedding.npy")
    cls_global = np.load(f"./tmp/{uid}/cls.npy")

    n_clusters = request.form.get("n_clusters", "5")
    chr_genes = request.form.get("chr_genes", "").strip().split('\n')
    exp_genes = request.form.get("exp_genes", "").strip().split('\n')

    chr_genes = [g.strip().upper() for g in chr_genes]
    exp_genes = [g.strip().upper() for g in exp_genes]

    cell_line_names = ch_ex.index.astype(str).tolist()
    df = pd.DataFrame({
        "DepMapID":[name_to_id.get(name,'') for name in cell_line_names],
        "CellLine": cell_line_names,
        "x": embedding_global[:, 0],
        "y": embedding_global[:, 1],
        "Cluster": cls_global
    })

    # Round x, y to 3 decimal places
    df["x"] = df["x"].round(3)
    df["y"] = df["y"].round(3)

    header_lines = [
        f"# Expression genes: {' '.join(exp_genes)}",
        f"# Chronos genes: {' '.join(chr_genes)}",
        f"# Number of clusters: {n_clusters if n_clusters else 'unknown'}"
    ]
    csv_data = '\n'.join(header_lines) + '\n\n'
    csv_data += df.to_csv(index=False)

    buf = BytesIO()
    buf.write(csv_data.encode('utf-8'))
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name="embedding_clusters.csv"
    )

@app.route("/download_feature", methods=["POST"])
def download_feature():
    uid = request.args.get('uid')
    embedding_global = np.load(f"./tmp/{uid}/embedding.npy")
    cls_global = np.load(f"./tmp/{uid}/cls.npy")
    last_chr_genes = request.form.get("chr_genes", "").strip().split('\n')
    last_exp_genes = request.form.get("exp_genes", "").strip().split('\n')
    last_chr_genes = [g.strip().upper() for g in last_chr_genes]
    last_exp_genes = [g.strip().upper() for g in last_exp_genes]
    last_n_clusters = request.form.get("n_clusters", "5")

    data = ch_ex.join(prism, how='left') # データフレーム: サンプル × 遺伝子
    X = (data - data.mean(axis=0)) / data.std(axis=0)  # Z-score標準化

    cluster_labels = np.unique(cls_global)
    results = []

    for cluster in cluster_labels:
        in_cluster = (cls_global == cluster)
        out_cluster = ~in_cluster

        mean_in = X.loc[in_cluster].mean(axis=0)
        mean_out = X.loc[out_cluster].mean(axis=0)
        effect_size = mean_in - mean_out

        for gene, eff in effect_size.items():
            gn = gene.split("_")[0]
            gf = gene.split("_")[1]
            if eff > 0.5:
                results.append({
                    "Cluster": f"Cluster{cluster}",
                    "Type": f"High_{gf}",
                    "gene": gn,
                    "effect size": round(eff, 3)
                })
            elif eff < -0.5:
                results.append({
                    "Cluster": f"Cluster{cluster}",
                    "Type": f"Low_{gf}",
                    "gene": gn,
                    "effect size": round(eff, 3)
                })

    # DataFrame化してソート
    df = pd.DataFrame(results)
    df = df.sort_values(["Type", "effect size"], ascending=[True, False])

    header_lines = [
        "# This file lists cluster-specific high/low expression features.",
        "# For each cluster, the z-score for each gene is computed, and genes with a mean difference >0.5 or <-0.5 (effect size) between in-cluster and out-cluster are shown.",
        f"# Expression genes: {' '.join(last_exp_genes)}",
        f"# Chronos genes: {' '.join(last_chr_genes)}",
        f"# Number of clusters: {last_n_clusters if last_n_clusters else 'unknown'}"
    ]

    csv_data = '\n'.join(header_lines) + '\n\n'
    # CSV出力
    csv_data += df.to_csv(index=False)

    buf = BytesIO()
    buf.write(csv_data.encode('utf-8'))
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name="cluster_specific_features.csv"
    )
from flask import send_file
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from io import BytesIO
from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import os
import math

@app.route("/download_summary", methods=["POST"])
def download_summary():
    # --- 1. リクエストからデータを取得 ---
    uid = request.args.get('uid')
    embedding_global = np.load(f"./tmp/{uid}/embedding.npy")
    cls_global = np.load(f"./tmp/{uid}/cls.npy")
    n_clusters = request.form.get("n_clusters", "5")
    last_chr_genes = [g.strip().upper() for g in request.form.get("chr_genes", "").strip().split('\n')]
    last_exp_genes = [g.strip().upper() for g in request.form.get("exp_genes", "").strip().split('\n')]

    # 一時PDFファイルを管理するための変数
    temp_pdf_path = None
    try:
        # --- 2. PDFキャンバスの準備 ---
        # delete=False で、後で手動で削除できるようにする
        temp_pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf_path = temp_pdf_file.name
        temp_pdf_file.close() # ファイルハンドルはすぐに閉じる

        c = canvas.Canvas(temp_pdf_path, pagesize=A4)
        width, height = A4

        # --- 3. PDFの1ページ目（概要テキスト）を作成 ---
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, height - 60, "TriMap Clustering Summary")
        c.setFont("Helvetica", 12)
        y = height - 100
        c.drawString(50, y, f"Number of clusters: {n_clusters}")
        y -= 20
        
        # (wrap_text_list 関数は元のコードと同じなので省略)
        def wrap_text_list(title, genes, maxlen=80):
            lines = []
            line = title
            for g in genes:
                if len(line) + len(g) + 2 > maxlen:
                    lines.append(line)
                    line = "    " + g
                else:
                    if line.endswith(":"):
                        line += " " + g
                    else:
                        line += ", " + g
            if line: lines.append(line)
            return lines
            
        text_lines = []
        for line in wrap_text_list("Chronos genes:", last_chr_genes): text_lines.append(line)
        for line in wrap_text_list("Expression genes:", last_exp_genes): text_lines.append(line)

        for line in text_lines:
            c.drawString(50, y, line)
            y -= 16

        # --- 4. PDFの2ページ目（クラスタ散布図）を作成 ---
        temp_img_path = None
        try:
            # (オブジェクト指向APIを使用してプロットを生成)
            fig, ax = plt.subplots(figsize=(5, 5))
            for i in sorted(set(cls_global)):
                idx = (cls_global == i)
                ax.scatter(embedding_global[idx, 0], embedding_global[idx, 1], label=f"Cluster {i}", s=10)
            
            ax.legend(fontsize=8)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title("TriMap: colored by cluster")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
                temp_img_path = temp_img_file.name
            
            fig.savefig(temp_img_path, bbox_inches="tight", dpi=120)
            plt.close(fig) # 必ずfigオブジェクトを閉じる

            # PDFに描画
            img_max_h = y - 60
            img_w = 400
            img_h = min(400, img_max_h)
            c.setFont("Helvetica", 14)
            c.drawString(50, y-20, "Cluster-colored scatter map")
            c.drawImage(ImageReader(temp_img_path), 80, max(60, y - img_h - 40), width=img_w, height=img_h, preserveAspectRatio=True)
            c.showPage()
        finally:
            # 一時画像ファイルを確実に削除
            if temp_img_path and os.path.exists(temp_img_path):
                os.unlink(temp_img_path)

        # --- 5. PDFの3ページ目以降（各遺伝子の散布図）を作成 ---
        all_features = [g + "_Chronos" for g in last_chr_genes] + [g + "_Expression" for g in last_exp_genes]
        generated_images = [] # 生成した画像ファイルのパスを管理するリスト
        try:
            for feature_name in all_features:
                if feature_name not in ch_ex.columns:
                    continue
                
                # (オブジェクト指向APIを使用してプロットを生成)
                fig, ax = plt.subplots(figsize=(3, 3))
                values = ch_ex[feature_name].values
                vmin, vmax = np.percentile(values, 5), np.percentile(values, 95)
                
                sc = ax.scatter(embedding_global[:, 0], embedding_global[:, 1], c=values, cmap="RdBu", s=10, vmax=vmax, vmin=vmin)
                ax.set_xlabel("C1")
                ax.set_ylabel("C2")
                ax.set_title(feature_name)
                fig.colorbar(sc, ax=ax, label="", fraction=0.04, pad=0.04)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
                    img_path = temp_img_file.name
                
                fig.savefig(img_path, bbox_inches="tight", dpi=120)
                plt.close(fig) # 必ずfigオブジェクトを閉じる
                generated_images.append((feature_name, img_path))

            # 生成したすべての画像をPDFに描画
            plots_per_page = 9
            num_pages = math.ceil(len(generated_images) / plots_per_page)
            for page_num in range(num_pages):
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, height - 60, f"Scatter plots colored by feature (page {page_num + 1})")
                
                page_images = generated_images[page_num * plots_per_page:(page_num + 1) * plots_per_page]
                for k, (feature, path) in enumerate(page_images):
                    i, j = k % 3, k // 3
                    x = 45 + i * (150 + 25)
                    y = height - 100 - (j + 1) * 125 - j * 25
                    c.drawImage(ImageReader(path), x, y, width=150, height=125, preserveAspectRatio=True)
                c.showPage()
        finally:
            # このセクションで生成したすべての一時画像ファイルを確実に削除
            for _, path in generated_images:
                if os.path.exists(path):
                    os.unlink(path)

        # --- 6. PDFを保存して送信 ---
        c.save()
        return send_file(
            temp_pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="trimap_summary.pdf"
        )
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

if __name__ == "__main__":
    app.run()