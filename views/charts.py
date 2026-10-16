"""Fungsi grafik (Plotly) yang dipakai di Single dan Batch Mode."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core import config

_FONT = dict(family="Inter, Segoe UI, sans-serif", size=13)


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        font=_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def probability_chart(probabilities: dict[str, float]) -> go.Figure:
    labels = config.SENTIMENT_ORDER
    values = [probabilities.get(l, 0.0) for l in labels]
    fig = go.Figure(go.Bar(
        x=values,
        y=[l.capitalize() for l in labels],
        orientation="h",
        marker_color=[config.SENTIMENT_COLORS[l] for l in labels],
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.15], visible=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _base_layout(fig, height=150)


def sentiment_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["sentimen"].value_counts().reindex(config.SENTIMENT_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)
    fig = go.Figure(go.Bar(
        x=[l.capitalize() for l in counts.index],
        y=counts.values,
        marker_color=[config.SENTIMENT_COLORS[l] for l in counts.index],
        text=[f"{c} ({c / total:.0%})" for c in counts.values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}: %{y} review<extra></extra>",
    ))
    fig.update_yaxes(title="Jumlah review", rangemode="tozero", gridcolor="rgba(128,128,128,0.2)")
    return _base_layout(fig, height=320)


def _topic_label(row) -> str:
    return f"T{row['topik_id']} · {row['topik']}"


def topic_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = (
        df.assign(label=df.apply(_topic_label, axis=1))
        .groupby("label").size().sort_values(ascending=True)
    )
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        marker_color="#3E63DD",
        text=counts.values,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x} review<extra></extra>",
    ))
    fig.update_xaxes(title="Jumlah review", gridcolor="rgba(128,128,128,0.2)")
    return _base_layout(fig, height=max(260, 40 * len(counts) + 60))


def sentiment_topic_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    labelled = df.assign(label=df.apply(_topic_label, axis=1))
    order = labelled.sort_values("topik_id")["label"].unique()
    table = pd.crosstab(labelled["label"], labelled["sentimen"])
    table = table.reindex(index=order, columns=config.SENTIMENT_ORDER, fill_value=0)
    table.columns = [c.capitalize() for c in table.columns]
    return table


def sentiment_topic_heatmap(df: pd.DataFrame) -> go.Figure:
    table = sentiment_topic_crosstab(df)
    fig = px.imshow(
        table,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Sentimen", y="Topik", color="Jumlah"),
    )
    fig.update_xaxes(side="top")
    fig = _base_layout(fig, height=max(280, 44 * len(table) + 80))
    fig.update_layout(coloraxis_showscale=False)
    return fig


def sentiment_share_per_topic_chart(df: pd.DataFrame) -> go.Figure:
    table = sentiment_topic_crosstab(df)
    share = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    fig = go.Figure()
    for label in config.SENTIMENT_ORDER:
        col = label.capitalize()
        fig.add_trace(go.Bar(
            name=col,
            y=share.index,
            x=share[col],
            orientation="h",
            marker_color=config.SENTIMENT_COLORS[label],
            text=[f"{v:.0%}" if v >= 0.08 else "" for v in share[col]],
            textposition="inside",
            customdata=table[col],
            hovertemplate=f"{col}: %{{x:.0%}} (%{{customdata}} review)<extra></extra>",
        ))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1], gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(autorange="reversed")
    fig = _base_layout(fig, height=max(280, 44 * len(share) + 80))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
    return fig
