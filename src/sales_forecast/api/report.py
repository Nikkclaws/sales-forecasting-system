"""Multi-page PDF report generator: forecast, CV table, drift summary."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

if TYPE_CHECKING:
    from .service import ForecastService


def _forecast_chart(points: list[dict]) -> bytes:
    """Render a forecast line+CI chart as PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=130)
    dates = [p["date"] for p in points]
    yhat = [p["yhat"] for p in points]
    lo = [p.get("yhat_lower") for p in points]
    hi = [p.get("yhat_upper") for p in points]
    ax.plot(dates, yhat, marker="o", color="#1f77b4", label="Forecast")
    if all(v is not None for v in lo) and all(v is not None for v in hi):
        ax.fill_between(dates, lo, hi, alpha=0.18, color="#1f77b4", label="Prediction interval")
    ax.set_title("Forecast (next 8 weeks)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Predicted weekly sales")
    plt.xticks(rotation=35, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_pdf_report(service: ForecastService, state: str, horizon: int = 8) -> bytes:
    """Render a multi-page PDF: cover, forecast chart, CV metrics table, drift."""
    forecast = service.predict(state=state, horizon=horizon)
    metrics = service.metadata.get("states", {}).get(state, {}).get("aggregate_metrics", {})
    drift = forecast.get("drift") or {}

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph(f"<b>Sales Forecast Report — {state}</b>", styles["Title"]))
    flow.append(Paragraph(f"Registry version: {forecast['registry_version']}", styles["Normal"]))
    flow.append(Paragraph(f"Horizon: {forecast['horizon_weeks']} weeks", styles["Normal"]))
    flow.append(Paragraph(f"CI method: {forecast.get('ci_method', 'model_native')}", styles["Normal"]))
    flow.append(Paragraph(f"Selected models: {', '.join(forecast['selected_models'])}", styles["Normal"]))
    weights = ", ".join(f"{k}={v:.3f}" for k, v in forecast["ensemble_weights"].items())
    flow.append(Paragraph(f"Ensemble weights: {weights}", styles["Normal"]))
    flow.append(Spacer(1, 0.2 * inch))

    # Forecast chart.
    img_bytes = _forecast_chart(forecast["forecast"])
    flow.append(Image(io.BytesIO(img_bytes), width=6.6 * inch, height=3.0 * inch))
    flow.append(Spacer(1, 0.15 * inch))

    # Forecast table.
    rows = [["Date", "Forecast", "Lower", "Upper"]]
    for p in forecast["forecast"]:
        lo = "—" if p.get("yhat_lower") is None else f"{p['yhat_lower']:.0f}"
        hi = "—" if p.get("yhat_upper") is None else f"{p['yhat_upper']:.0f}"
        rows.append([p["date"], f"{p['yhat']:.0f}", lo, hi])
    tbl = Table(rows, colWidths=[1.4 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    flow.append(tbl)

    flow.append(PageBreak())

    # CV metrics page.
    flow.append(Paragraph("<b>Cross-validation metrics (per model)</b>", styles["Heading2"]))
    if metrics:
        keys = sorted({k for m in metrics.values() for k in m})
        head = ["model", *keys]
        rows = [head]
        for model, vals in metrics.items():
            rows.append([model, *[f"{vals.get(k, float('nan')):.3f}" for k in keys]])
        tbl = Table(rows)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#444")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        flow.append(tbl)
    else:
        flow.append(Paragraph("No CV metrics recorded.", styles["Normal"]))

    flow.append(Spacer(1, 0.3 * inch))
    flow.append(Paragraph("<b>Drift indicators</b>", styles["Heading2"]))
    if drift:
        rows = [["Metric", "Value"]]
        for k, v in drift.items():
            rows.append([str(k), f"{v}"])
        tbl = Table(rows, colWidths=[2.5 * inch, 4.0 * inch])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#bb5500")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        flow.append(tbl)
    else:
        flow.append(Paragraph("No drift indicators available.", styles["Normal"]))

    doc.build(flow)
    return buf.getvalue()
