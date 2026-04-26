"""
report/generate.py
------------------
Auto-generates a downloadable PDF analysis report for a completed prediction.
Uses fpdf2 for layout; embeds the SHAP waterfall chart as an image.
"""

import io
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

log = logging.getLogger(__name__)


def _sanitize_text(text: str) -> str:
    """Remove Unicode characters not supported by Helvetica font."""
    if not isinstance(text, str):
        return str(text)
    # Replace Unicode dashes and other problematic characters
    text = text.replace("—", "-")   # em dash
    text = text.replace("–", "-")   # en dash
    text = text.replace("―", "-")   # horizontal bar
    text = text.replace("…", "...")  # ellipsis
    return text


def generate_report(result: Dict[str, Any], filename: str = "malsentinel_report.pdf") -> bytes:
    """
    Build a PDF report from a prediction result dict.

    Parameters
    ----------
    result   : dict returned by predict.run_prediction()
    filename : output filename (used only for the FPDF title)

    Returns
    -------
    bytes  - the raw PDF content ready for st.download_button
    """
    try:
        from fpdf import FPDF, XPos, YPos
    except ImportError:
        log.error("fpdf2 not installed. Run: pip install fpdf2")
        raise

    pe_info     = result.get("pe_info", {})
    lgbm_prob   = result.get("lgbm_prob", 0.0)
    lgbm_verdict= result.get("lgbm_verdict", "UNKNOWN")
    bayes_prob  = result.get("bayes_prob", -1.0)
    metrics     = result.get("lgbm_metrics", {})
    shap_fig    = result.get("shap_figure")
    bayes_det   = result.get("bayes_details", {})

    # ── Save SHAP figure to a temp PNG ────────────────────────────────────────
    shap_img_path = None
    if shap_fig is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            shap_fig.savefig(f.name, dpi=120, bbox_inches="tight")
            shap_img_path = f.name

    # ── Build PDF ─────────────────────────────────────────────────────────────
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ─ Header ─────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, _sanitize_text("MalSentinel - Malware Analysis Report"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, _sanitize_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ─ Verdict banner ─────────────────────────────────────────────────────────
    if lgbm_verdict == "MALICIOUS":
        pdf.set_fill_color(220, 38, 38)
        banner_text = _sanitize_text(f"[!] VERDICT: MALICIOUS   ({lgbm_prob*100:.1f}% confidence)")
    else:
        pdf.set_fill_color(22, 163, 74)
        banner_text = _sanitize_text(f"[OK] VERDICT: BENIGN   ({(1-lgbm_prob)*100:.1f}% confidence)")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, banner_text, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(4)

    # ─ File info ──────────────────────────────────────────────────────────────
    _section_header(pdf, _sanitize_text("File Information"))
    rows = [
        ("File Size",      _sanitize_text(f"{pe_info.get('file_size_bytes', 0):,} bytes")),
        ("Sections",       _sanitize_text(str(pe_info.get("num_sections", "N/A")))),
        ("Machine Type",   _sanitize_text(str(pe_info.get("machine_type", "N/A")))),
        ("Entry Point",    _sanitize_text(str(pe_info.get("entry_point", "N/A")))),
        ("Has TLS",        _sanitize_text("Yes" if pe_info.get("has_tls") else "No")),
        ("Has Debug Info", _sanitize_text("Yes" if pe_info.get("has_debug") else "No")),
        ("Has Resources",  _sanitize_text("Yes" if pe_info.get("has_resources") else "No")),
    ]
    _table(pdf, rows)
    pdf.ln(4)

    # ─ ML Results ─────────────────────────────────────────────────────────────
    _section_header(pdf, _sanitize_text("Machine Learning Results"))
    ml_rows = [
        ("LightGBM Probability",  _sanitize_text(f"{lgbm_prob:.4f}")),
        ("LightGBM Verdict",      _sanitize_text(lgbm_verdict)),
    ]
    if bayes_prob >= 0:
        ml_rows.append(("Bayesian (NB) Probability", _sanitize_text(f"{bayes_prob:.4f}")))
    if metrics:
        ml_rows += [
            ("Model Accuracy",  _sanitize_text(f"{metrics.get('accuracy', 0):.4f}")),
            ("Model F1 Score",  _sanitize_text(f"{metrics.get('f1', 0):.4f}")),
            ("Model AUC-ROC",   _sanitize_text(f"{metrics.get('auc', 0):.4f}")),
        ]
    _table(pdf, ml_rows)
    pdf.ln(4)

    # ─ Suspicious APIs ────────────────────────────────────────────────────────
    suspicious = pe_info.get("suspicious_apis", [])
    _section_header(pdf, _sanitize_text("Suspicious API Imports"))
    if suspicious:
        pdf.set_font("Helvetica", "", 10)
        for api in suspicious:
            pdf.cell(0, 6, _sanitize_text(f"  - {api}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 6, _sanitize_text("  None detected."), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ─ Bayesian details ───────────────────────────────────────────────────────
    if bayes_det.get("feature_values"):
        _section_header(pdf, _sanitize_text("Bayesian Network Feature Values"))
        rows = [(_sanitize_text(k), _sanitize_text(f"{v:.4f}")) for k, v in bayes_det["feature_values"].items()]
        _table(pdf, rows)
        pdf.ln(4)

    # ─ SHAP chart ─────────────────────────────────────────────────────────────
    if shap_img_path and Path(shap_img_path).exists():
        pdf.add_page()
        _section_header(pdf, _sanitize_text("SHAP Waterfall Prediction Explanation"))
        pdf.image(shap_img_path, x=10, w=190)
        pdf.ln(4)

    # ─ Footer ─────────────────────────────────────────────────────────────────
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, _sanitize_text("MalSentinel - CS 2005 AI Project | Static analysis only. Never executes files."),
             align="C")

    pdf_bytes = pdf.output()

    # Cleanup temp file
    if shap_img_path:
        try:
            Path(shap_img_path).unlink()
        except Exception:
            pass

    return bytes(pdf_bytes)


def _section_header(pdf, title: str):
    from fpdf import XPos, YPos
    title = _sanitize_text(title)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(0.5)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
    pdf.ln(2)
    pdf.set_text_color(30, 30, 30)


def _table(pdf, rows):
    from fpdf import XPos, YPos
    pdf.set_font("Helvetica", "", 10)
    col_w = [75, 110]
    for label, value in rows:
        label = _sanitize_text(str(label))
        value = _sanitize_text(str(value))
        pdf.set_fill_color(245, 247, 250)
        pdf.cell(col_w[0], 7, f"  {label}", border=1, fill=True)
        pdf.cell(col_w[1], 7, f"  {value}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
