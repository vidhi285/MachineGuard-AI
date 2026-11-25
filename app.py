"""
app.py - MachineGuard AI: complete Streamlit frontend (v3).

Run with:  streamlit run app.py
Backend modules (unchanged): config.py, prediction.py, database.py.

v3 summary
* No technical connection / model-loading information is shown anywhere in the UI.
  Connections and models still load exactly as before; if either is unavailable the
  user sees a short plain-language message instead of technical details.
* Sidebar: logo + name with an icon-only collapse control beside them; icon rail when collapsed.
* Theme toggle: icon button at the top-right of every page header (session state).
* Dashboard, Machine Registry and Maintenance Planner redesigned (planner has full CRUD).
* Wording: model class labels are translated into plain language for display and downloads.
"""

from __future__ import annotations

import base64
import html
import io
import time as _time
from datetime import date, datetime, time, timezone
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
import prediction as pred
from database import DatabaseError, MachineGuardDB

try:
    from PIL import Image
    _PAGE_ICON = Image.open(cfg.LOGO_PATH) if cfg.LOGO_PATH.exists() else "⚙️"
except Exception:  # noqa: BLE001
    _PAGE_ICON = "⚙️"

st.set_page_config(page_title=cfg.APP_NAME, page_icon=_PAGE_ICON, layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================================
# THEMES
# ============================================================================
THEMES = {
    "dark": {
        "bg": "#171D1B", "sidebar": "#202725", "card": "#212A27", "card2": "#2A3531", "input": "#1B2321",
        "border": "#33403B", "text": "#F2F0E9", "muted": "#97A89D", "hover": "rgba(242,240,233,.07)",
        "ok": "#A6BBAB", "amber": "#E2B450", "red": "#E58373", "scheme": "dark",
        "colorway": ["#E58A32", "#78877C", "#D8D0BD", "#B94B3D", "#A9B8AE", "#D6A63A"],
    },
    "light": {
        "bg": "#F2F0E9", "sidebar": "#FAF8F3", "card": "#FFFFFF", "card2": "#F5F3EC", "input": "#FFFFFF",
        "border": "#DDD8CA", "text": "#202725", "muted": "#64736A", "hover": "rgba(32,39,37,.06)",
        "ok": "#4D6254", "amber": "#8A6410", "red": "#B94B3D", "scheme": "light",
        "colorway": ["#E58A32", "#78877C", "#202725", "#B94B3D", "#8F7A52", "#D6A63A"],
    },
}

CSS = Template("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
:root{--bg:${bg};--sidebar:${sidebar};--card:${card};--card2:${card2};--input:${input};--border:${border};--text:${text};
--muted:${muted};--hover:${hover};--ok:${ok};--amber:${amber};--red:${red};
--orange:#E58A32;--sage:#78877C;--alert:#B94B3D;--amber-solid:#D6A63A;color-scheme:${scheme};
--font:"Inter","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg)!important;
font-family:var(--font)!important;color:var(--text)}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stDecoration"],footer,[data-testid="stAppDeployButton"]{display:none!important}
.block-container{padding-top:4.3rem!important;padding-bottom:3rem!important;max-width:1400px}
h1,h2,h3,h4,h5,h6{font-family:var(--font)!important;color:var(--text)!important;letter-spacing:-.01em}
p,li,span,div{font-family:var(--font)}
.stMarkdown p,.stMarkdown li,[data-testid="stMarkdownContainer"] p{color:var(--text)}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted)!important}
hr{border-color:var(--border)!important;margin:1.2rem 0!important}
a{color:var(--orange)!important}
code{background:var(--card2)!important;color:var(--text)!important;border-radius:6px;padding:.1rem .35rem}

/* ---------------- sidebar ---------------- */
@media (min-width:769px){
[data-testid="stSidebar"]{width:${sbw}!important;min-width:${sbw}!important;max-width:${sbw}!important}
[data-testid="stSidebarHeader"],[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarResizeHandle"]{display:none!important}
[data-testid="stSidebarContent"]{overflow-x:hidden;padding-left:0!important;padding-right:0!important}
}
[data-testid="stSidebar"]{background:var(--sidebar)!important;border-right:1px solid var(--border)}
[data-testid="stSidebarUserContent"]{padding:1.1rem ${sbpad} 1rem ${sbpad}!important}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.25rem}
.mg-brand{display:flex;align-items:center;gap:.55rem;padding:.1rem .1rem;${brandjustify}}
.mg-brand .t b{display:block;font-size:.98rem;font-weight:700;color:var(--text);letter-spacing:-.01em;line-height:1.2;white-space:nowrap}
.mg-navlabel{display:block;font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:1.3rem 0 .9rem 0;padding:0 .5rem;font-weight:600;line-height:1}
[class*="st-key-collapsebtn"]{display:flex;justify-content:${collapsejustify}}
[class*="st-key-collapsebtn"] [data-testid="stButton"],[class*="st-key-collapsebtn"] [data-testid="stTooltipHoverTarget"]{width:auto!important}
[data-testid="stSidebar"] [class*="st-key-collapsebtn"] button{width:36px!important;height:36px;min-height:36px;padding:0;border-radius:10px;
background:transparent;border:1px solid var(--border);color:var(--muted);display:flex;justify-content:center;align-items:center}
[data-testid="stSidebar"] [class*="st-key-collapsebtn"] button:hover{color:var(--orange);border-color:var(--orange);background:transparent}
[data-testid="stSidebar"] [class*="st-key-nav"],[data-testid="stSidebar"] [class*="st-key-nav"] [data-testid="stTooltipHoverTarget"],
[data-testid="stSidebar"] [class*="st-key-nav"] [data-testid="stButton"]{width:100%!important;display:block}
[data-testid="stSidebar"] [class*="st-key-nav"] button{width:100%;height:auto;min-height:42px;display:flex;justify-content:${navjustify};align-items:center;
gap:.7rem;border:1px solid transparent;background:transparent;color:var(--muted);border-radius:10px;
padding:.4rem ${navpad};font-weight:500;box-shadow:none;text-align:left}
[data-testid="stSidebar"] [class*="st-key-nav"] button>div,[data-testid="stSidebar"] [class*="st-key-nav"] button>div>div,
[data-testid="stSidebar"] [class*="st-key-nav"] button [data-testid="stMarkdownContainer"]{
text-align:left!important;justify-content:${navjustify}!important;width:100%;min-width:0}
[data-testid="stSidebar"] [class*="st-key-nav"] button p{margin:0;font-size:.86rem;line-height:1.25;color:inherit!important;white-space:normal;text-align:left}
[data-testid="stSidebar"] [class*="st-key-nav"] button span{color:inherit!important}
[data-testid="stSidebar"] [class*="st-key-nav"] button:hover{background:var(--hover);color:var(--text);border-color:transparent}
[data-testid="stSidebar"] [class*="st-key-nav"] button:focus,[data-testid="stSidebar"] [class*="st-key-nav"] button:active{outline:none;box-shadow:none;color:var(--text)}
[data-testid="stSidebar"] [class*="st-key-navact_"] button{background:rgba(229,138,50,.13);color:var(--orange)!important;border-color:rgba(229,138,50,.32);
box-shadow:inset 3px 0 0 var(--orange)}
[data-testid="stSidebar"] [class*="st-key-navact_"] button:hover{background:rgba(229,138,50,.18);color:var(--orange)!important}

/* ---------------- theme toggle (top-right of every page) ---------------- */
[class*="st-key-themetoggle"]{display:flex;justify-content:flex-end}
[class*="st-key-themetoggle"] [data-testid="stButton"],[class*="st-key-themetoggle"] [data-testid="stTooltipHoverTarget"]{width:auto!important}
[class*="st-key-themetoggle"] button{width:44px!important;height:44px;min-height:44px;padding:0;border-radius:12px;background:var(--card);
border:1px solid var(--border);color:var(--orange);display:flex;justify-content:center;align-items:center}
[class*="st-key-themetoggle"] button:hover{border-color:var(--orange);background:var(--card2)}

/* ---------------- generic widgets ---------------- */
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *{color:var(--muted)!important;font-size:.8rem!important;font-weight:600!important}
[data-testid="stButton"],[data-testid="stDownloadButton"]{width:100%}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{width:100%;border-radius:10px;min-height:2.6rem;
border:1px solid var(--border);background:var(--card);color:var(--text);font-weight:600;font-family:var(--font);box-shadow:none}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{border-color:var(--orange);color:var(--orange)}
.stButton>button:disabled,.stDownloadButton>button:disabled{opacity:.45;cursor:not-allowed;border-color:var(--border);color:var(--muted)}
.stApp button p,.stApp button span{color:inherit!important}
button[data-testid="stBaseButton-primary"],button[kind="primary"],button[data-testid="stBaseButton-primaryFormSubmit"],
button[kind="primaryFormSubmit"]{background:var(--orange)!important;border:1px solid var(--orange)!important;color:#1B1F1D!important}
button[data-testid="stBaseButton-primary"]:hover,button[kind="primary"]:hover,button[data-testid="stBaseButton-primaryFormSubmit"]:hover{
filter:brightness(1.07);color:#1B1F1D!important}
button[data-testid="stBaseButton-primary"]:disabled,button[kind="primary"]:disabled{background:var(--card2)!important;border-color:var(--border)!important;color:var(--muted)!important;filter:none}
[data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"],[data-testid="stTextAreaRootElement"],[data-testid="stDateInputField"],
[data-testid="stTimeInputField"],[data-testid="stSelectbox"] [role="group"],[data-testid="stMultiSelect"] [role="group"],[data-baseweb="input"],[data-baseweb="base-input"],
[data-baseweb="textarea"],[data-baseweb="select"]>div{background-color:var(--input)!important;border:1px solid var(--border)!important;border-radius:9px!important}
[data-testid="stTextInputRootElement"] *,[data-testid="stTextAreaRootElement"] *,[data-testid="stDateInputField"] *,[data-testid="stTimeInputField"] *,
[data-testid="stNumberInputContainer"] input,[data-testid="stSelectbox"] [role="group"] *,[data-testid="stMultiSelect"] [role="group"] *,
[data-baseweb="input"] *,[data-baseweb="textarea"] *,[data-baseweb="select"] div{background-color:transparent!important;color:var(--text)!important;-webkit-text-fill-color:var(--text)!important}
[data-testid="stNumberInputContainer"] button{background-color:var(--card2)!important;color:var(--text)!important;border-radius:0!important;border:none!important}
[data-testid="stNumberInputContainer"] button:hover{color:var(--orange)!important}
[data-testid="stSelectbox"] [role="group"] svg,[data-testid="stDateInputField"] svg,[data-baseweb="select"] svg{fill:var(--muted)!important;color:var(--muted)!important}
[data-testid="stTextInputRootElement"]:focus-within,[data-testid="stNumberInputContainer"]:focus-within,[data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stDateInputField"]:focus-within,[data-testid="stSelectbox"] [role="group"]:focus-within{border-color:var(--orange)!important}
input,textarea{color:var(--text)!important;-webkit-text-fill-color:var(--text)!important;font-family:var(--font)!important;caret-color:var(--orange)}
input::placeholder,textarea::placeholder{color:var(--muted)!important;-webkit-text-fill-color:var(--muted)!important;opacity:.85}
[data-testid="stSelectboxVirtualDropdown"],[data-testid="stSelectboxVirtualDropdown"] ul,[data-testid="stMultiSelectVirtualDropdown"],
[role="listbox"],div[data-baseweb="popover"],div[data-baseweb="popover"]>div,[data-baseweb="menu"]{background-color:var(--card)!important;border-radius:10px}
[role="listbox"],[data-testid="stSelectboxVirtualDropdown"]{border:1px solid var(--border)}
[role="listbox"] *,[data-testid="stSelectboxVirtualDropdown"] *,div[data-baseweb="popover"] li{color:var(--text)!important;background-color:transparent}
[role="option"]:hover,[role="option"][data-focused="true"],[role="option"][aria-selected="true"],div[data-baseweb="popover"] li:hover{background-color:var(--hover)!important}
[data-testid="stPopoverBody"]{background-color:var(--card)!important;border:1px solid var(--border);border-radius:12px;min-width:260px}
[data-testid="stPopover"] button{background:var(--orange)!important;border:1px solid var(--orange)!important;color:#1B1F1D!important;font-weight:600}
[data-testid="stTooltipContent"],[data-testid="stTooltipContent"] *{background-color:var(--card2)!important;color:var(--text)!important;font-size:.78rem!important;font-weight:500!important}
[role="dialog"],[data-testid="stDatePicker"],[data-baseweb="calendar"]{background-color:var(--card)!important;color:var(--text)!important}
[role="dialog"] *,[data-baseweb="calendar"] *{color:var(--text)!important}
[role="grid"] [aria-selected="true"],[role="gridcell"][aria-selected="true"] *{background-color:var(--orange)!important;color:#1B1F1D!important}
[data-testid="stCheckbox"] label,[data-testid="stCheckbox"] p{color:var(--text)!important;font-weight:500!important;font-size:.88rem!important}
[data-testid="stRadio"] label,[data-testid="stRadio"] p{color:var(--text)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--card);border:2px dashed var(--border);border-radius:14px;padding:1.4rem}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--orange)}
[data-testid="stFileUploaderDropzone"] *{color:var(--muted)!important}
[data-testid="stFileUploaderDropzone"] button{background:var(--card2)!important;color:var(--text)!important;border:1px solid var(--border)!important}
[data-testid="stFileUploaderFile"] *{color:var(--text)!important}
[data-testid="stExpander"],[data-testid="stExpander"] details{border:1px solid var(--border)!important;border-radius:12px!important;background:var(--card)!important}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:var(--text)!important;font-weight:600}
[data-testid="stForm"]{border:1px solid var(--border)!important;border-radius:14px!important;background:var(--card)!important;padding:1.1rem 1.2rem}
[data-testid="stTab"],button[data-baseweb="tab"]{background:transparent!important;padding:.55rem 1rem}
[data-testid="stTab"] p,button[data-baseweb="tab"] p{color:var(--muted)!important;font-weight:600;font-size:.9rem}
[data-testid="stTab"][aria-selected="true"] p,button[data-baseweb="tab"][aria-selected="true"] p{color:var(--orange)!important}
[data-testid="stTab"][aria-selected="true"]{border-bottom-color:var(--orange)!important}
[data-testid="stTab"] .react-aria-SelectionIndicator,.react-aria-SelectionIndicator,[data-baseweb="tab-highlight"]{background-color:var(--orange)!important;background:var(--orange)!important}
[role="tablist"],[data-baseweb="tab-border"]{border-color:var(--border)!important}
[data-testid="stSpinner"] *{color:var(--muted)!important}
[class*="st-key-card_"]{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.25rem;margin-bottom:.9rem}
[class*="st-key-segb_"] button{min-height:2.3rem;height:2.3rem;border-radius:999px;font-size:.85rem;white-space:nowrap;padding:0 .6rem;background:transparent;color:var(--muted);border:1px solid var(--border)}
[class*="st-key-segb_"] button:hover{color:var(--text);border-color:var(--muted)}
[class*="__sel"] button{background:rgba(229,138,50,.14)!important;color:var(--orange)!important;border-color:var(--orange)!important}
[class*="st-key-qa_"] button{min-height:3rem;height:auto;padding:.4rem .9rem}
[class*="st-key-qa_"] button p{white-space:normal;line-height:1.25}
[class*="st-key-dangerbtn"] button{background:var(--alert)!important;border-color:var(--alert)!important;color:#fff!important}

/* ---------------- components ---------------- */
.mg-ph{display:flex;align-items:center;gap:.95rem;margin:0}
.mg-ph-ic{width:52px;height:52px;border-radius:14px;background:rgba(229,138,50,.12);border:1px solid rgba(229,138,50,.3);
display:flex;align-items:center;justify-content:center;color:var(--orange);flex:none}
.mg-ph h1{margin:0;font-size:1.65rem;font-weight:700;line-height:1.2;padding:0}
.mg-ph p{margin:.2rem 0 0 0;color:var(--muted)!important;font-size:.9rem;line-height:1.4}
.mg-divider{height:1px;background:var(--border);margin:.9rem 0 1.1rem 0}
.mg-sec{margin:1.3rem 0 .55rem 0}
.mg-sec h3{margin:0;font-size:1.02rem;font-weight:700;padding:0}
.mg-sec p{margin:.15rem 0 0 0;color:var(--muted)!important;font-size:.84rem}
.mg-intro{background:var(--card);border:1px solid var(--border);border-left:4px solid var(--orange);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1rem}
.mg-intro p{margin:0 0 .8rem 0;font-size:.95rem;line-height:1.6;color:var(--text)}
.mg-steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.7rem}
.mg-step{display:flex;gap:.65rem;align-items:flex-start;font-size:.84rem;color:var(--muted);line-height:1.4}
.mg-step b{flex:none;width:24px;height:24px;border-radius:50%;background:rgba(229,138,50,.15);color:var(--orange);display:flex;align-items:center;justify-content:center;font-size:.76rem}
.mg-kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:.9rem 1rem;min-height:136px;position:relative;box-sizing:border-box}
.mg-kpi .top{display:flex;justify-content:space-between;align-items:flex-start;gap:.3rem;margin-bottom:.4rem}
.mg-kpi .l{font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600;padding-right:.5rem;line-height:1.3}
.mg-kpi .ic{color:var(--muted);display:flex}
.mg-kpi .v{font-size:1.65rem;font-weight:700;line-height:1.15;color:var(--text);word-break:break-word}
.mg-kpi .v.sm{font-size:1.12rem;line-height:1.35}
.mg-kpi .s{font-size:.76rem;color:var(--muted);margin-top:.25rem;line-height:1.35}
.mg-kpi.ok{border-top:3px solid var(--sage)}.mg-kpi.warn{border-top:3px solid var(--amber-solid)}.mg-kpi.bad{border-top:3px solid var(--alert)}
.mg-kpi.ok .ic{color:var(--ok)}.mg-kpi.warn .ic{color:var(--amber)}.mg-kpi.bad .ic{color:var(--red)}
.mg-chip{display:inline-flex;align-items:center;gap:.4rem;border:1px solid var(--border);background:var(--card);color:var(--text);
border-radius:999px;padding:.2rem .7rem;margin:.12rem .25rem .12rem 0;font-size:.76rem;font-weight:500;white-space:nowrap}
.mg-chip.plain{background:var(--card2)}
.mg-dot{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;background:var(--sage);flex:none}
.mg-dot.bad{background:var(--alert)}.mg-dot.warn{background:var(--amber-solid)}
.mg-pill{display:inline-flex;align-items:center;border-radius:999px;padding:.1rem .6rem;font-size:.73rem;font-weight:600;border:1px solid transparent;white-space:nowrap}
.mg-pill.ok{color:var(--ok);background:rgba(120,135,124,.18);border-color:rgba(120,135,124,.4)}
.mg-pill.warn{color:var(--amber);background:rgba(214,166,58,.14);border-color:rgba(214,166,58,.4)}
.mg-pill.bad{color:var(--red);background:rgba(185,75,61,.16);border-color:rgba(185,75,61,.45)}
.mg-pill.neutral{color:var(--muted);background:var(--card2);border-color:var(--border)}
.mg-alert{display:flex;gap:.7rem;border-radius:12px;padding:.8rem 1rem;border:1px solid;margin:.5rem 0;font-size:.88rem;line-height:1.5;color:var(--text)}
.mg-alert .ic{flex:none;margin-top:.1rem;display:flex}
.mg-alert.info{background:rgba(120,135,124,.1);border-color:rgba(120,135,124,.35)}.mg-alert.info .ic{color:var(--ok)}
.mg-alert.ok{background:rgba(120,135,124,.18);border-color:rgba(120,135,124,.5)}.mg-alert.ok .ic{color:var(--ok)}
.mg-alert.warn{background:rgba(214,166,58,.11);border-color:rgba(214,166,58,.45)}.mg-alert.warn .ic{color:var(--amber)}
.mg-alert.error{background:rgba(185,75,61,.13);border-color:rgba(185,75,61,.5)}.mg-alert.error .ic{color:var(--red)}
.mg-mcard{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:1.4rem 1.5rem 1.2rem 1.5rem;margin-bottom:.6rem;
min-height:19rem;position:relative;overflow:hidden}
.mg-mcard:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--orange)}
.mg-mcard.sage:before{background:var(--sage)}
.mg-mcard .hd{display:flex;gap:1rem;align-items:center;margin-bottom:.8rem}
.mg-mcard .ill{width:76px;height:62px;border-radius:14px;background:var(--card2);border:1px solid var(--border);display:flex;
align-items:center;justify-content:center;color:var(--orange);flex:none}
.mg-mcard.sage .ill{color:var(--ok)}
.mg-mcard h3{margin:0;font-size:1.3rem;font-weight:700;padding:0}
.mg-mcard .tag{color:var(--muted);font-size:.8rem;margin-top:.15rem}
.mg-mcard p{font-size:.9rem;line-height:1.55;color:var(--text);margin:.2rem 0 .8rem 0}
.mg-mcard .lbl{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:600;margin:.5rem 0 .3rem 0}
.mg-mcard .m{display:flex;gap:1.8rem;margin-top:.9rem;padding-top:.8rem;border-top:1px solid var(--border);font-size:.68rem;
text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:600}
.mg-mcard .m b{display:block;font-size:1.25rem;color:var(--text);letter-spacing:0;text-transform:none;font-weight:700}
.mg-tablewrap{overflow:auto;border:1px solid var(--border);border-radius:12px;background:var(--card);margin:.3rem 0 .7rem 0}
table.mg-table{border-collapse:collapse;width:100%;font-size:.8rem}
.mg-table th{position:sticky;top:0;background:var(--card2);color:var(--muted);text-align:left;padding:.65rem .9rem;font-weight:600;
font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap;border-bottom:1px solid var(--border);z-index:1}
.mg-table td{padding:.6rem .9rem;border-bottom:1px solid var(--border);white-space:nowrap;color:var(--text)}
.mg-table tr:last-child td{border-bottom:none}.mg-table tbody tr:hover td{background:var(--hover)}
.mg-table tr.fail td{background:rgba(185,75,61,.1)}
.mg-empty{background:var(--card);border:1px dashed var(--border);border-radius:16px;padding:2rem 1.5rem;text-align:center;margin:.4rem 0 .8rem 0}
.mg-empty .ic{width:60px;height:60px;border-radius:16px;margin:0 auto .8rem auto;background:rgba(229,138,50,.12);color:var(--orange);
display:flex;align-items:center;justify-content:center}
.mg-empty h3{margin:0 0 .35rem 0;font-size:1.12rem;padding:0}
.mg-empty p{margin:0 auto;max-width:560px;color:var(--muted)!important;font-size:.88rem;line-height:1.55}
.mg-empty ol{display:inline-block;text-align:left;margin:.7rem auto 0 auto;color:var(--text);font-size:.86rem;line-height:1.7;padding-left:1.2rem}
.mg-task{background:var(--card);border:1px solid var(--border);border-left:3px solid var(--sage);border-radius:12px;padding:.75rem .9rem;margin-bottom:.55rem}
.mg-task.p-high{border-left-color:var(--amber-solid)}.mg-task.p-critical{border-left-color:var(--alert)}.mg-task.done{opacity:.72}
.mg-task .d{font-size:.86rem;font-weight:600;color:var(--text);line-height:1.35}
.mg-task .mt{font-size:.75rem;color:var(--muted);margin:.25rem 0 .45rem 0;line-height:1.4}
.mg-task .ft{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center}
.mg-task .dt{margin-left:auto;font-size:.72rem;color:var(--muted)}
.mg-col-h{display:flex;justify-content:space-between;align-items:center;font-size:.76rem;text-transform:uppercase;letter-spacing:.09em;
color:var(--muted);font-weight:700;margin:.2rem 0 .6rem 0;padding-bottom:.45rem;border-bottom:2px solid var(--border)}
.mg-col-h span.n{background:var(--card2);border-radius:999px;padding:.05rem .55rem;color:var(--text)}
.mg-info{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.9rem 1.4rem;margin:.2rem 0 .6rem 0}
.mg-info .k{font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:600}
.mg-info .v{font-size:.92rem;color:var(--text);font-weight:500;word-break:break-word}
.mg-file{display:flex;align-items:center;gap:.9rem;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.8rem 1rem;margin:.6rem 0}
.mg-file .ic{width:40px;height:40px;border-radius:10px;background:rgba(120,135,124,.18);color:var(--ok);display:flex;align-items:center;justify-content:center;flex:none}
.mg-file b{display:block;color:var(--text);font-size:.92rem}.mg-file small{color:var(--muted);font-size:.78rem}
</style>
""")


def inject_css() -> None:
    collapsed = st.session_state.collapsed
    css = CSS.substitute(
        **THEMES[st.session_state.theme],
        sbw="76px" if collapsed else "290px",
        sbpad=".5rem" if collapsed else ".9rem",
        navpad="0" if collapsed else ".8rem",
        navjustify="center" if collapsed else "flex-start",
        collapsejustify="center" if collapsed else "flex-end",
        brandjustify="justify-content:center;margin-bottom:.6rem;" if collapsed else "",
    )
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# ICONS (inline SVG, so they follow the theme colour)
# ============================================================================
ICONS = {
    "dashboard": '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
    "registry": '<rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4M12 16h4M8 11h.01M8 16h.01"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "cog": '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9L7 7M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 16v-5M12 16V8M17 16v-8"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    "check": '<path d="M20 6L9 17l-5-5"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5M12 3v12"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M9 13h6M9 17h6"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.3-4.3"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    "compressor": '<rect x="3" y="21" width="34" height="15" rx="7.5"/><path d="M10 36v5M30 36v5M1 41h40"/><rect x="42" y="17" width="16" height="20" rx="3"/><path d="M37 29h5"/><circle cx="19" cy="12" r="6"/><path d="M19 12l3-3M19 18v3"/><path d="M46 23h8M46 27h8M46 31h8"/>',
}


def svg(name: str, size: int = 20) -> str:
    view = "0 0 64 48" if name == "compressor" else "0 0 24 24"
    return (f'<svg width="{size}" height="{size}" viewBox="{view}" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


# ============================================================================
# PLAIN-LANGUAGE TEXT
# ============================================================================
FAIL_LABEL = "Possible failure condition"
OK_LABEL = "No failure predicted"
DATA_UNAVAILABLE = ("Your saved data is not available right now, so this page cannot show or save information. "
                    "Please try again in a moment or contact your administrator.")
PREDICTION_UNAVAILABLE = ("Predictions are not available for this machine type right now. "
                          "Please try again in a moment or contact your administrator.")
NOTICE = ("The prediction models were trained on example datasets: a metro-train air compressor dataset and a synthetic "
          "(computer-generated) industrial dataset. They do not know your own machines, so use a result as a prompt to inspect, "
          "not as a diagnosis. A prediction never confirms a physical failure and does not estimate remaining life or time to failure.")
MEANING = {
    "compressor": ("A possible failure condition means the readings resemble the pattern recorded during known air-leak failure "
                   "periods in the training data. A reading outside that pattern is not guaranteed to be healthy."),
    "ai4i": ("A possible failure condition means the readings resemble records labelled as machine failures in the training "
             "data, which is synthetic example data."),
}
CALC_TEXT = {
    "Temperature_Difference": "Process temperature minus air temperature",
    "Power_Proxy": "Rotational speed × torque (a relative measure of mechanical load, not electrical power)",
    "Tool_Wear_Category": "Tool wear grouped as Low (up to 100 min), Medium (101-200 min) or High (over 200 min)",
    "Torque_Per_RPM": "Torque divided by rotational speed",
    "Pressure_Difference_TP3_TP2": "Panel pressure (TP3) minus compressor pressure (TP2)",
    "Pressure_Difference_Reservoir_TP2": "Reservoir pressure minus compressor pressure (TP2)",
    "Motor_Current_Change": "Motor current now minus the previous reading",
    "Oil_Temperature_Change": "Oil temperature now minus the previous reading",
}
DASH_CARDS = {
    "compressor": {
        "description": "Monitor compressor operating conditions using pressure, temperature, motor current and reservoir sensor readings.",
        "params": ["Compressor pressure", "Panel pressure", "Reservoir pressure", "Oil temperature", "Motor current"],
    },
    "ai4i": {
        "description": "Monitor industrial operating parameters such as air temperature, process temperature, rotational speed, torque and tool wear.",
        "params": ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"],
    },
}
CATEGORY_CHOICES = ["compressor", "ai4i", "none"]
CATEGORY_LABELS = {
    "compressor": "Industrial Air Compressor (prediction supported)",
    "ai4i": "General Industrial Machine (prediction supported)",
    "none": "Other machine (maintenance tracking only)",
}
TASK_STATUS_OPTIONS = ["Pending", "In Progress", "Completed"]
OTHER_MACHINE = "Other / Enter machine name"


def pretty_feature(name: str) -> tuple[str, str | None]:
    """Readable feature name and, for one-hot columns, the category it stands for."""
    for prefix, label in (("Type_", "Product type"), ("Tool_Wear_Category_", "Tool wear category")):
        if name.startswith(prefix):
            value = name[len(prefix):]
            return f"{label} = {value}", value
    return name.replace("_", " "), None


# ============================================================================
# HTML HELPERS (all dynamic text is escaped)
# ============================================================================
def esc(value) -> str:
    return html.escape(str(value)).replace("$", "&#36;")


def notify(kind: str, text: str) -> None:
    icon = {"info": "info", "ok": "check", "warn": "alert", "error": "alert"}[kind]
    st.markdown(f'<div class="mg-alert {kind}"><span class="ic">{svg(icon, 18)}</span><div>{esc(text)}</div></div>',
                unsafe_allow_html=True)


def flash(kind: str, text: str) -> None:
    st.session_state["flash"] = (kind, text)


def show_flash() -> None:
    item = st.session_state.pop("flash", None)
    if item:
        notify(*item)


def chip(text: str, tone: str = "") -> str:
    dot = f'<span class="mg-dot {"" if tone == "ok" else tone}"></span>' if tone in ("ok", "bad", "warn") else ""
    return f'<span class="mg-chip{"" if dot else " plain"}">{dot}{esc(text)}</span>'


PILL_TONES = {
    FAIL_LABEL: "bad", "Error": "bad", "Critical": "bad", "Inactive": "bad", "Overdue": "bad",
    "Reviewed - issue confirmed by inspection": "bad",
    "Warning": "warn", "High": "warn", "Pending review": "warn", "Under inspection": "warn", "In Progress": "warn",
    OK_LABEL: "ok", "Active": "ok", "Completed": "ok", "Reviewed - no issue found": "ok", "Low": "ok",
    "Pending": "neutral", "Medium": "neutral", "Cancelled": "neutral",
}
PILL_COLUMNS = {"Status", "Level", "Model result", "Priority", "Review", "Review status", "State"}


def pill(text: str) -> str:
    return f'<span class="mg-pill {PILL_TONES.get(str(text), "neutral")}">{esc(text)}</span>'


def kpi(label: str, value, sub: str = "", tone: str = "", icon: str = "activity", small: bool = False) -> str:
    return (f'<div class="mg-kpi {tone}"><div class="top"><span class="l">{esc(label)}</span><span class="ic">{svg(icon, 17)}</span></div>'
            f'<div class="v {"sm" if small else ""}">{esc(value)}</div><div class="s">{esc(sub) if sub else "&nbsp;"}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    """items: (label, value, sub, tone, icon[, small])"""
    for column, item in zip(st.columns(len(items)), items):
        column.markdown(kpi(*item), unsafe_allow_html=True)


def logo_html(size: int = 40) -> str:
    if cfg.LOGO_PATH.exists():
        data = base64.b64encode(cfg.LOGO_PATH.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" width="{size}" height="{size}" style="border-radius:10px;flex:none" alt="MachineGuard AI"/>'
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 40 40"><rect width="40" height="40" rx="9" fill="#202725"/>'
            '<path d="M20 6l13 5v10L20 34 7 21V11z" fill="#E58A32"/></svg>')


def page_header(title: str, subtitle: str = "", icon: str = "dashboard") -> None:
    """Consistent header with the theme toggle at the top-right of the content area."""
    dark = st.session_state.theme == "dark"
    left, right = st.columns([24, 1.6], vertical_alignment="center")
    with left:
        st.markdown(f'<div class="mg-ph"><div class="mg-ph-ic">{svg(icon, 26 if icon != "compressor" else 34)}</div>'
                    f'<div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div></div>', unsafe_allow_html=True)
    with right:
        st.button("", icon=":material/light_mode:" if dark else ":material/dark_mode:", key="themetoggle", on_click=toggle_theme,
                  help="Switch to light mode" if dark else "Switch to dark mode")
    st.markdown('<div class="mg-divider"></div>', unsafe_allow_html=True)
    show_flash()


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="mg-sec"><h3>{esc(title)}</h3>{f"<p>{esc(caption)}</p>" if caption else ""}</div>', unsafe_allow_html=True)


def spacer(rem: float = .6) -> None:
    st.markdown(f'<div style="height:{rem}rem"></div>', unsafe_allow_html=True)


def help_box(bullets: list[str], title: str = "How this page works") -> None:
    with st.expander(title):
        for line in bullets:
            st.markdown(f"- {line}")


def info_grid(pairs: list[tuple]) -> None:
    cells = "".join(f'<div><div class="k">{esc(k)}</div><div class="v">{v}</div></div>' for k, v in pairs)
    st.markdown(f'<div class="mg-info">{cells}</div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, text: str, steps: list[str] | None = None, action: tuple | None = None) -> None:
    """action = (label, callback, args, key). Explains why the page is empty and what to do next."""
    steps_html = f'<ol>{"".join(f"<li>{esc(s)}</li>" for s in steps)}</ol>' if steps else ""
    st.markdown(f'<div class="mg-empty"><div class="ic">{svg(icon, 28)}</div><h3>{esc(title)}</h3><p>{esc(text)}</p>{steps_html}</div>',
                unsafe_allow_html=True)
    if action:
        label, callback, args, key = action
        _, middle, _ = st.columns([1.2, 1, 1.2])
        with middle:
            st.button(label, key=key, on_click=callback, args=args, type="primary")


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        text = f"{float(value):.4f}".rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def html_table(df: pd.DataFrame, max_rows: int | None = None, height: int = 420, pct_cols=(), flag_col: str | None = None) -> None:
    """Theme-aware, horizontally scrollable table. Status-like columns render as pills."""
    if df is None or df.empty:
        notify("info", "Nothing to show here yet.")
        return
    max_rows = max_rows or cfg.TABLE_PREVIEW_ROWS
    shown = df.head(max_rows)
    columns = list(shown.columns)
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = []
    for record in shown.itertuples(index=False, name=None):
        cls, cells = "", []
        for column, value in zip(columns, record):
            if column in pct_cols and isinstance(value, (int, float, np.floating)) and not pd.isna(value):
                text = f"{value * 100:.1f}%"
            else:
                text = _fmt(value)
            if column == flag_col and text in (FAIL_LABEL, "Error"):
                cls = "fail"
            cells.append(f"<td>{pill(text) if column in PILL_COLUMNS else esc(text)}</td>")
        body.append(f'<tr class="{cls}">{"".join(cells)}</tr>')
    st.markdown(f'<div class="mg-tablewrap" style="max-height:{height}px"><table class="mg-table"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)
    if len(df) > max_rows:
        st.caption(f"Showing the first {max_rows:,} of {len(df):,} rows. Download the file for the complete data.")


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    t = THEMES[st.session_state.theme]
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=t["text"], family="Inter, Segoe UI, system-ui, sans-serif", size=12),
                      colorway=t["colorway"], margin=dict(l=10, r=10, t=56, b=10),
                      title=dict(font=dict(size=15, color=t["text"]), x=0.01),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t["text"]), orientation="h", y=-0.18),
                      hoverlabel=dict(bgcolor=t["card2"], font_color=t["text"], bordercolor=t["border"]))
    fig.update_xaxes(gridcolor=t["border"], zerolinecolor=t["border"], tickfont=dict(color=t["muted"]), title_font=dict(color=t["muted"]))
    fig.update_yaxes(gridcolor=t["border"], zerolinecolor=t["border"], tickfont=dict(color=t["muted"]), title_font=dict(color=t["muted"]))
    return fig


CLASS_COLORS = {OK_LABEL: "#78877C", FAIL_LABEL: "#B94B3D"}


def aware(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def local_time(value, fmt: str = "%d %b %Y, %H:%M") -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return aware(value).astimezone().strftime(fmt)
    return str(value)


# ============================================================================
# STATE, CACHED RESOURCES, CALLBACKS
# ============================================================================
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("theme", "dark")
    ss.setdefault("collapsed", False)
    ss.setdefault("page", "dashboard")
    ss.setdefault("results", {})
    ss.setdefault("uploads", {})
    ss.setdefault("last_recover", _time.time())


@st.cache_resource(show_spinner=False)
def get_db() -> MachineGuardDB:
    return MachineGuardDB()


@st.cache_resource(show_spinner=False)
def get_model(key: str):
    try:
        return pred.load_model(key), None
    except pred.PredictionError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error while loading the model: {exc}"


def maybe_recover() -> None:
    """Quietly retry a failed connection or model load (at most every 30 s); nothing is shown to the user."""
    now = _time.time()
    if now - st.session_state.last_recover < 30:
        return
    db = get_db()
    models_missing = any(get_model(k)[0] is None for k in cfg.MODELS)
    if db.connected and not models_missing:
        return
    st.session_state.last_recover = now
    if not db.connected:
        db.connect()
    if models_missing:
        get_model.clear()


def friendly_error(exc: Exception) -> str:
    message = str(exc)
    if any(k in message for k in ("already exists", "required", "not found")):
        return message
    return "Something went wrong while accessing your saved data. Please try again."


def db_call(fn, *args, default=None, quiet: bool = False, **kwargs):
    try:
        return fn(*args, **kwargs)
    except DatabaseError as exc:
        if not quiet:
            notify("error", friendly_error(exc))
        return default


def go_to(page: str) -> None:
    st.session_state.page = page


def go_view(page: str, view_name: str, view: str) -> None:
    st.session_state.page = page
    st.session_state[f"view_{view_name}"] = view


def set_view(state_key: str, value: str) -> None:
    st.session_state[state_key] = value


def toggle_theme() -> None:
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


def toggle_sidebar() -> None:
    st.session_state.collapsed = not st.session_state.collapsed


def view_switch(name: str, options: list[tuple[str, str]], default: str) -> str:
    """Pill-style view switcher, so other widgets can change the active view."""
    state_key = f"view_{name}"
    st.session_state.setdefault(state_key, default)
    if st.session_state[state_key] not in [v for v, _ in options]:
        st.session_state[state_key] = default
    current = st.session_state[state_key]
    columns = st.columns([len(label) + 3 for _, label in options] + [40])
    for column, (value, label) in zip(columns, options):
        with column:
            st.button(label, key=f"segb_{name}_{value}__{'sel' if value == current else 'idle'}", on_click=set_view,
                      args=(state_key, value))
    spacer(.4)
    return current


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar() -> None:
    collapsed = st.session_state.collapsed
    with st.sidebar:
        if collapsed:
            st.markdown(f'<div class="mg-brand">{logo_html(38)}</div>', unsafe_allow_html=True)
            st.button("", icon=":material/chevron_right:", key="collapsebtn", on_click=toggle_sidebar, help="Expand sidebar")
        else:
            brand, control = st.columns([6.2, 1], vertical_alignment="center")
            with brand:
                st.markdown(f'<div class="mg-brand">{logo_html(38)}<div class="t"><b>{esc(cfg.APP_NAME)}</b></div></div>', unsafe_allow_html=True)
            with control:
                st.button("", icon=":material/chevron_left:", key="collapsebtn", on_click=toggle_sidebar, help="Collapse sidebar")
        if not collapsed:
            st.markdown('<div class="mg-navlabel">Navigation</div>', unsafe_allow_html=True)
        else:
            spacer(.6)
        for key, label, icon in cfg.PAGES:
            active = st.session_state.page == key
            st.button("" if collapsed else label, icon=f":material/{icon}:", key=f"{'navact' if active else 'nav'}_{key}",
                      on_click=go_to, args=(key,), help=label if collapsed else None)


# ============================================================================
# SHARED PREDICTION UI
# ============================================================================
def pick_machine(key: str) -> tuple[dict, bool]:
    """Returns (machine, is_registered). Only machines assigned to this prediction model are offered."""
    spec = cfg.MODELS[key]
    db = get_db()
    machines = []
    if db.connected:
        machines = [m for m in db_call(db.list_machines, True, default=[], quiet=True) if m.get("assigned_model") == key]
    options = {f'{m["name"]} ({m["machine_id"]})': m for m in machines}
    adhoc = "A machine that is not in the registry"
    choice = st.selectbox("Which machine are these readings from?", [*options.keys(), adhoc], key=f"machine_{key}",
                          help="Register machines in the Machine Registry so results are linked to them.")
    if choice == adhoc:
        return {"machine_id": f"UNREGISTERED-{key.upper()}", "name": f'Unregistered {spec["machine_name"]}'}, False
    return options[choice], True


def _summary_for_db(result: pred.PredictionResult, machine: dict) -> dict:
    probs = result.results["Failure probability"] if result.has_probability else None
    return {
        "created_at": result.created_at, "machine_id": machine["machine_id"], "machine_name": machine["name"],
        "model_key": result.machine_key, "model_name": result.model_name, "algorithm": result.algorithm,
        "input_method": result.input_method, "source_name": result.source_name,
        "n_input_rows": result.n_input_rows, "n_records": result.n_records, "n_invalid": result.n_invalid,
        "n_failure": result.n_failure, "n_non_failure": result.n_non_failure,
        "has_probability": result.has_probability,
        "mean_failure_probability": float(probs.mean()) if probs is not None else None,
        "max_failure_probability": float(probs.max()) if probs is not None else None,
        "warning_count": int((result.issues["Severity"] == "Warning").sum()) if len(result.issues) else 0,
    }


def store_result(key: str, result: pred.PredictionResult, machine: dict) -> None:
    entry = {"result": result, "machine": machine, "saved_id": None, "save_error": None}
    if result.error is None:
        db = get_db()
        if db.connected:
            try:
                entry["saved_id"] = db.save_prediction(_summary_for_db(result, machine), result.detail_records())
            except DatabaseError as exc:
                entry["save_error"] = friendly_error(exc)
        else:
            entry["save_error"] = "Your saved data is not available right now."
    st.session_state.results[key] = entry


def clear_result(key: str) -> None:
    st.session_state.results.pop(key, None)


def read_upload(data: bytes, name: str, sheet: str | None) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0, engine="openpyxl")


def issue_table(issues: pd.DataFrame) -> pd.DataFrame:
    shown = issues.copy()
    shown["File row"] = shown["File row"].map(lambda v: "Previous reading" if v == 0 else v)
    return shown.rename(columns={"File row": "Row in file", "Column": "Field", "Severity": "Level"})


def friendly_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Plain-language version of a prediction result table (used for display and downloads)."""
    out = frame.copy()
    out["Prediction"] = out["Prediction"].map({"Predicted failure": FAIL_LABEL, "Predicted non-failure": OK_LABEL}).fillna(out["Prediction"])
    out = out.drop(columns=["Predicted class"], errors="ignore")
    return out.rename(columns={"File row": "Row in file", "Prediction": "Model result", "Failure probability": "Failure likelihood score"})


# ----------------------------------------------------------------------------
# Manual input
# ----------------------------------------------------------------------------
def _input_widget(key: str, bundle: pred.ModelBundle, column: str, stats: dict):
    meta = cfg.input_meta(key, column)
    short = meta["label"].split(" - ")[0]
    label = f'{short} [{meta["unit"]}]' if meta["unit"] and f'[{meta["unit"]}]' not in short else short
    if " - " in meta["label"] and not meta.get("help"):
        meta = {**meta, "help": meta["label"]}
    wkey = f"man_{key}_{column}"
    if column == "Type":
        return st.selectbox(label, cfg.AI4I_TYPES, key=wkey, help=meta.get("help") or None)
    if pred.is_digital(bundle, column):
        return float(st.selectbox(label, [0, 1], key=wkey, help="On/off signal: enter 1 for on, 0 for off."))
    default = stats[column]["median"] if column in stats else meta["default"]
    return st.number_input(label, value=float(round(default, 4)), step=float(meta["step"]), format="%.4f",
                           key=wkey, help=meta.get("help") or None)


def manual_panel(key: str, bundle: pred.ModelBundle, machine: dict) -> None:
    stats = pred.stats_for(bundle)
    if stats:
        notify("info", "The boxes start with typical values from the example data so the form is filled in. "
                       "Replace every value with your own readings before running a prediction.")
    values: dict = {}
    previous = None
    if key == "ai4i":
        section("Operating conditions", "Enter the current readings. Extra values are calculated for you.")
        with st.container(key="card_man_ai4i"):
            for row in (bundle.input_columns[:3], bundle.input_columns[3:]):
                for column_widget, column in zip(st.columns(3), row):
                    with column_widget:
                        values[column] = _input_widget(key, bundle, column, stats)
    else:
        analog = [c for c in bundle.input_columns if not pred.is_digital(bundle, c)]
        digital = [c for c in bundle.input_columns if pred.is_digital(bundle, c)]
        section("Sensor readings", "Pressures in bar, oil temperature in °C and motor current in A. Hover over the ? icon for a description.")
        with st.container(key="card_man_analog"):
            grid = st.columns(3)
            for i, column in enumerate(analog):
                with grid[i % 3]:
                    values[column] = _input_widget(key, bundle, column, stats)
        if digital:
            section("On/off signals", "Enter 1 when a signal is on and 0 when it is off.")
            with st.container(key="card_man_digital"):
                grid = st.columns(4)
                for i, column in enumerate(digital):
                    with grid[i % 4]:
                        values[column] = _input_widget(key, bundle, column, stats)
        section("Previous reading", "Two values describe how the compressor changed since its last reading, so the previous reading is needed.")
        with st.container(key="card_man_prev"):
            st.markdown("Enter the motor current and oil temperature from the reading taken **immediately before** this one.")
            unavailable = st.checkbox("I do not have a previous reading", key="man_compressor_noprev")
            if unavailable:
                notify("warn", "Without a previous reading, the two change values are filled in with typical values, so this result is less reliable.")
            else:
                previous = {}
                for column_widget, column in zip(st.columns(2), cfg.COMPRESSOR_PREVIOUS_COLUMNS):
                    meta = cfg.input_meta(key, column)
                    default = stats[column]["median"] if column in stats else 0.0
                    with column_widget:
                        previous[column] = st.number_input(
                            f'Previous {meta["label"].lower()} [{meta["unit"]}]', value=float(round(default, 4)),
                            step=float(meta["step"]), format="%.4f", key=f"man_prev_{column}")

    section("Check your entries", "Calculated values update as you type.")
    frame, _, first_row = pred.manual_frame(bundle, values, previous)
    report = pred.validate_input(bundle, frame, first_row)
    engineered = pred.preview_features(bundle, values, previous) if not report.fatal else None
    rows = [{"Item": c, "Value": values[c], "Source": "Entered by you", "Unit / how it is calculated": cfg.input_meta(key, c)["unit"]}
            for c in bundle.input_columns]
    if previous:
        rows += [{"Item": f"Previous {c}", "Value": v, "Source": "Previous reading", "Unit / how it is calculated": ""} for c, v in previous.items()]
    if engineered is not None:
        rows += [{"Item": c, "Value": v, "Source": "Calculated", "Unit / how it is calculated": CALC_TEXT.get(c, "")} for c, v in engineered.items()]
    html_table(pd.DataFrame(rows), height=340)

    has_error = report.n_invalid > 0 or bool(report.fatal)
    for _, issue in issue_table(report.issues).iterrows():
        notify("error" if issue["Level"] == "Error" else "warn", f'{issue["Field"]}: {issue["Problem"]} (you entered: {issue["Value"]})')
    if has_error:
        notify("error", "Please correct the entries above to enable the prediction.")
    if st.button("Run prediction", type="primary", key=f"run_manual_{key}", disabled=has_error, icon=":material/play_arrow:"):
        try:
            with st.spinner("Checking the readings…"):
                result = pred.predict_manual(bundle, values, previous)
            store_result(key, result, machine)
        except pred.PredictionError:
            notify("error", "The prediction could not be completed. Please check the entries and try again.")


# ----------------------------------------------------------------------------
# File upload
# ----------------------------------------------------------------------------
def upload_panel(key: str, bundle: pred.ModelBundle, machine: dict) -> None:
    spec = cfg.MODELS[key]
    section("Upload a data file", f"Check many readings at once for the {spec['machine_name']}.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("".join(chip(t) for t in ("CSV (.csv)", "Excel (.xlsx)", f"Up to {cfg.MAX_UPLOAD_ROWS:,} rows")), unsafe_allow_html=True)
        spacer(.2)
        file = st.file_uploader("Drag and drop a file here, or browse", type=["csv", "xlsx"], key=f"uploader_{key}")
    with right:
        st.download_button("Download a blank template", data=pred.blank_template(bundle).to_csv(index=False),
                           file_name=f"{key}_input_template.csv", mime="text/csv", key=f"tpl_{key}")
        with st.expander("Which columns does the file need?", expanded=False):
            lines = [f"`{c}`" for c in bundle.input_columns]
            if key == "compressor":
                lines.append("`timestamp` (optional): rows are sorted by time before changes between readings are calculated. "
                             "Without it, the file order is treated as time order.")
            st.markdown("  \n".join(lines))
            st.caption("Each machine type only accepts the columns listed here.")
    if file is None:
        st.session_state.uploads.pop(key, None)
        st.caption("Upload a file to preview it and see the checks before predicting.")
        return
    data = file.getvalue()
    sheet = None
    if file.name.lower().endswith(".xlsx"):
        try:
            sheets = pd.ExcelFile(io.BytesIO(data), engine="openpyxl").sheet_names
        except Exception:  # noqa: BLE001
            notify("error", "The Excel file could not be opened. Please check that it is a valid .xlsx file.")
            return
        sheet = sheets[0] if len(sheets) == 1 else st.selectbox("Which worksheet should be used?", sheets, key=f"sheet_{key}")

    signature = (file.name, len(data), sheet)
    cached = st.session_state.uploads.get(key)
    if not cached or cached["sig"] != signature:
        with st.spinner("Reading and checking the file…"):
            try:
                frame = read_upload(data, file.name, sheet)
            except Exception:  # noqa: BLE001
                notify("error", "The file could not be read. Please check that it is a valid CSV or Excel file.")
                return
            if len(frame) > cfg.MAX_UPLOAD_ROWS:
                notify("error", f"The file has {len(frame):,} rows; the limit is {cfg.MAX_UPLOAD_ROWS:,}. Please split it and try again.")
                return
            cached = {"sig": signature, "frame": frame, "report": pred.validate_input(bundle, frame, 2)}
            st.session_state.uploads[key] = cached
    frame, report = cached["frame"], cached["report"]

    size_kb = len(data) / 1024
    size_text = f"{size_kb / 1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
    st.markdown(f'<div class="mg-file"><div class="ic">{svg("file", 20)}</div><div><b>{esc(file.name)}</b>'
                f'<small>{size_text} · {len(frame):,} rows · {frame.shape[1]} columns' + (f' · worksheet "{esc(sheet)}"' if sheet else "")
                + '</small></div></div>', unsafe_allow_html=True)
    section("File preview", "The first 10 rows. Scroll sideways to see every column.")
    html_table(frame.head(10), height=300)
    for note in report.notes:
        notify("info", note)
    if report.fatal:
        notify("error", report.fatal)
        return

    valid = report.n_rows - report.n_invalid
    section("File check results")
    kpi_row([("Total rows", f"{report.n_rows:,}", "in the file", "", "file"),
             ("Valid rows", f"{valid:,}", "will be predicted", "ok" if valid else "bad", "check"),
             ("Rows with errors", f"{report.n_invalid:,}", "listed below, not predicted", "bad" if report.n_invalid else "", "alert"),
             ("Warnings", f"{report.n_warnings:,}", "predicted, but worth a look", "warn" if report.n_warnings else "", "info")])
    spacer(.5)
    if len(report.issues):
        st.caption("Rows with errors are never silently removed. They are listed here and left out of the prediction only.")
        html_table(issue_table(report.issues), max_rows=cfg.ISSUE_PREVIEW_ROWS, height=300, flag_col="Level")
        st.download_button("Download the full issue report (CSV)", data=issue_table(report.issues).to_csv(index=False),
                           file_name=f"{key}_file_issues.csv", mime="text/csv", key=f"issues_{key}")
    else:
        notify("ok", "No problems were found in this file.")
    if valid == 0:
        notify("error", "There are no valid rows, so the prediction is switched off. Please correct the file and upload it again.")
    if st.button("Run prediction on valid rows", type="primary", key=f"run_file_{key}", disabled=valid == 0, icon=":material/play_arrow:"):
        try:
            with st.spinner("Checking the readings…"):
                result = pred.predict_frame(bundle, frame, "File upload", first_row_number=2, report=report, source_name=file.name)
            store_result(key, result, machine)
        except pred.PredictionError:
            notify("error", "The prediction could not be completed. Please check the file and try again.")


# ----------------------------------------------------------------------------
# Result display
# ----------------------------------------------------------------------------
def prefill_task(key: str) -> None:
    entry = st.session_state.results.get(key)
    if not entry:
        return
    res, machine = entry["result"], entry["machine"]
    desc = (f'Inspect {machine["name"]}: the model predicted a possible failure condition for {res.n_failure} of {res.n_records} '
            f'record(s) (prediction reference {entry["saved_id"] or "not saved"}). Please check the machine; this is a prediction, not a confirmed failure.')
    prefill_task_values(machine["machine_id"], machine["name"] if not str(machine["machine_id"]).startswith("UNREGISTERED") else "", desc)


def prefill_task_from_prediction(p: dict) -> None:
    desc = (f'Follow up on prediction {p["prediction_id"]} for {p["machine_name"]}: {p["n_failure"]} of {p["n_records"]} record(s) '
            f'were predicted as a possible failure condition. Review status: {p["review_status"]}. '
            "Please check the machine; this is a prediction, not a confirmed failure.")
    prefill_task_values(p["machine_id"], p["machine_name"] if str(p["machine_id"]).startswith("UNREGISTERED") else "", desc)


def prefill_task_values(machine_id: str, other_name: str, description: str) -> None:
    """Set the create-task form so it opens ready to review. Runs inside a button callback."""
    ss = st.session_state
    db = get_db()
    options = machine_options(db)
    label = next((lab for lab, m in options.items() if m["machine_id"] == machine_id), None)
    ss["task_machine"] = label or OTHER_MACHINE
    ss["task_other"] = "" if label else other_name
    ss["task_desc"] = description
    ss.page = "planner"
    ss.view_planner = "create"


def render_result(key: str, bundle: pred.ModelBundle) -> None:
    entry = st.session_state.results.get(key)
    if not entry:
        return
    res: pred.PredictionResult = entry["result"]
    machine = entry["machine"]
    st.markdown("---")
    st.markdown(f'<div class="mg-ph"><div><h1 style="font-size:1.35rem">Prediction results</h1>'
                f'<p>{esc(machine["name"])} · {esc(local_time(res.created_at))} · {esc(res.input_method)} · prediction method: {esc(res.algorithm)}</p></div></div>',
                unsafe_allow_html=True)
    spacer(.5)
    if res.error:
        notify("error", res.error)
        if len(res.issues):
            html_table(issue_table(res.issues), max_rows=cfg.ISSUE_PREVIEW_ROWS, height=260, flag_col="Level")
        st.button("Clear result", key=f"clear_{key}", on_click=clear_result, args=(key,))
        return

    probs = res.results["Failure probability"] if res.has_probability else None
    kpi_row([("Records checked", f"{res.n_records:,}", f"of {res.n_input_rows:,} supplied" + (f", {res.n_invalid} not usable" if res.n_invalid else ""), "", "database"),
             ("Possible failure conditions", f"{res.n_failure:,}", "predicted by the model, not confirmed", "bad" if res.n_failure else "", "alert"),
             (OK_LABEL, f"{res.n_non_failure:,}", "records", "ok" if res.n_non_failure else "", "check"),
             ("Highest failure likelihood", f"{probs.max() * 100:.1f}%" if probs is not None else "n/a",
              f"average {probs.mean() * 100:.1f}%" if probs is not None else "this model gives no likelihood score", "", "gauge")])
    spacer(.4)
    if res.n_failure:
        notify("warn", f"The model predicted a possible failure condition for {res.n_failure:,} of {res.n_records:,} record(s). "
                       "This is a prediction from sensor patterns, not a confirmed breakdown. Please have the machine checked. "
                       + MEANING[key])
    else:
        notify("ok", "The model predicted no failure for these records. This does not guarantee the machine is healthy, so keep up routine checks. "
                     + MEANING[key])
    if entry["saved_id"]:
        notify("info", f'Saved to Prediction History (reference {entry["saved_id"]}).')
    elif entry["save_error"]:
        notify("warn", "This result could not be saved to Prediction History.")
    if res.algorithm == "Decision Tree" and res.has_probability:
        st.caption("For this prediction method the likelihood score takes only a few distinct values and is a rough guide, not an exact chance of failure.")

    shown = friendly_results(res.results)
    section("Results by record", "Highlighted rows are possible failure conditions. Your readings and the calculated values are included.")
    html_table(shown, pct_cols=("Failure likelihood score",), flag_col="Model result", height=420)
    c1, c2 = st.columns(2)
    c1.download_button("Download results (CSV)", data=shown.to_csv(index=False),
                       file_name=f"{key}_results_{res.created_at:%Y%m%d_%H%M%S}.csv", mime="text/csv", key=f"dl_res_{key}")
    c2.button("Clear result", key=f"clear_{key}", on_click=clear_result, args=(key,))

    if len(res.issues):
        section("Checks and warnings", "Items found while checking your readings.")
        html_table(issue_table(res.issues), max_rows=cfg.ISSUE_PREVIEW_ROWS, height=260, flag_col="Level")
        st.download_button("Download the issue report (CSV)", data=issue_table(res.issues).to_csv(index=False),
                           file_name=f"{key}_issue_report.csv", mime="text/csv", key=f"dl_issue_{key}")

    section("Sensor parameters that influenced the model's prediction",
            "Shows which sensor parameters the model relies on most, next to this record's value and the usual range in the example data. "
            "It describes how the model works and does not prove why a failure would happen.")
    index = 0
    if len(res.results) > 1:
        flagged = [i for i in range(len(res.results)) if res.results.loc[i, "Predicted class"] == 1]
        ordered = (flagged + [i for i in range(len(res.results)) if i not in flagged])[:300]
        index = st.selectbox("Which record would you like to look at?", ordered, key=f"explain_{key}",
                             format_func=lambda i: f'Row {res.results.loc[i, "File row"]}: {FAIL_LABEL if res.results.loc[i, "Predicted class"] == 1 else OK_LABEL}')
    explanation = pred.explain_row(bundle, res.features.iloc[index])
    if explanation is None:
        notify("info", "This prediction method does not report which sensor parameters influenced it.")
    else:
        total = float(explanation["Importance"].sum()) or 1.0
        rows = []
        for _, item in explanation.iterrows():
            name, category = pretty_feature(item["Model feature"])
            value = item["This record"]
            if category is not None:
                value = "Yes" if str(value) == category else "No"
            usual = f'{_fmt(item["Reference min"])} to {_fmt(item["Reference max"])}' if item["Reference min"] is not None else "—"
            rows.append({"Sensor parameter": name, "Influence on the prediction": item["Importance"], "This record": value,
                         "Typical value": item["Reference median"], "Usual range": usual})
        table = pd.DataFrame(rows)
        table["Influence on the prediction"] = table["Influence on the prediction"] / total
        html_table(table, pct_cols=("Influence on the prediction",), height=320)

    section("What to do next")
    actions = []
    if res.n_failure:
        actions += ["Treat flagged records as an early warning that needs a human check, not as a confirmed failure.",
                    "Compare the flagged readings with the machine's alarms, maintenance history and normal operating values.",
                    "Ask a qualified technician to inspect the machine before taking action."]
    else:
        actions.append("No record was flagged. Keep routine monitoring and maintenance in place.")
    if res.n_invalid or len(res.issues):
        actions.append("Review the checks and warnings above. Rows that could not be used were not predicted.")
    if key == "compressor":
        actions.append("Make sure the previous reading you used was the true reading before this one.")
    for action in actions:
        st.markdown(f"- {action}")
    st.button("Create a maintenance task from this result", icon=":material/event_note:", key=f"task_{key}",
              on_click=prefill_task, args=(key,), type="primary" if res.n_failure else "secondary")


def about_panel(key: str, bundle: pred.ModelBundle) -> None:
    spec = cfg.MODELS[key]
    
    
    section("Readings you provide", "Enter these on the 'Enter readings' tab, or include them as columns in an uploaded file.")
    html_table(pd.DataFrame([{"Reading": cfg.input_meta(key, c)["label"], "Column name for files": c, "Unit": cfg.input_meta(key, c)["unit"]}
                             for c in bundle.input_columns]), height=360)
    section("Values calculated for you", "You never enter these. They are worked out from your readings.")
    html_table(pd.DataFrame([{"Value": pretty_feature(c)[0], "How it is calculated": CALC_TEXT.get(c, "")} for c in bundle.engineered_columns]), height=260)
    if key == "compressor":
        st.caption("Two values compare the current reading with the previous one, which is why a previous reading is requested.")


# ============================================================================
# PAGES
# ============================================================================
def page_workspace(key: str) -> None:
    spec = cfg.MODELS[key]
    bundle, _ = get_model(key)
    page_header(f'{spec["machine_name"]} Workspace',
                "Check readings from this type of machine and see whether the model predicts a possible failure condition.",
                "compressor" if key == "compressor" else "cog")
    if bundle is None:
        notify("error", PREDICTION_UNAVAILABLE)
        return
    help_box(["Pick the machine the readings come from (or choose one that is not in the registry).",
              "Enter readings by hand, or upload a CSV / Excel file with many readings.",
              "Run the prediction. Each record is marked as a possible failure condition or no failure predicted.",
              "A result is a prediction, not a confirmed failure. Use it to decide what to inspect, and create a maintenance task if needed.",
              "Every prediction is saved automatically to Prediction History."])
    
    machine, registered = pick_machine(key)
    if not registered:
        st.caption("These readings are not linked to a registered machine. Register it in the Machine Registry to keep its history together.")
    with st.spinner("Preparing…"):
        pred.stats_for(bundle)
    tab_manual, tab_upload, tab_info = st.tabs(["Enter readings", "Upload a file", "About this machine type"])
    with tab_manual:
        manual_panel(key, bundle, machine)
    with tab_upload:
        upload_panel(key, bundle, machine)
    with tab_info:
        about_panel(key, bundle)
    render_result(key, bundle)


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
def norm_status(status) -> str:
    s = str(status or "")
    if s in ("Completed", "Cancelled"):
        return "Completed"
    if s in ("In Progress", "In progress"):
        return "In Progress"
    return "Pending"


def status_label(status) -> str:
    return "Cancelled" if str(status) == "Cancelled" else norm_status(status)


def task_card(t: dict) -> str:
    done = norm_status(t.get("status")) == "Completed"
    overdue = (not done) and bool(t.get("scheduled_date")) and str(t["scheduled_date"]) < date.today().isoformat()
    cls = f'p-{str(t.get("priority", "")).lower()}' + (" done" if done else "")
    scheduled = (str(t["scheduled_date"]) + (f' {t["scheduled_time"]}' if t.get("scheduled_time") else "")) if t.get("scheduled_date") else "Not scheduled"
    return (f'<div class="mg-task {cls}"><div class="d">{esc(t["description"][:90] + ("…" if len(t["description"]) > 90 else ""))}</div>'
            f'<div class="mt">{esc(t.get("machine_name") or "Unknown machine")} · {esc(t.get("technician") or "No technician assigned")}'
            f'<br>Created {esc(local_time(t.get("created_at")))}</div>'
            f'<div class="ft">{pill(t.get("priority", ""))}{pill(status_label(t.get("status")))}{pill("Overdue") if overdue else ""}'
            f'<span class="dt">{esc(scheduled)}</span></div></div>')


def machine_card(key: str, tone: str, machines: list[dict]) -> None:
    spec, card = cfg.MODELS[key], DASH_CARDS[key]
    mine = [m for m in machines if m.get("assigned_model") == key]
    active = sum(1 for m in mine if m.get("active"))
    params = "".join(chip(p) for p in card["params"])
    st.markdown(
        f'<div class="mg-mcard {tone}"><div class="hd"><div class="ill">{svg("compressor", 52) if key == "compressor" else svg("cog", 34)}</div>'
        f'<div><h3>{esc(spec["machine_name"])}</h3><div class="tag">{len(mine)} registered · {active} active</div></div></div>'
        f'<p>{esc(card["description"])}</p><div class="lbl">Key parameters monitored</div><div>{params}</div>'
        f'<div class="m"><div>Registered machines<b>{len(mine)}</b></div><div>Active<b>{active}</b></div></div></div>', unsafe_allow_html=True)
    st.button("Open workspace", icon=":material/arrow_forward:", key=f"open_{key}", on_click=go_to, args=(key,), type="primary")


def page_dashboard() -> None:
    db = get_db()
    page_header("Industrial Machine Overview",
                "Monitor machine activity, review prediction results and manage maintenance operations from one place.", "dashboard")
    if not db.connected:
        notify("warn", DATA_UNAVAILABLE)
    machines = db_call(db.list_machines, default=[], quiet=True) if db.connected else []
    tasks = db_call(db.list_tasks, default=[], quiet=True) if db.connected else []
    counts = db_call(db.dashboard_counts, default=None, quiet=True) if db.connected else None
    preds = db_call(db.list_predictions, limit=6, default=[], quiet=True) if db.connected else []

    st.markdown(
        '<div class="mg-intro"><p>MachineGuard AI helps you monitor industrial machine sensor data, identify potential failure conditions '
        'using trained machine learning models and organise maintenance activities.</p><div class="mg-steps">'
        '<div class="mg-step"><b>1</b><span>Register the machines you look after.</span></div>'
        '<div class="mg-step"><b>2</b><span>Enter sensor readings or upload a file to run a prediction.</span></div>'
        '<div class="mg-step"><b>3</b><span>Review the results and plan maintenance tasks.</span></div></div></div>', unsafe_allow_html=True)

    pending = sum(1 for t in tasks if norm_status(t.get("status")) == "Pending")
    in_progress = sum(1 for t in tasks if norm_status(t.get("status")) == "In Progress")
    completed = sum(1 for t in tasks if norm_status(t.get("status")) == "Completed")
    kpi_row([("Registered machines", f"{len(machines):,}" if db.connected else "—", "machines you look after", "", "registry"),
             ("Prediction runs", f'{counts["predictions"]:,}' if counts else "—", "saved in Prediction History", "", "clock"),
             ("Records predicted as failure", f'{counts["failures"]:,}' if counts else "—", "possible failure conditions, not confirmed",
              "bad" if counts and counts["failures"] else "", "alert"),
             ("Pending maintenance tasks", f"{pending:,}" if db.connected else "—", "waiting to be started", "warn" if pending else "", "wrench")])

    section("Machine overview", "Choose a machine category to check readings or upload a file.")
    left, right = st.columns(2, gap="large")
    with left:
        machine_card("compressor", "", machines)
    with right:
        machine_card("ai4i", "sage", machines)

    a, b = st.columns(2, gap="large")
    with a:
        section("Recent prediction activity", "The latest checks that were run.")
        if preds:
            html_table(pd.DataFrame([{"Machine": p["machine_name"], "Date and time": local_time(p["created_at"]), "Records": p["n_records"],
                                      "Possible failures": p["n_failure"], "No failure predicted": p["n_non_failure"]} for p in preds]), height=300)
            st.button("View all", key="dash_viewall", on_click=go_to, args=("history",), icon=":material/history:")
        else:
            empty_state("activity", "No predictions yet", "When you run a prediction in a machine workspace, the latest results appear here.",
                        action=("Run your first prediction", go_to, ("compressor",), "dash_empty_pred"))
    with b:
        section("Maintenance overview", "Where your maintenance work stands.")
        kpi_row([("Pending", f"{pending:,}", "", "warn" if pending else "", "clock"), ("In progress", f"{in_progress:,}", "", "", "wrench"),
                 ("Completed", f"{completed:,}", "", "ok" if completed else "", "check")])
        spacer(.4)
        recent = sorted(tasks, key=lambda t: aware(t.get("created_at")), reverse=True)[:3]
        if recent:
            st.markdown("**Recently created tasks**")
            st.markdown("".join(task_card(t) for t in recent), unsafe_allow_html=True)
        else:
            notify("info", "No maintenance tasks have been created yet.")
        st.button("Add maintenance task", key="dash_addtask", on_click=go_view, args=("planner", "planner", "create"), icon=":material/add:")

    section("Quick actions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.popover("Run a prediction", icon=":material/play_arrow:"):
            st.button("Industrial Air Compressor", key="qa_run_compressor", on_click=go_to, args=("compressor",))
            st.button("General Industrial Machine", key="qa_run_ai4i", on_click=go_to, args=("ai4i",))
    with c2:
        st.button("Register a machine", key="qa_register", on_click=go_view, args=("registry", "registry", "add"), icon=":material/add_circle:")
    with c3:
        st.button("Create a maintenance task", key="qa_task", on_click=go_view, args=("planner", "planner", "create"), icon=":material/event_note:")
    with c4:
        st.button("View prediction history", key="qa_history", on_click=go_to, args=("history",), icon=":material/history:")


# ----------------------------------------------------------------------------
# Machine Registry
# ----------------------------------------------------------------------------
def support_text(m: dict) -> str:
    key = m.get("assigned_model")
    return f'{cfg.MODELS[key]["short_name"]} prediction' if key in cfg.MODELS else "Maintenance tracking only"


def category_choice(m: dict) -> str:
    return m.get("assigned_model") if m.get("assigned_model") in cfg.MODELS else "none"


def machine_activity(db: MachineGuardDB, machine_id: str) -> dict:
    runs = db_call(db.list_predictions, machine_id=machine_id, limit=1000, default=[], quiet=True)
    tasks = [t for t in db_call(db.list_tasks, machine_id=machine_id, default=[], quiet=True) if norm_status(t.get("status")) != "Completed"]
    upcoming = sorted(t["scheduled_date"] for t in tasks if t.get("scheduled_date"))
    return {"runs": runs, "open_tasks": len(tasks), "next_task": upcoming[0] if upcoming else None}


def add_machine_cb() -> None:
    ss, db = st.session_state, get_db()
    choice = ss.reg_cat
    spec = cfg.MODELS.get(choice)
    category = spec["machine_name"] if spec else (ss.reg_type.strip() or "Other machine")
    machine_id = ss.reg_id.strip()
    try:
        db.add_machine({"machine_id": machine_id, "name": ss.reg_name.strip(), "category": category, "manufacturer": ss.reg_mfr.strip(),
                        "model": ss.reg_model.strip(), "installation_date": ss.reg_date, "location": ss.reg_loc.strip(),
                        "notes": ss.reg_notes.strip(), "assigned_model": choice if spec else None})
        if ss.reg_status == "Inactive":
            db.set_machine_active(machine_id, False)
        flash("ok", f"Machine {machine_id} was registered." + ("" if spec else " It is tracked for maintenance only; predictions are not available for this machine type."))
        for k, v in (("reg_id", ""), ("reg_name", ""), ("reg_type", ""), ("reg_mfr", ""), ("reg_model", ""), ("reg_loc", ""), ("reg_notes", "")):
            ss[k] = v
        ss.reg_date = date.today()
        ss.view_registry = "overview"
    except DatabaseError as exc:
        flash("error", friendly_error(exc))


def save_machine_cb(machine_id: str) -> None:
    ss, db = st.session_state, get_db()
    choice = ss[f"edit_cat_{machine_id}"]
    spec = cfg.MODELS.get(choice)
    category = spec["machine_name"] if spec else (ss[f"edit_type_{machine_id}"].strip() or "Other machine")
    try:
        db.update_machine(machine_id, {"name": ss[f"edit_name_{machine_id}"].strip(), "category": category,
                                       "manufacturer": ss[f"edit_mfr_{machine_id}"].strip(), "model": ss[f"edit_model_{machine_id}"].strip(),
                                       "installation_date": ss[f"edit_date_{machine_id}"].isoformat(), "location": ss[f"edit_loc_{machine_id}"].strip(),
                                       "notes": ss[f"edit_notes_{machine_id}"].strip(), "assigned_model": choice if spec else None})
        db.set_machine_active(machine_id, ss[f"edit_status_{machine_id}"] == "Active")
        flash("ok", "Your changes were saved.")
    except DatabaseError as exc:
        flash("error", friendly_error(exc))


def toggle_machine_cb(machine_id: str, make_active: bool) -> None:
    try:
        get_db().set_machine_active(machine_id, make_active)
        flash("ok", "The machine is now active." if make_active else "The machine was deactivated. Its history is kept.")
    except DatabaseError as exc:
        flash("error", friendly_error(exc))


def category_notice(choice: str) -> None:
    if choice == "none":
        notify("warn", "Predictions are not currently supported for this kind of machine. It can still be registered and used for maintenance tasks.")
    else:
        notify("info", f'Readings from this machine will be checked with the {cfg.MODELS[choice]["machine_name"]} prediction model. '
                       "Registering a machine does not create or train a new model.")


def page_registry() -> None:
    db = get_db()
    page_header("Machine Registry", "Record the real machines your company looks after, such as Compressor-01 or Production-Machine-01.", "registry")
    if not db.connected:
        notify("warn", DATA_UNAVAILABLE)
        return
    machines = db_call(db.list_machines, default=[])
    help_box(["Each entry is one physical machine you want to manage.",
              "Choose its category. Air compressors and general industrial machines can be checked with a prediction model.",
              "Any other machine can still be registered and given maintenance tasks, but predictions are not available for it.",
              "Registering a machine never creates or trains a model."])
    active_n = sum(1 for m in machines if m.get("active"))
    kpi_row([("Registered machines", f"{len(machines):,}", "in total", "", "registry"),
             ("Active", f"{active_n:,}", "available for predictions and tasks", "ok" if active_n else "", "check"),
             ("Inactive", f"{len(machines) - active_n:,}", "kept for history", "warn" if len(machines) - active_n else "", "alert")])
    spacer(.7)
    view = view_switch("registry", [("overview", "Registered machines"), ("add", "Add a machine"), ("edit", "Edit or deactivate")], "overview")

    if view == "overview":
        if not machines:
            empty_state("registry", "No machines have been registered yet",
                        "The registry is empty because no machine has been added. Registered machines are used to link predictions and maintenance tasks to the right equipment.",
                        steps=["Add a machine with its ID, name and category.", "Open its workspace and run a prediction, or create a maintenance task.",
                               "Review results in Failure Intelligence and Prediction History."],
                        action=("Add your first machine", set_view, ("view_registry", "add"), "reg_first"))
            return
        f1, f2, f3 = st.columns([2, 1.3, 1])
        search = f1.text_input("Search", placeholder="Search by ID, name, location or manufacturer", key="reg_search")
        f_cat = f2.selectbox("Category", ["All", *CATEGORY_CHOICES], format_func=lambda k: "All categories" if k == "All" else CATEGORY_LABELS[k], key="reg_cat_filter")
        f_status = f3.selectbox("Status", ["All", "Active", "Inactive"], key="reg_status_filter")
        shown = [m for m in machines
                 if (f_cat == "All" or category_choice(m) == f_cat)
                 and (f_status == "All" or (m.get("active") and f_status == "Active") or (not m.get("active") and f_status == "Inactive"))
                 and (not search or search.lower() in " ".join(str(m.get(k, "")) for k in ("machine_id", "name", "location", "manufacturer", "model")).lower())]
        if not shown:
            empty_state("search", "No machines match your search", "Change the search text or filters to see more machines.")
            return
        html_table(pd.DataFrame([{
            "Machine ID": m["machine_id"], "Name": m["name"], "Category": m.get("category"), "Manufacturer": m.get("manufacturer"),
            "Model number": m.get("model"), "Installed": m.get("installation_date"), "Location": m.get("location"),
            "Prediction support": support_text(m), "Status": "Active" if m.get("active") else "Inactive"} for m in shown]), height=340)
        section("Machine details")
        choice = st.selectbox("Select a machine", [m["machine_id"] for m in shown], key="reg_detail",
                              format_func=lambda i: next(f'{m["name"]} ({i})' for m in shown if m["machine_id"] == i))
        m = next(x for x in shown if x["machine_id"] == choice)
        act = machine_activity(db, choice)
        last = act["runs"][0] if act["runs"] else None
        with st.container(key="card_reg_detail"):
            st.markdown(f'### {esc(m["name"])} &nbsp; {pill("Active" if m.get("active") else "Inactive")}', unsafe_allow_html=True)
            info_grid([("Machine ID", esc(m["machine_id"])), ("Category", esc(m.get("category") or "—")),
                       ("Manufacturer", esc(m.get("manufacturer") or "—")), ("Model number", esc(m.get("model") or "—")),
                       ("Installation date", esc(m.get("installation_date") or "—")), ("Location", esc(m.get("location") or "—")),
                       ("Prediction support", esc(support_text(m))), ("Registered on", esc(local_time(m.get("created_at"), "%d %b %Y")))])
            if m.get("notes"):
                st.markdown(f'**Notes**  \n{esc(m["notes"])}', unsafe_allow_html=True)
            if category_choice(m) == "none":
                notify("warn", "Predictions are not currently supported for this machine. It is tracked for maintenance only.")
            st.markdown("**Operating information**")
            info_grid([("Prediction runs", f'{len(act["runs"]):,}'),
                       ("Possible failure conditions found", f'{sum(r["n_failure"] for r in act["runs"]):,}'),
                       ("Last prediction", esc(local_time(last["created_at"])) if last else "None yet"),
                       ("Latest result", esc(f'{last["n_failure"]} of {last["n_records"]} flagged') if last else "—")])
            st.markdown("**Maintenance status**")
            info_grid([("Open tasks", f'{act["open_tasks"]:,}'), ("Next scheduled task", esc(act["next_task"] or "None scheduled"))])
            b1, b2 = st.columns(2)
            b1.button("Edit this machine", key="reg_to_edit", on_click=set_view, args=("view_registry", "edit"), icon=":material/edit:")
            b2.button("Deactivate machine" if m.get("active") else "Reactivate machine", key=f"reg_toggle_{choice}",
                      on_click=toggle_machine_cb, args=(choice, not m.get("active")), icon=":material/power_settings_new:")

    elif view == "add":
        section("Add a machine", "Fields marked * are required.")
        ss = st.session_state
        ss.setdefault("reg_id", ""); ss.setdefault("reg_name", ""); ss.setdefault("reg_type", ""); ss.setdefault("reg_mfr", "")
        ss.setdefault("reg_model", ""); ss.setdefault("reg_loc", ""); ss.setdefault("reg_notes", ""); ss.setdefault("reg_date", date.today())
        with st.container(key="card_reg_add"):
            c1, c2 = st.columns(2)
            c1.text_input("Machine ID *", placeholder="e.g. Compressor-01", key="reg_id")
            c2.text_input("Machine name *", placeholder="e.g. Workshop air compressor", key="reg_name")
            st.selectbox("Machine category *", CATEGORY_CHOICES, format_func=lambda k: CATEGORY_LABELS[k], key="reg_cat")
            category_notice(ss.reg_cat)
            if ss.reg_cat == "none":
                st.text_input("What kind of machine is it? (optional)", placeholder="e.g. Water pump", key="reg_type")
            c1, c2 = st.columns(2)
            c1.text_input("Manufacturer", placeholder="e.g. Atlas Copco", key="reg_mfr")
            c2.text_input("Model number", placeholder="e.g. GA 30", key="reg_model")
            c1.date_input("Installation date", key="reg_date")
            c2.text_input("Location", placeholder="e.g. Plant 1, Bay 3", key="reg_loc")
            st.text_area("Notes", placeholder="Anything useful to know about this machine", key="reg_notes")
            st.radio("Status", ["Active", "Inactive"], horizontal=True, key="reg_status")
            st.button("Register machine", type="primary", key="reg_submit", on_click=add_machine_cb, icon=":material/add:")

    else:
        if not machines:
            empty_state("registry", "There are no machines to edit", "Register a machine first.",
                        action=("Add your first machine", set_view, ("view_registry", "add"), "reg_edit_first"))
            return
        section("Edit or deactivate a machine", "Deactivated machines keep their history but are no longer offered in the workspaces or task forms.")
        mid = st.selectbox("Machine", [m["machine_id"] for m in machines], key="edit_machine_choice",
                           format_func=lambda i: next(f'{m["name"]} ({i})' for m in machines if m["machine_id"] == i))
        m = next(x for x in machines if x["machine_id"] == mid)
        ss = st.session_state
        try:
            installed_default = date.fromisoformat(str(m.get("installation_date"))[:10])
        except ValueError:
            installed_default = date.today()
        ss.setdefault(f"edit_name_{mid}", m["name"]); ss.setdefault(f"edit_cat_{mid}", category_choice(m))
        ss.setdefault(f"edit_type_{mid}", m.get("category") if category_choice(m) == "none" and m.get("category") != "Other machine" else "")
        ss.setdefault(f"edit_mfr_{mid}", m.get("manufacturer") or ""); ss.setdefault(f"edit_model_{mid}", m.get("model") or "")
        ss.setdefault(f"edit_date_{mid}", installed_default); ss.setdefault(f"edit_loc_{mid}", m.get("location") or "")
        ss.setdefault(f"edit_notes_{mid}", m.get("notes") or ""); ss.setdefault(f"edit_status_{mid}", "Active" if m.get("active") else "Inactive")
        with st.container(key="card_reg_edit"):
            c1, c2 = st.columns(2)
            c1.text_input("Machine name *", key=f"edit_name_{mid}")
            c2.text_input("Machine ID", value=mid, disabled=True, key=f"edit_idshow_{mid}")
            st.selectbox("Machine category *", CATEGORY_CHOICES, format_func=lambda k: CATEGORY_LABELS[k], key=f"edit_cat_{mid}")
            category_notice(ss[f"edit_cat_{mid}"])
            if ss[f"edit_cat_{mid}"] == "none":
                st.text_input("What kind of machine is it? (optional)", key=f"edit_type_{mid}")
            c1, c2 = st.columns(2)
            c1.text_input("Manufacturer", key=f"edit_mfr_{mid}")
            c2.text_input("Model number", key=f"edit_model_{mid}")
            c1.date_input("Installation date", key=f"edit_date_{mid}")
            c2.text_input("Location", key=f"edit_loc_{mid}")
            st.text_area("Notes", key=f"edit_notes_{mid}")
            st.radio("Status", ["Active", "Inactive"], horizontal=True, key=f"edit_status_{mid}")
            st.button("Save changes", type="primary", key=f"edit_save_{mid}", on_click=save_machine_cb, args=(mid,), icon=":material/save:")


# ----------------------------------------------------------------------------
# Failure Intelligence
# ----------------------------------------------------------------------------
def predictions_frame(preds: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(preds)
    if frame.empty:
        return frame
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True).dt.tz_convert(datetime.now().astimezone().tzinfo)
    return frame


def details_frame(details: list[dict]) -> pd.DataFrame:
    rows = []
    for d in details:
        row = {"Row in file": d.get("row_number"), "Model result": FAIL_LABEL if d.get("predicted_class") == 1 else OK_LABEL,
               "Failure likelihood score": d.get("failure_probability"), "Machine ID": d.get("machine_id")}
        row.update(d.get("inputs") or {})
        row.update(d.get("engineered") or {})
        rows.append(row)
    return pd.DataFrame(rows)


def date_filter(prefix: str, label: str = "Filter by date") -> tuple[date | None, date | None]:
    """Optional date filter: off by default (no date restriction)."""
    if not st.checkbox(label, key=f"{prefix}_date_on", help="Leave unticked for no date restriction."):
        return None, None
    a, b = st.columns(2)
    return a.date_input("From", value=date.today(), key=f"{prefix}_from"), b.date_input("To", value=date.today(), key=f"{prefix}_to")


def page_intelligence() -> None:
    db = get_db()
    page_header("Failure Intelligence", "Review what the models predicted, record what your inspections found and follow up with maintenance.", "alert")
    if not db.connected:
        notify("warn", DATA_UNAVAILABLE)
        return
    preds = db_call(db.list_predictions, limit=5000, default=[])
    if not preds:
        empty_state("alert", "There are no predictions to review yet",
                    "This page summarises predictions made in the machine workspaces. It is empty because no prediction has been run and saved yet.",
                    steps=["Open the Air Compressor or General Industrial Machine workspace.", "Enter readings or upload a file.",
                           "Click Run prediction. The result is saved automatically and appears here."],
                    action=("Open Air Compressor Workspace", go_to, ("compressor",), "intel_empty"))
        return
    help_box(["Each row is one prediction run. 'Possible failures' are records the model flagged, not confirmed failures.",
              "Open a run to see its records, then set a review status and add a note about what an inspection found.",
              "Once a run is reviewed you can create a maintenance task from it."])
    c1, c2, c3, c4, c5 = st.columns(5)
    machines = {p["machine_id"]: p["machine_name"] for p in preds}
    f_machine = c1.selectbox("Machine", ["All", *machines], format_func=lambda i: "All machines" if i == "All" else f"{machines[i]} ({i})", key="fi_machine")
    models = sorted({p["model_key"] for p in preds if p.get("model_key")})
    f_model = c2.selectbox("Machine type", ["All", *models], format_func=lambda k: "All types" if k == "All" else cfg.MODELS[k]["machine_name"], key="fi_model")
    f_pred = c3.selectbox("Result", ["All", "With possible failures", "No failures predicted"], key="fi_pred")
    f_review = c4.selectbox("Review status", ["All", *cfg.REVIEW_STATUSES], key="fi_review")
    with c5:
        d_from, d_to = date_filter("fi")
    shown = [p for p in preds
             if (f_machine == "All" or p["machine_id"] == f_machine) and (f_model == "All" or p.get("model_key") == f_model)
             and (f_pred == "All" or (p["n_failure"] > 0) == (f_pred == "With possible failures"))
             and (f_review == "All" or p["review_status"] == f_review)
             and (d_from is None or d_from <= p["created_at"].astimezone().date() <= d_to)]
    kpi_row([("Prediction runs", f"{len(shown):,}", "matching the filters", "", "clock"),
             ("Possible failure records", f'{sum(p["n_failure"] for p in shown):,}', "predicted, not confirmed", "bad" if any(p["n_failure"] for p in shown) else "", "alert"),
             ("No failure predicted", f'{sum(p["n_non_failure"] for p in shown):,}', "records", "ok", "check"),
             ("Waiting for review", f'{sum(1 for p in shown if p["review_status"] == cfg.REVIEW_STATUSES[0]):,}', "runs", "warn" if shown else "", "info")])
    spacer(.5)
    if not shown:
        empty_state("search", "No predictions match the filters", "Change or clear the filters to see more prediction runs.")
        return

    frame = predictions_frame(shown)
    summary = (frame.groupby(["machine_id", "machine_name", "algorithm"], as_index=False)
               .agg(Runs=("prediction_id", "count"), Records=("n_records", "sum"), Failures=("n_failure", "sum"), NonFailures=("n_non_failure", "sum"),
                    Last_run=("created_at", "max")))
    summary["Possible failure rate"] = np.where(summary["Records"] > 0, summary["Failures"] / summary["Records"], 0.0)
    summary["Last_run"] = summary["Last_run"].dt.strftime("%d %b %Y %H:%M")
    summary = summary.rename(columns={"machine_id": "Machine ID", "machine_name": "Machine", "algorithm": "Prediction method", "Failures": "Possible failures",
                                      "NonFailures": "No failure predicted", "Last_run": "Last run"})
    section("Summary by machine")
    html_table(summary, pct_cols=("Possible failure rate",), height=260)

    section("Prediction runs")
    html_table(pd.DataFrame([{"Reference": p["prediction_id"], "Date and time": local_time(p["created_at"]), "Machine": p["machine_name"],
                              "Records": p["n_records"], "Possible failures": p["n_failure"], "No failure predicted": p["n_non_failure"],
                              "Review status": p["review_status"]} for p in shown]), height=320)

    section("Open a prediction run")
    labels = {p["prediction_id"]: f'{local_time(p["created_at"], "%d %b %H:%M")} | {p["machine_name"]} | {p["n_failure"]} possible failure(s)' for p in shown}
    pid = st.selectbox("Prediction run", list(labels), format_func=lambda i: labels[i], key="fi_pid")
    p = next(x for x in shown if x["prediction_id"] == pid)
    kpi_row([
        ("Machine", p["machine_name"], f'ID: {p["machine_id"]}', "", "registry", True),

        ("ML algorithm used", p["algorithm"], p["input_method"], "", "cog", True),

        ("Records checked", f'{p["n_records"]:,}',
        f'{p["n_invalid"]} not usable' if p["n_invalid"] else "",
        "", "database"),

        ("Possible failures", f'{p["n_failure"]:,}',
        "predicted, not confirmed",
        "bad" if p["n_failure"] else "", "alert")
    ])

    spacer(.5)
    left, right = st.columns([3, 2], gap="large")
    with left:
        view = st.radio("Show", ["Possible failures", "No failure predicted", "All saved records"], horizontal=True, key=f"fi_rows_{pid}")
        cls = {"Possible failures": 1, "No failure predicted": 0}.get(view)
        details = db_call(db.get_details, pid, cls, 2000, default=[])
        if p.get("n_records", 0) > p.get("stored_detail_rows", 0):
            st.caption(f'Individual records are saved for {p.get("stored_detail_rows", 0):,} of {p["n_records"]:,} (possible failures are saved first).')
        html_table(details_frame(details).drop(columns=["Machine ID"], errors="ignore"), pct_cols=("Failure likelihood score",), flag_col="Model result", height=360)
    with right:
        with st.form(f"form_review_{pid}"):
            st.markdown("**Record your review**")
            status = st.selectbox("Review status", cfg.REVIEW_STATUSES,
                                  index=cfg.REVIEW_STATUSES.index(p["review_status"]) if p["review_status"] in cfg.REVIEW_STATUSES else 0)
            note = st.text_area("Review note", value=p.get("review_note") or "", placeholder="What did the inspection find?")
            if st.form_submit_button("Save review", type="primary"):
                if db_call(db.update_review, pid, status, note, default=False) is not False:
                    notify("ok", "Your review was saved. It will show in the table the next time this page loads.")
        reviewed = p["review_status"] != cfg.REVIEW_STATUSES[0]
        st.button("Create a maintenance task", icon=":material/event_note:", key=f"fi_task_{pid}", on_click=prefill_task_from_prediction, args=(p,), disabled=not reviewed)
        if not reviewed:
            st.caption("Available after you set a review status other than 'Pending review'.")


# ----------------------------------------------------------------------------
# Risk & Analytics
# ----------------------------------------------------------------------------
def page_analytics() -> None:
    db = get_db()
    page_header("Risk & Analytics", "See how predictions are spread across your machines and how readings change over time.", "chart")
    if not db.connected:
        notify("warn", DATA_UNAVAILABLE)
    preds = db_call(db.list_predictions, limit=5000, default=[], quiet=True) if db.connected else []
    help_box(["The charts use predictions you have run and saved. Nothing is estimated or made up.",
              "'Possible failure condition' means the model flagged a record. It is not a confirmed failure.",
              "These charts do not estimate remaining life or when a failure might happen."])
    c1, c2 = st.columns(2)
    machines = {p["machine_id"]: p["machine_name"] for p in preds}
    f_machine = c1.selectbox("Machine", ["All", *machines], format_func=lambda i: "All machines" if i == "All" else f"{machines[i]} ({i})", key="an_machine")
    models = sorted({p["model_key"] for p in preds if p.get("model_key")})
    f_model = c2.selectbox("Machine type", ["All", *models], format_func=lambda k: "All types" if k == "All" else cfg.MODELS[k]["machine_name"], key="an_model")
    shown = [p for p in preds if (f_machine == "All" or p["machine_id"] == f_machine) and (f_model == "All" or p.get("model_key") == f_model)]
    frame = predictions_frame(shown)
    if not frame.empty:
        total = int(frame["n_records"].sum())
        kpi_row([("Prediction runs", f"{len(frame):,}", "", "", "clock"), ("Records checked", f"{total:,}", "", "", "database"),
                 ("Possible failure conditions", f'{int(frame["n_failure"].sum()):,}', "predicted, not confirmed", "bad" if frame["n_failure"].sum() else "", "alert"),
                 ("Share flagged", f'{frame["n_failure"].sum() / total * 100:.1f}%' if total else "—", "of records checked", "", "gauge")])
        spacer(.6)
    tabs = st.tabs(["Prediction outcomes", "Machine comparison", "Sensor trends", "Activity over time", "Influential sensor parameters"])

    def no_data(tab_key: str) -> None:
        empty_state("chart", "No analytics yet", "The charts use predictions saved from the machine workspaces. None match the current selection.",
                    steps=["Run a prediction in a machine workspace.", "Come back here to see the charts."],
                    action=("Open Air Compressor Workspace", go_to, ("compressor",), f"an_empty_{tab_key}"))

    with tabs[0]:
        if frame.empty:
            no_data("outcomes")
        else:
            failures, non = int(frame["n_failure"].sum()), int(frame["n_non_failure"].sum())
            a, b = st.columns(2)
            donut = go.Figure(go.Pie(labels=[FAIL_LABEL, OK_LABEL], values=[failures, non], hole=.6, sort=False,
                                     marker=dict(colors=[CLASS_COLORS[FAIL_LABEL], CLASS_COLORS[OK_LABEL]]), textfont=dict(color="#F2F0E9")))
            donut.update_layout(title="Share of records by model result")
            a.plotly_chart(style_fig(donut))
            by_type = frame.groupby("machine_name", as_index=False)[["n_failure", "n_non_failure"]].sum()
            bar = go.Figure([go.Bar(name=FAIL_LABEL, x=by_type["machine_name"], y=by_type["n_failure"], marker_color=CLASS_COLORS[FAIL_LABEL]),
                             go.Bar(name=OK_LABEL, x=by_type["machine_name"], y=by_type["n_non_failure"], marker_color=CLASS_COLORS[OK_LABEL])])
            bar.update_layout(barmode="stack", title="Records by machine", yaxis_title="Records")
            b.plotly_chart(style_fig(bar))

    with tabs[1]:
        if frame.empty:
            no_data("compare")
        else:
            grouped = frame.groupby("machine_name", as_index=False)[["n_records", "n_failure"]].sum()
            grouped["rate"] = grouped["n_failure"] / grouped["n_records"].where(grouped["n_records"] > 0) * 100
            a, b = st.columns(2)
            rate = go.Figure(go.Bar(x=grouped["machine_name"], y=grouped["rate"], marker_color="#E58A32"))
            rate.update_layout(title="Share of records flagged, by machine", yaxis_title="Records flagged (%)")
            a.plotly_chart(style_fig(rate))
            counts = go.Figure([go.Bar(name="Records checked", x=grouped["machine_name"], y=grouped["n_records"], marker_color="#78877C"),
                                go.Bar(name=FAIL_LABEL, x=grouped["machine_name"], y=grouped["n_failure"], marker_color=CLASS_COLORS[FAIL_LABEL])])
            counts.update_layout(barmode="group", title="Records checked and flagged", yaxis_title="Records")
            b.plotly_chart(style_fig(counts))

    with tabs[2]:
        model_key = st.selectbox("Machine type", list(cfg.MODELS), format_func=lambda k: cfg.MODELS[k]["machine_name"], key="trend_model")
        bundle, _ = get_model(model_key)
        details = db_call(db.get_details_by_model, model_key, 20000, default=[], quiet=True) if db.connected else []
        if f_machine != "All":
            details = [d for d in details if d.get("machine_id") == f_machine]
        if not details or bundle is None:
            empty_state("activity", "No saved readings yet", "Sensor trends are drawn from the readings saved with each prediction. None are saved for this selection.",
                        action=("Open Air Compressor Workspace", go_to, ("compressor",), "trend_empty"))
        else:
            data = details_frame(details)
            data.insert(0, "Sequence", range(1, len(data) + 1))
            candidates = [c for c in [*bundle.input_columns, *bundle.engineered_columns] if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]
            sensor = st.selectbox("Sensor or calculated value", candidates, format_func=lambda c: pretty_feature(c)[0], key="trend_sensor")
            t = THEMES[st.session_state.theme]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["Sequence"], y=data[sensor], mode="lines", name=pretty_feature(sensor)[0], line=dict(width=1, color=t["muted"])))
            for label, color in CLASS_COLORS.items():
                part = data[data["Model result"] == label]
                fig.add_trace(go.Scatter(x=part["Sequence"], y=part[sensor], mode="markers", name=label, marker=dict(color=color, size=6)))
            fig.update_layout(title=f"{pretty_feature(sensor)[0]} across saved records (oldest to newest)", xaxis_title="Saved record number", yaxis_title=pretty_feature(sensor)[0])
            st.plotly_chart(style_fig(fig, 420))
            st.caption("Records are shown in the order they were saved, which is not necessarily the order of machine operation.")

    with tabs[3]:
        if frame.empty:
            no_data("activity")
        else:
            runs = frame.sort_values("created_at").copy()
            runs["rate"] = runs["n_failure"] / runs["n_records"].where(runs["n_records"] > 0) * 100
            a, b = st.columns(2)
            line = go.Figure()
            for name, part in runs.groupby("machine_name"):
                line.add_trace(go.Scatter(x=part["created_at"], y=part["rate"], mode="lines+markers", name=name))
            line.update_layout(title="Share of records flagged in each run", yaxis_title="Records flagged (%)", xaxis_title="Run time")
            a.plotly_chart(style_fig(line))
            runs["Day"] = runs["created_at"].dt.date.astype(str)
            daily = runs.groupby("Day", as_index=False)[["n_failure", "n_non_failure"]].sum()
            stack = go.Figure([go.Bar(name=FAIL_LABEL, x=daily["Day"], y=daily["n_failure"], marker_color=CLASS_COLORS[FAIL_LABEL]),
                               go.Bar(name=OK_LABEL, x=daily["Day"], y=daily["n_non_failure"], marker_color=CLASS_COLORS[OK_LABEL])])
            stack.update_layout(barmode="stack", title="Records checked per day", yaxis_title="Records")
            b.plotly_chart(style_fig(stack))

    with tabs[4]:
        model_key = st.selectbox("Machine type", list(cfg.MODELS), format_func=lambda k: cfg.MODELS[k]["machine_name"], key="imp_model")
        bundle, _ = get_model(model_key)
        table = pred.feature_importance(bundle) if bundle else None
        if table is None:
            notify("info", "This prediction method does not report which sensor parameters influenced it.")
        else:
            top = table.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["Importance"] / table["Importance"].sum() * 100, y=[pretty_feature(f)[0] for f in top["Feature"]], orientation="h", marker_color="#E58A32"))
            fig.update_layout(title="Sensor parameters that influenced the model's predictions (top 15)", xaxis_title="Share of the model's overall reliance (%)")
            st.plotly_chart(style_fig(fig, 480))
            st.caption("Longer bars are parameters the model relies on more. This describes how the model works; it is not a cause of failure.")


# ----------------------------------------------------------------------------
# Maintenance Planner (create / read / update / delete)
# ----------------------------------------------------------------------------
def machine_options(db: MachineGuardDB) -> dict:
    """Label -> machine for every active registered machine (any category)."""
    if not db.connected:
        return {}
    machines = db_call(db.list_machines, True, default=[], quiet=True)
    return {f'{m["name"]} ({m["machine_id"]})': m for m in machines}


def resolve_machine(choice: str, other: str, options: dict) -> tuple[str | None, str]:
    if choice == OTHER_MACHINE or choice not in options:
        return None, other.strip()
    return options[choice]["machine_id"], options[choice]["name"]


def scheduled_parts(enabled: bool, d, t) -> tuple[str | None, str | None]:
    if not enabled:
        return None, None
    return d.isoformat(), t.strftime("%H:%M") if t else None


def create_task_cb() -> None:
    ss, db = st.session_state, get_db()
    options = machine_options(db)
    machine_id, machine_name = resolve_machine(ss.task_machine, ss.get("task_other", ""), options)
    if not machine_name:
        flash("error", "Please choose a machine or enter a machine name.")
        return
    if not ss.task_desc.strip():
        flash("error", "Please describe what needs to be done.")
        return
    sched_date, sched_time = scheduled_parts(ss.task_sched_on, ss.task_date, ss.task_time)
    try:
        db.add_task({"machine_id": machine_id, "machine_name": machine_name, "description": ss.task_desc.strip(), "technician": ss.task_tech.strip(),
                     "priority": ss.task_priority, "status": ss.task_status, "notes": ss.task_notes.strip(),
                     "scheduled_date": sched_date, "scheduled_time": sched_time})
    except DatabaseError as exc:
        flash("error", friendly_error(exc))
        return
    for k, v in (("task_desc", ""), ("task_tech", ""), ("task_notes", ""), ("task_other", ""), ("task_priority", "Medium"),
                 ("task_status", "Pending"), ("task_sched_on", False)):
        ss[k] = v
    ss.view_planner = "board"
    flash("ok", "The maintenance task was created. The date and time it was created were saved automatically.")


def save_task_cb(tid: str) -> None:
    ss, db = st.session_state, get_db()
    options = machine_options(db)
    machine_id, machine_name = resolve_machine(ss[f"te_machine_{tid}"], ss.get(f"te_other_{tid}", ""), options)
    if not machine_name:
        flash("error", "Please choose a machine or enter a machine name.")
        return
    if not ss[f"te_desc_{tid}"].strip():
        flash("error", "The description cannot be empty.")
        return
    sched_date, sched_time = scheduled_parts(ss[f"te_sched_{tid}"], ss[f"te_date_{tid}"], ss[f"te_time_{tid}"])
    try:
        db.update_task(tid, {"machine_id": machine_id, "machine_name": machine_name, "description": ss[f"te_desc_{tid}"].strip(),
                             "technician": ss[f"te_tech_{tid}"].strip(), "priority": ss[f"te_priority_{tid}"], "status": ss[f"te_status_{tid}"],
                             "notes": ss[f"te_notes_{tid}"].strip(), "scheduled_date": sched_date, "scheduled_time": sched_time})
        flash("ok", "The task was updated. Its last-updated time was saved automatically.")
    except DatabaseError as exc:
        flash("error", friendly_error(exc))


def change_task_status(tid: str, status: str) -> None:
    try:
        get_db().update_task_status(tid, status)
        flash("ok", f"The task is now marked {status}.")
    except DatabaseError as exc:
        flash("error", friendly_error(exc))


def ask_delete(tid: str) -> None:
    st.session_state.pending_delete = tid


def cancel_delete() -> None:
    st.session_state.pending_delete = None


def delete_task_cb(tid: str) -> None:
    try:
        get_db().delete_task(tid)
        flash("ok", "The task was deleted.")
    except DatabaseError as exc:
        flash("error", friendly_error(exc))
    st.session_state.pending_delete = None


def machine_fields(prefix: str, options: dict, current_choice: str, current_other: str) -> None:
    """Machine dropdown with an 'Other' option that reveals a text box."""
    ss = st.session_state
    labels = [*options.keys(), OTHER_MACHINE]
    key_machine = "task_machine" if prefix == "task_" else f"te_machine_{prefix}"
    key_other = "task_other" if prefix == "task_" else f"te_other_{prefix}"
    if ss.get(key_machine) not in labels:
        ss[key_machine] = current_choice if current_choice in labels else OTHER_MACHINE
    ss.setdefault(key_other, current_other)
    st.selectbox("Machine name *", labels, key=key_machine, help="Choose a registered machine, or pick 'Other' to type any machine name.")
    if ss[key_machine] == OTHER_MACHINE:
        st.text_input("Machine name (type it here) *", placeholder="e.g. Packing line pump", key=key_other)
        st.caption("Maintenance tasks do not need a prediction or a registered machine.")


def task_filters(all_tasks: list[dict]) -> list[dict]:
    names = sorted({t.get("machine_name") for t in all_tasks if t.get("machine_name")})
    f1, f2, f3, f4 = st.columns(4)
    f_machine = f1.selectbox("Machine", ["All", *names], format_func=lambda n: "All machines" if n == "All" else n, key="pl_machine")
    f_priority = f2.selectbox("Priority", ["All", *cfg.TASK_PRIORITIES], key="pl_priority")
    f_status = f3.selectbox("Status", ["All", *TASK_STATUS_OPTIONS], key="pl_status")
    with f4:
        on = st.checkbox("Filter by date", key="pl_date_on", help="Leave unticked for no date restriction.")
    d_from = d_to = None
    kind = "Created"
    if on:
        g1, g2, g3 = st.columns(3)
        kind = g1.selectbox("Date type", ["Created", "Scheduled"], key="pl_date_kind")
        d_from = g2.date_input("From", value=date.today(), key="pl_from")
        d_to = g3.date_input("To", value=date.today(), key="pl_to")

    def in_range(t: dict) -> bool:
        if d_from is None:
            return True
        if kind == "Created":
            day = aware(t.get("created_at")).astimezone().date()
        else:
            try:
                day = date.fromisoformat(str(t.get("scheduled_date"))[:10])
            except ValueError:
                return False
        return d_from <= day <= d_to

    return [t for t in all_tasks if (f_machine == "All" or t.get("machine_name") == f_machine) and (f_priority == "All" or t.get("priority") == f_priority)
            and (f_status == "All" or norm_status(t.get("status")) == f_status) and in_range(t)]


def page_planner() -> None:
    db = get_db()
    page_header("Maintenance Planner", "Create, track and complete maintenance work for any machine, with or without a prediction.", "wrench")
    if not db.connected:
        notify("warn", DATA_UNAVAILABLE)
        return
    help_box(["Create a task whenever something needs to be checked or repaired. A prediction is not required.",
              "Each task gets its created date and time automatically. You can add a scheduled date and time if you want one.",
              "Use the board to see tasks as Pending, In Progress or Completed, and open a task to edit it, change its status or delete it."])
    options = machine_options(db)
    view = view_switch("planner", [("board", "Task board"), ("table", "All tasks"), ("create", "Create a task")], "board")

    if view == "create":
        section("Create a maintenance task", "The creation date and time are recorded for you.")
        ss = st.session_state
        for k, v in (("task_desc", ""), ("task_tech", ""), ("task_notes", ""), ("task_priority", "Medium"), ("task_status", "Pending"),
                     ("task_sched_on", False), ("task_date", date.today()), ("task_time", time(9, 0))):
            ss.setdefault(k, v)
        with st.container(key="card_task_create"):
            machine_fields("task_", options, next(iter(options), OTHER_MACHINE), "")
            st.text_area("Maintenance description *", key="task_desc",
                         placeholder="What needs to be repaired, inspected or checked? e.g. Check abnormal compressor pressure.")
            c1, c2 = st.columns(2)
            c1.selectbox("Priority", cfg.TASK_PRIORITIES, key="task_priority")
            c2.selectbox("Status", TASK_STATUS_OPTIONS, key="task_status")
            st.text_input("Technician (optional)", placeholder="Who will do the work?", key="task_tech")
            st.text_area("Notes (optional)", key="task_notes")
            st.checkbox("Set a scheduled date and time", key="task_sched_on")
            if ss.task_sched_on:
                d1, d2 = st.columns(2)
                d1.date_input("Scheduled date", key="task_date")
                d2.time_input("Scheduled time", key="task_time")
            st.button("Create task", type="primary", key="task_submit", on_click=create_task_cb, icon=":material/add:")
        return

    all_tasks = db_call(db.list_tasks, default=[])
    if not all_tasks:
        empty_state("wrench", "No maintenance tasks have been created yet.",
                    "The planner is empty because no task has been added. You can create a task for any machine at any time.",
                    steps=["Choose a machine, or type a machine name.", "Describe what needs to be checked or repaired.", "Track it from Pending to Completed here."],
                    action=("Create Your First Task", set_view, ("view_planner", "create"), "plan_first"))
        return
    tasks = task_filters(all_tasks)
    pending = sum(1 for t in tasks if norm_status(t.get("status")) == "Pending")
    overdue = sum(1 for t in tasks if norm_status(t.get("status")) != "Completed" and t.get("scheduled_date") and str(t["scheduled_date"]) < date.today().isoformat())
    kpi_row([("Pending", f"{pending:,}", "waiting to be started", "warn" if pending else "", "clock"),
             ("In progress", f'{sum(1 for t in tasks if norm_status(t.get("status")) == "In Progress"):,}', "being worked on", "", "wrench"),
             ("Completed", f'{sum(1 for t in tasks if norm_status(t.get("status")) == "Completed"):,}', "finished", "ok", "check"),
             ("Overdue", f"{overdue:,}", "past their scheduled date", "bad" if overdue else "", "alert")])
    spacer(.6)
    if not tasks:
        empty_state("search", "No tasks match the filters", "Change the filters to see more tasks.")
        return

    if view == "board":
        for column, title in zip(st.columns(3, gap="medium"), TASK_STATUS_OPTIONS):
            group = [t for t in tasks if norm_status(t.get("status")) == title]
            with column:
                st.markdown(f'<div class="mg-col-h"><span>{title}</span><span class="n">{len(group)}</span></div>', unsafe_allow_html=True)
                st.markdown("".join(task_card(t) for t in group) or '<div style="color:var(--muted);font-size:.82rem">No tasks here</div>', unsafe_allow_html=True)
    else:
        html_table(pd.DataFrame([{
            "Task ID": t["task_id"], "Machine": t.get("machine_name"), "Description": (t["description"][:80] + "…") if len(t["description"]) > 80 else t["description"],
            "Priority": t.get("priority"), "Status": status_label(t.get("status")), "Created": local_time(t.get("created_at")),
            "Scheduled": (str(t["scheduled_date"]) + (f' {t["scheduled_time"]}' if t.get("scheduled_time") else "")) if t.get("scheduled_date") else "—",
            "Technician": t.get("technician") or "—", "Last updated": local_time(t.get("updated_at"))} for t in tasks]), height=380)

    section("Open a task", "Select a task to see its details, change its status, edit it or delete it.")
    tid = st.selectbox("Task", [t["task_id"] for t in tasks], key="pl_task",
                       format_func=lambda i: f'{i} - {next(t["description"][:55] for t in tasks if t["task_id"] == i)}')
    t = next(x for x in tasks if x["task_id"] == tid)
    scheduled = (str(t["scheduled_date"]) + (f' {t["scheduled_time"]}' if t.get("scheduled_time") else "")) if t.get("scheduled_date") else "Not scheduled"
    with st.container(key="card_task_detail"):
        info_grid([("Task ID", esc(t["task_id"])), ("Machine", esc(t.get("machine_name") or "—")), ("Technician", esc(t.get("technician") or "Not assigned")),
                   ("Priority", pill(t.get("priority", ""))), ("Status", pill(status_label(t.get("status")))), ("Scheduled", esc(scheduled)),
                   ("Created", esc(local_time(t.get("created_at")))), ("Last updated", esc(local_time(t.get("updated_at")))),
                   ("Completed", esc(local_time(t.get("completed_at")) if t.get("completed_at") else "—"))])
        st.markdown(f'**Description**  \n{esc(t["description"])}', unsafe_allow_html=True)
        if t.get("notes"):
            st.markdown(f'**Notes**  \n{esc(t["notes"])}', unsafe_allow_html=True)
        current = norm_status(t.get("status"))
        q1, q2, q3 = st.columns(3)
        q1.button("Move to Pending", key=f"pl_pending_{tid}", on_click=change_task_status, args=(tid, "Pending"), disabled=current == "Pending", icon=":material/pause:")
        q2.button("Move to In Progress", key=f"pl_start_{tid}", on_click=change_task_status, args=(tid, "In Progress"), disabled=current == "In Progress", icon=":material/play_arrow:")
        q3.button("Mark as Completed", key=f"pl_done_{tid}", on_click=change_task_status, args=(tid, "Completed"), disabled=current == "Completed", icon=":material/check:")

    with st.expander("Edit this task"):
        ss = st.session_state
        label = next((lab for lab, m in options.items() if m["machine_id"] == t.get("machine_id")), OTHER_MACHINE)
        ss.setdefault(f"te_desc_{tid}", t["description"]); ss.setdefault(f"te_priority_{tid}", t.get("priority") if t.get("priority") in cfg.TASK_PRIORITIES else "Medium")
        ss.setdefault(f"te_status_{tid}", current); ss.setdefault(f"te_tech_{tid}", t.get("technician") or ""); ss.setdefault(f"te_notes_{tid}", t.get("notes") or "")
        try:
            d_default = date.fromisoformat(str(t.get("scheduled_date"))[:10])
        except ValueError:
            d_default = date.today()
        try:
            t_default = time.fromisoformat(str(t.get("scheduled_time"))) if t.get("scheduled_time") else time(9, 0)
        except ValueError:
            t_default = time(9, 0)
        ss.setdefault(f"te_sched_{tid}", bool(t.get("scheduled_date"))); ss.setdefault(f"te_date_{tid}", d_default); ss.setdefault(f"te_time_{tid}", t_default)
        machine_fields(tid, options, label, t.get("machine_name") or "")
        st.text_area("Maintenance description *", key=f"te_desc_{tid}")
        c1, c2 = st.columns(2)
        c1.selectbox("Priority", cfg.TASK_PRIORITIES, key=f"te_priority_{tid}")
        c2.selectbox("Status", TASK_STATUS_OPTIONS, key=f"te_status_{tid}")
        st.text_input("Technician", key=f"te_tech_{tid}")
        st.text_area("Notes", key=f"te_notes_{tid}")
        st.checkbox("Set a scheduled date and time", key=f"te_sched_{tid}")
        if ss[f"te_sched_{tid}"]:
            d1, d2 = st.columns(2)
            d1.date_input("Scheduled date", key=f"te_date_{tid}")
            d2.time_input("Scheduled time", key=f"te_time_{tid}")
        st.button("Save changes", type="primary", key=f"te_save_{tid}", on_click=save_task_cb, args=(tid,), icon=":material/save:")

    with st.expander("Delete this task"):
        if st.session_state.get("pending_delete") == tid:
            notify("warn", "Are you sure you want to delete this task? This cannot be undone.")
            d1, d2 = st.columns(2)
            with d1:
                st.button("Yes, delete this task", key=f"dangerbtn_{tid}", on_click=delete_task_cb, args=(tid,), icon=":material/delete:")
            d2.button("No, keep it", key=f"del_cancel_{tid}", on_click=cancel_delete)
        else:
            st.button("Delete task", key=f"del_{tid}", on_click=ask_delete, args=(tid,), icon=":material/delete:")


# ----------------------------------------------------------------------------
# Prediction History
# ----------------------------------------------------------------------------
def build_report_xlsx(p: dict, details: pd.DataFrame) -> bytes:
    summary = pd.DataFrame({"Item": ["Reference", "Date and time", "Machine", "Machine ID", "Prediction method", "Input method", "Source file",
                                     "Records supplied", "Records checked", "Rows not usable", "Possible failure conditions", "No failure predicted",
                                     "Review status", "Review note", "Please note"],
                            "Value": [p["prediction_id"], local_time(p["created_at"]), p["machine_name"], p["machine_id"], p["algorithm"], p["input_method"],
                                      p.get("source_name") or "", p["n_input_rows"], p["n_records"], p["n_invalid"], p["n_failure"], p["n_non_failure"],
                                      p["review_status"], p.get("review_note") or "",
                                      "These are model predictions, not confirmed failures."]})
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        details.to_excel(writer, sheet_name="Saved records", index=False)
    return buffer.getvalue()


def reset_history_filters() -> None:
    for k, v in (("hist_machine", "All"), ("hist_model", "All"), ("hist_method", "All"), ("hist_date_on", False)):
        st.session_state[k] = v


def page_history() -> None:
    db = get_db()
    page_header("Prediction History", "Every prediction you have run, saved so you can look back at it or download a report.", "clock")
    if not db.connected:
        notify("warn", DATA_UNAVAILABLE)
        return
    everything = db_call(db.list_predictions, limit=5000, default=[])  # unfiltered: used to fill the dropdowns
    if not everything:
        empty_state("clock", "No prediction history yet",
                    "No predictions have been saved yet. Run a prediction from either machine workspace to create your first history record.",
                    steps=["Choose a workspace and enter readings, or upload a file.",
                           "Checking a file is not enough: click Run prediction (or Run prediction on valid rows).",
                           "The saved prediction appears here straight away."],
                    action=("Open Air Compressor Workspace", go_to, ("compressor",), "hist_empty_1"))
        return
    known = {p["machine_id"]: p["machine_name"] for p in everything}
    model_keys = sorted({p["model_key"] for p in everything if p.get("model_key")})
    methods = sorted({p["input_method"] for p in everything if p.get("input_method")})
    c1, c2, c3, c4 = st.columns(4)
    f_machine = c1.selectbox("Machine", ["All", *known], format_func=lambda i: "All machines" if i == "All" else f"{known[i]} ({i})", key="hist_machine")
    f_model = c2.selectbox("Machine type", ["All", *model_keys], format_func=lambda k: "All types" if k == "All" else cfg.MODELS[k]["machine_name"], key="hist_model")
    f_method = c3.selectbox("How readings were entered", ["All", *methods], format_func=lambda m: "All methods" if m == "All" else m, key="hist_method")
    with c4:
        d_from, d_to = date_filter("hist")
    start = datetime.combine(d_from, time.min).astimezone(timezone.utc) if d_from else None
    end = datetime.combine(d_to, time.max).astimezone(timezone.utc) if d_to else None
    preds = db_call(db.list_predictions, machine_id=None if f_machine == "All" else f_machine, model_key=None if f_model == "All" else f_model,
                    input_method=None if f_method == "All" else f_method, date_from=start, date_to=end, limit=2000, default=[])
    kpi_row([("Matching runs", f"{len(preds):,}", f"of {len(everything):,} saved", "", "clock"),
             ("Records checked", f'{sum(p["n_records"] for p in preds):,}', "", "", "database"),
             ("Possible failure conditions", f'{sum(p["n_failure"] for p in preds):,}', "predicted, not confirmed", "bad" if any(p["n_failure"] for p in preds) else "", "alert"),
             (OK_LABEL, f'{sum(p["n_non_failure"] for p in preds):,}', "records", "ok", "check")])
    spacer(.5)
    if not preds:
        empty_state("search", "No predictions match the current filters",
                    "There are saved predictions, but none match the selected machine, type, method or dates.",
                    action=("Reset filters", reset_history_filters, (), "hist_reset"))
        return
    html_table(pd.DataFrame([{
        "Reference": p["prediction_id"], "Date and time": local_time(p["created_at"]), "Machine": f'{p["machine_name"]} ({p["machine_id"]})',
        "Prediction method": p["algorithm"], "Entered by": p["input_method"], "Records": p["n_records"], "Possible failures": p["n_failure"],
        "No failure predicted": p["n_non_failure"], "Review status": p["review_status"]} for p in preds]), height=340)
    section(
        "Prediction details",
        "Select a saved prediction to view its results, machine information and review status."
    )

    pid = st.selectbox(
        "Select a prediction record",
        [p["prediction_id"] for p in preds],
        key="hist_pid",
        format_func=lambda i: next(
            f'{local_time(p["created_at"], "%d %b %Y, %H:%M")} | '
            f'{p["machine_name"]} | '
            f'{p["n_records"]} records | '
            f'{p["n_failure"]} possible failures'
            for p in preds if p["prediction_id"] == i
        )
    )
    p = next(x for x in preds if x["prediction_id"] == pid)
    kpi_row([("Machine", p["machine_name"], f'ID: {p["machine_id"]}', "", "registry", True), ("Prediction method"("ML algorithm used", p["algorithm"], p["input_method"], "", "cog", True), p["algorithm"], p["input_method"], "", "cog", True),
             ("Records checked", f'{p["n_records"]:,}', f'{p["n_invalid"]} not usable' if p["n_invalid"] else "", "", "database"),
             ("Possible failures", f'{p["n_failure"]:,}', "predicted, not confirmed", "bad" if p["n_failure"] else "", "alert")])
    spacer(.5)
    details = db_call(db.get_details, pid, None, 5000, default=[])
    frame = details_frame(details).drop(columns=["Machine ID"], errors="ignore")
    if p.get("n_records", 0) > p.get("stored_detail_rows", 0):
        st.caption(f'Individual records saved: {p.get("stored_detail_rows", 0):,} of {p["n_records"]:,} (possible failures are saved first).')
    html_table(frame, pct_cols=("Failure likelihood score",), flag_col="Model result", height=360)
    d1, d2 = st.columns(2)
    d1.download_button("Download report (Excel)", data=build_report_xlsx(p, frame), file_name=f"{pid}_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xlsx_{pid}")
    d2.download_button("Download saved records (CSV)", data=frame.to_csv(index=False), file_name=f"{pid}_records.csv", mime="text/csv", key=f"csv_{pid}")


# ============================================================================
# MAIN
# ============================================================================
PAGE_FUNCTIONS = {
    "dashboard": page_dashboard,
    "registry": page_registry,
    "compressor": lambda: page_workspace("compressor"),
    "ai4i": lambda: page_workspace("ai4i"),
    "intelligence": page_intelligence,
    "analytics": page_analytics,
    "planner": page_planner,
    "history": page_history,
}


def main() -> None:
    init_state()
    if "booted" not in st.session_state:
        with st.spinner("Getting things ready…"):
            get_db()
            for model_key in cfg.MODELS:
                get_model(model_key)
        st.session_state.booted = True
    maybe_recover()
    inject_css()
    render_sidebar()
    PAGE_FUNCTIONS[st.session_state.page]()


main()