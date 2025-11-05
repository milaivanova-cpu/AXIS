import streamlit as st
import yaml, regex as re, json
from typing import Dict, List

st.set_page_config(page_title="AXIS+Construct Assessor", layout="wide")
st.title("🧪 AXIS-style Appraisal + Construct Addendum")

# ---------------- PDF text extraction ----------------
def extract_text(file) -> str:
    # Prefer PyMuPDF for cleaner text; fallback to pypdf
    try:
        import fitz
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([p.get_text("text") for p in doc])
        return text
    except Exception:
        file.seek(0)
        from pypdf import PdfReader
        r = PdfReader(file)
        chunks = []
        for p in r.pages:
            try:
                t = p.extract_text()
                if t: chunks.append(t)
            except: pass
        return "\n".join(chunks)

# ---------------- utilities ----------------
SPLIT = re.compile(r'(?<=\.)\s+(?=[A-Z\(])|(?<=[!?])\s+')
SECTION_HEAD = re.compile(r'\b(Abstract|Introduction|Background|Theory|Method|Methods|Measures?|Participants?|Procedure|Results?|Discussion|Conclusion|Funding|Acknowledgements|Ethics)\b', re.I)

def sentences(text: str) -> List[str]:
    clean = re.sub(r'\s+', ' ', text)
    return [s.strip() for s in SPLIT.split(clean) if s.strip()]

def sectionize(text: str) -> Dict[str,str]:
    secs = {}
    current = "Full Text"
    secs[current] = []
    for line in text.splitlines():
        m = SECTION_HEAD.search(line.strip())
        if m:
            current = m.group(0).title()
            secs.setdefault(current, [])
        secs[current].append(line)
    return {k: "\n".join(v) for k,v in secs.items()}

# ---------------- load configuration ----------------
@st.cache_data
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

AXIS = load_yaml("axis.yaml")
KB   = load_yaml("constructs.yaml")

# ---------------- pattern banks ----------------
RE_DEF      = re.compile(r'\b(defined as|we define|is defined as|refers to)\b', re.I)
RE_BOUNDARY = re.compile(r'\b(distinct from|differs from|as opposed to|not merely|boundary|scope conditions?)\b', re.I)
RE_THEORY   = re.compile(r'\b(model|mechanism|dual[\s-]?systems?|process model|expected value of control|valuation|control theory)\b', re.I)
RE_VALIDITY = re.compile(r'\b(convergent|discriminant|criterion|predictive|known[- ]groups|response[- ]process|factor validity)\b', re.I)
RE_RELIAB   = re.compile(r'\b(alpha|cronbach|omega|test[- ]?retest|ICC)\b', re.I)
RE_DESIGN   = re.compile(r'\b(randomi[sz]ed|experiment|intervention|longitudinal|cross[- ]sectional|pre[- ]post|RCT)\b', re.I)
RE_FUND     = re.compile(r'\bfund(ing|ed)|grant|sponsor|conflict of interest|COI\b', re.I)
RE_ETHICS   = re.compile(r'\b(IRB|ethic(al)? committee|approved|consent)\b', re.I)

def find_hits(blob, pattern, maxn=6):
    sents = sentences(blob)
    out = [s for s in sents if pattern.search(s)]
    return out[:maxn]

def detect_construct_labels(blob):
    found = {}
    for cname, cnode in KB["constructs"].items():
        labels = cnode["canonical"] + cnode["neighbors"]
        for lbl in labels:
            if re.search(rf'\b{re.escape(lbl)}\b', blob, re.I):
                found.setdefault(cname, set()).add(lbl)
    return {k: sorted(list(v)) for k,v in found.items()}

def detect_measures(blob):
    hits = []
    for mid, mnode in KB["measures"].items():
        for alias in mnode["aliases"]:
            if re.search(rf'\b{re.escape(alias)}\b', blob, re.I):
                hits.append({"measure": mid, "alias": alias, "type": mnode["type"], "targets": mnode["targets"]})
                break
    return hits

def map_targets(measures):
    buckets = {}
    for m in measures:
        for t in m["targets"]:
            buckets.setdefault(t, set()).add(m["measure"])
    return {k: sorted(list(v)) for k,v in buckets.items()}

# ---------------- heuristics per item ----------------
def propose_axis_score(item, sections) -> Dict:
    # crude, transparent heuristics that you can override
    sec_text = " ".join([sections.get(h,"") for h in item.get("section_hint",[])])
    label = item["label"]

    if "aims" in label.lower():
        hits = find_hits(sec_text or sections.get("Full Text",""), re.compile(r'\b(aim|objective|we (aim|seek)|research question)\b', re.I))
        score = "Yes" if hits else "Unclear"
        return {"proposed": score, "evidence": hits}

    if "design appropriate" in label.lower():
        hits = find_hits(sec_text, RE_DESIGN)
        score = "Yes" if hits else "Unclear"
        return {"proposed": score, "evidence": hits}

    if "sampling frame" in label.lower():
        hits = find_hits(sec_text, re.compile(r'\b(sample|participants?|recruit|eligibility|inclusion|exclusion)\b', re.I))
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "validity" in label.lower():
        hits = find_hits(sec_text, RE_VALIDITY)
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "reliability" in label.lower():
        hits = find_hits(sec_text, RE_RELIAB)
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "confounding" in label.lower():
        hits = find_hits(sec_text, re.compile(r'\b(confound|control(?:s|led)?|covariate|adjust(ed|ment))\b', re.I))
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "statistical methods" in label.lower():
        hits = find_hits(sec_text, re.compile(r'\b(regression|ANOVA|mixed[- ]effects|model|estimat|hypothesis test|assumption)\b', re.I))
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "precision" in label.lower():
        hits = find_hits(sec_text, re.compile(r'\b(effect size|confidence interval|CI|standard error|p\s*<)\b', re.I))
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "conclusions justified" in label.lower():
        hits = find_hits(sec_text or sections.get("Discussion",""), re.compile(r'\b(limit|caution|consistent with|cannot infer causality)\b', re.I))
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "funding" in label.lower():
        hits = find_hits(sections.get("Funding","") + " " + sections.get("Acknowledgements",""), RE_FUND)
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "ethical" in label.lower():
        hits = find_hits(sections.get("Ethics","") + " " + sections.get("Method",""), RE_ETHICS)
        return {"proposed": "Yes" if hits else "N/A", "evidence": hits}

    # defaults
    hits = find_hits(sec_text or sections.get("Full Text",""), re.compile(r'.'))
    return {"proposed": "Unclear", "evidence": hits[:3]}

def propose_construct_score(citem, sections, full):
    hint = " ".join([sections.get(h,"") for h in citem.get("section_hint",[])]) or full

    if "definition" in citem["label"].lower():
        hits = find_hits(hint, RE_DEF)
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "subcomponents" in citem["label"].lower():
        # look for neighbor/subcomponent words
        subs = []
        for cname,cnode in KB["constructs"].items():
            subs += cnode["subcomponents"]
        pat = re.compile("|".join([re.escape(s) for s in subs]), re.I) if subs else re.compile(r'')
        hits = find_hits(hint, pat)
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "model/theory" in citem["label"].lower():
        hits = find_hits(hint, RE_THEORY)
        return {"proposed": "Yes" if hits else "Unclear", "evidence": hits}

    if "measures align" in citem["label"].lower():
        measures = detect_measures(full)
        constructs = detect_construct_labels(full)
        hits = [f"{m['alias']} → {', '.join(m['targets'])}" for m in measures]
        score = "Yes" if measures else "No"
        # crude jingle guard: SC + Grit
        ops = {m["measure"] for m in measures}
        warn = []
        if "self-control" in constructs and "GritS" in ops:
            warn.append("Jingle risk: SC label with Grit-S measure.")
        return {"proposed": score, "evidence": hits, "warnings": warn}

    if "jingle–jangle" in citem["label"].lower():
        constructs = detect_construct_labels(full)
        hits = [f"{k}: {v}" for k,v in constructs.items()]
        # need boundary language
        boundary_hit = bool(RE_BOUNDARY.search(full))
        score = "Yes" if boundary_hit else ("Unclear" if constructs else "N/A")
        return {"proposed": score, "evidence": hits}

    if "evidence type supports" in citem["label"].lower():
        has_valid = bool(find_hits(full, RE_VALIDITY))
        has_fit   = bool(find_hits(full, re.compile(r'\b(CFI|TLI|RMSEA|SRMR|CFA|factor)\b', re.I)))
        score = "Yes" if (has_valid or has_fit) else "Unclear"
        ev = []
        ev += find_hits(full, RE_VALIDITY, 3)
        ev += find_hits(full, re.compile(r'\b(CFI|TLI|RMSEA|SRMR|CFA|factor)\b', re.I), 3)
        return {"proposed": score, "evidence": ev}

    return {"proposed": "Unclear", "evidence": []}

# ---------------- UI ----------------
uploaded = st.file_uploader("📄 Upload a PDF article", type=["pdf"])

if uploaded:
    with st.spinner("Parsing & scanning…"):
        blob = extract_text(uploaded)
        secs = sectionize(blob)

        # AXIS proposal
        axis_proposals = []
        for item in AXIS["axis_items"]:
            axis_proposals.append({
                "id": item["id"],
                "label": item["label"],
                "guidance": item["guidance"],
                "proposal": propose_axis_score(item, secs)
            })

        # Construct addendum proposal
        add_proposals = []
        for citem in AXIS["construct_addendum"]:
            add_proposals.append({
                "id": citem["id"],
                "label": citem["label"],
                "guidance": citem["guidance"],
                "proposal": propose_construct_score(citem, secs, blob)
            })

    st.success("✅ Draft appraisal ready")

    # Tabs
    t1, t2, t3 = st.tabs(["AXIS checklist", "Construct addendum (SC/SRL)", "Export"])

    with t1:
        st.caption("Click each item to review the proposed score and evidence, then confirm or override.")
        axis_scores = {}
        for row in axis_proposals:
            with st.expander(f"[{row['id']}] {row['label']}"):
                st.write("**Guidance:**", row["guidance"])
                st.write("**Proposed:**", row["proposal"]["proposed"])
                st.write("**Evidence:**")
                for s in row["proposal"]["evidence"]:
                    st.write("•", s)
                axis_scores[row["id"]] = st.selectbox(
                    "Your score",
                    ["Yes","No","Unclear","N/A"],
                    index=["Yes","No","Unclear","N/A"].index(row["proposal"]["proposed"]) if row["proposal"]["proposed"] in ["Yes","No","Unclear","N/A"] else 2,
                    key=f"axis_{row['id']}"
                )
                st.text_area("Rationale/comments", key=f"axis_c_{row['id']}", placeholder="Why this score? Cite lines, note limitations…")

    with t2:
        add_scores = {}
        for row in add_proposals:
            with st.expander(f"[{row['id']}] {row['label']}"):
                st.write("**Guidance:**", row["guidance"])
                st.write("**Proposed:**", row["proposal"]["proposed"])
                if "warnings" in row["proposal"] and row["proposal"]["warnings"]:
                    for w in row["proposal"]["warnings"]:
                        st.warning(w)
                st.write("**Evidence:**")
                for s in row["proposal"]["evidence"]:
                    st.write("•", s)
                add_scores[row["id"]] = st.selectbox(
                    "Your score",
                    ["Yes","No","Unclear","N/A"],
                    index=["Yes","No","Unclear","N/A"].index(row["proposal"]["proposed"]) if row["proposal"]["proposed"] in ["Yes","No","Unclear","N/A"] else 2,
                    key=f"add_{row['id']}"
                )
                st.text_area("Rationale/comments", key=f"add_c_{row['id']}", placeholder="Why this score? Evidence?")

    with t3:
        if st.button("Build exportable report"):
            export = {
                "axis": [
                    {
                        "id": r["id"],
                        "label": r["label"],
                        "proposed": r["proposal"]["proposed"],
                        "final": st.session_state.get(f"axis_{r['id']}"),
                        "comments": st.session_state.get(f"axis_c_{r['id']}", ""),
                        "evidence": r["proposal"]["evidence"]
                    } for r in axis_proposals
                ],
                "construct_addendum": [
                    {
                        "id": r["id"],
                        "label": r["label"],
                        "proposed": r["proposal"]["proposed"],
                        "final": st.session_state.get(f"add_{r['id']}"),
                        "comments": st.session_state.get(f"add_c_{r['id']}", ""),
                        "evidence": r["proposal"]["evidence"],
                        "warnings": r["proposal"].get("warnings", [])
                    } for r in add_proposals
                ]
            }
            st.subheader("Report (JSON)")
            st.json(export)
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(export, indent=2).encode("utf-8"),
                file_name="axis_construct_appraisal.json",
                mime="application/json"
            )
            # Lightweight Markdown export
            md_lines = ["# AXIS Appraisal + Construct Addendum\n"]
            md_lines.append("## AXIS checklist")
            for r in export["axis"]:
                md_lines.append(f"- **[{r['id']}] {r['label']}** — Final: {r['final']} (Proposed: {r['proposed']})")
                if r["comments"]: md_lines.append(f"  - _Comments:_ {r['comments']}")
            md_lines.append("\n## Construct addendum (Self-Control / Self-Regulation)")
            for r in export["construct_addendum"]:
                md_lines.append(f"- **[{r['id']}] {r['label']}** — Final: {r['final']} (Proposed: {r['proposed']})")
                if r["warnings"]:
                    for w in r["warnings"]: md_lines.append(f"  - ⚠️ {w}")
                if r["comments"]: md_lines.append(f"  - _Comments:_ {r['comments']}")
            md = "\n".join(md_lines)
            st.download_button(
                "⬇️ Download Markdown",
                data=md.encode("utf-8"),
                file_name="axis_construct_appraisal.md",
                mime="text/markdown"
            )
