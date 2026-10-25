"""
The AI sections of the HTML reports (requirements REP-03, AI-04, AI-05). Every section carries the same marking as the
web app: the "AI-generated" badge, the sentence that it is decision support and not a diagnosis and has to be reviewed
by a psychologist, and the provenance (model, version, licence, parameters, prompt or rule version, confidence, evidence).
The provenance is printed in full, not folded away, because a printed report cannot be unfolded. All text is escaped.
"""

import json
from typing import Any

from analysis import catalog
from analysis.reports import esc
from ai.envelope import DISCLAIMER, REASONS

STYLES = """
section.ai{border:2px solid #7c3aed;background:#faf5ff}
.ai-head{display:flex;align-items:center;gap:8px;margin-bottom:6px}.ai-head h2{margin:0}
.ai-badge{background:#7c3aed;color:#fff;border-radius:999px;padding:2px 10px;font-size:12px;font-weight:700;white-space:nowrap}
.ai-warn{margin:0 0 10px;font-size:13px;color:#5b21b6;font-weight:600}
.ai-prov{margin-top:12px;border-top:1px dashed #c4b5fd;padding-top:8px;font-size:12px;color:#475569}
.ai-prov td,.ai-prov th{padding:2px 8px 2px 0;border:0;font-size:12px}
.ai ul{margin:4px 0 8px;padding-left:20px}
"""

TITLES = {
    "indicators": "Indicators from the fuzzy rule base", "cohort": "Comparison with other drawings",
    "groups": "Stroke groups", "labels": "Semantic labels (CLIP)", "description": "Description by a vision-language model",
    "summary": "Conclusion by a language model", "insights": "Session insights",
}


def _list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>" if items else '<p class="note">None</p>'


def _metric_name(metric_id: str) -> str:
    return catalog.metric(metric_id)["name"]


def _indicators(result: dict[str, Any]) -> str:
    rows = []
    for ind in result["indicators"]:
        fired = "; ".join(f"{f['rule']} ({round(f['strength'] * 100)} %)" for f in ind["fired"][:3]) or "none"
        rows.append(f"<tr><td>{esc(ind['name'])}</td><td>{esc(ind['text'] or 'no rule applies')}</td>"
                    f"<td class=\"num\">{round(ind['degree'] * 100)} %</td><td>{esc(fired)}</td></tr>")
    basis = {"cohort": "the percentiles of the comparison group", "fixed": "fixed ranges (the group is small)"}.get(
        result["basis"], "percentiles of the group and fixed ranges")
    rule_base = result["ruleBase"]
    return (f'<p class="note">Compared using {esc(basis)}, {result["cohortSize"]} other drawings. Rule base: '
            f'{esc(rule_base["name"])} v{esc(rule_base["version"])}, status {esc(rule_base["status"])}.</p>'
            '<table><tr><th>Indicator</th><th>Level</th><th class="num">Degree</th><th>Strongest rules</th></tr>'
            + "".join(rows) + "</table>")


def _cohort(result: dict[str, Any]) -> str:
    subject = result.get("subject")
    if not subject:
        return f"<p>{result['cohortSize']} drawings were compared.</p>"
    directory = result["directory"]
    notable = [f"{_metric_name(n['metric'])}: {'higher' if n['direction'] == 'high' else 'lower'} than the others "
               f"(robust z {n['z']})" for n in subject["notable"]]
    similar = [f"{directory.get(s['id'], {}).get('name', s['id'][:8])} (similarity {s['similarity']})" for s in subject["similar"]]
    verdict = ("This drawing is atypical for its comparison group on at least one measurement."
               if subject["atypical"] else "This drawing is not atypical for its comparison group.")
    return (f"<p>{esc(verdict)} ({result['cohortSize']} drawings compared.)</p><h3>Notable measurements</h3>{_list(notable)}"
            f"<h3>Most similar drawings</h3>{_list(similar)}")


def _groups(result: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{esc(g['id'])}</td><td class=\"num\">{g['strokeCount']}</td><td class=\"num\">{round(g['inkShare'] * 100)} %</td>"
        f"<td>{' × '.join(str(round(s * 100)) for s in g['size'])} cm</td>"
        f"<td>{''.join(f'<span class=swatch style=background:{esc(c['hex'])}></span>' for c in g['colours'])}</td></tr>"
        for g in result["groups"])
    return (f"<p>{len(result['groups'])} groups, {len(result['unclustered'])} strokes belong to no group.</p>"
            '<table><tr><th>Group</th><th class="num">Strokes</th><th class="num">Ink</th><th>Size (x × y × z)</th>'
            f"<th>Colours</th></tr>{rows}</table>")


def _labels(result: dict[str, Any]) -> str:
    def line(verdict: dict[str, Any], name: str) -> str:
        top = ", ".join(f"{esc(t['name'])} {round(t['score'] * 100)} %" for t in verdict["top"])
        flag = " (unspecific)" if verdict["unspecific"] else ""
        return f"<tr><td>{esc(name)}</td><td>{top}{esc(flag)}</td></tr>"

    return ("<table><tr><th>Subject</th><th>Best matching vocabulary entries</th></tr>"
            + line(result["whole"], "Whole drawing") + "".join(line(g, f"Group {g['groupId']}") for g in result["groups"])
            + "</table>")


def _description(result: dict[str, Any]) -> str:
    objects = [f"{o['name']} (model's certainty: {o['certainty']})" for o in result["depictedObjects"]]
    return (f"<p>{esc(result['scene'])}</p><h3>Depicted objects{' (abstract drawing)' if result['abstract'] else ''}</h3>{_list(objects)}"
            f"<h3>Notable features</h3>{_list(result['notableFeatures'])}"
            f"<h3>Colour words</h3>{_list(result['colourMoodWords'])}")


def _summary(result: dict[str, Any]) -> str:
    return (f"<p>{esc(result['overview'])}</p><h3>Observations</h3>{_list(result['observations'])}"
            f"<h3>Things the psychologist could explore</h3>{_list(result['thingsToExplore'])}"
            f"<h3>Limitations</h3>{_list(result['limitations'])}")


def _insights(result: dict[str, Any]) -> str:
    directory = result["directory"]
    lines = []
    for item in result["insights"]:
        if item["kind"] == "recurringLabel":
            lines.append(f"The label \"{item['name']}\" was named for {item['drawings']} drawings of {item['participants']} participants.")
        elif item["kind"] == "indicatorLevels":
            counts = ", ".join(f"{level} {n}" for level, n in item["counts"].items())
            lines.append(f"{item['name']}: {counts}" + (f" (mostly {item['common']})" if item["common"] else ""))
        elif item["kind"] == "atypical":
            names = ", ".join(directory.get(i, {}).get("name", i[:8]) for i in item["drawingIds"])
            lines.append(f"Atypical for the session on at least one measurement: {names}.")
    return f"<p>{result['drawingCount']} drawings.</p>{_list(lines)}"


BODIES = {"indicators": _indicators, "cohort": _cohort, "groups": _groups, "labels": _labels,
          "description": _description, "summary": _summary, "insights": _insights}


def _provenance(section: dict[str, Any]) -> str:
    model = section["model"]
    confidence = (f"{round(section['confidence'] * 100)} % ({section['confidenceKind']})"
                  if section["confidence"] is not None else "none")
    evidence = section["evidence"]
    kinds = ", ".join(sorted({e["kind"] for e in evidence})) or "none"
    rows = [("Model", f"{model['name']}, version {model['version']}, licence {model['licence']}"),
            ("Parameters", json.dumps(section["params"], sort_keys=True)),
            ("Prompt / rules / vocabulary", section["promptVersion"] or "none"),
            ("Confidence", confidence), ("Evidence", f"{len(evidence)} items ({kinds})"),
            ("Generated", section["generatedAt"][:16].replace("T", " ") + " UTC"),
            ("Evaluated with psychologists", "yes" if section["evaluated"] else "not yet")]
    notes = "".join(f"<br>{esc(n)}" for n in section["notes"])
    return ('<div class="ai-prov"><table>' + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
            + f"</table>{notes}</div>")


def sections_html(sections: list[dict[str, Any]]) -> str:
    """The report sections of AI results (envelopes with status ok or notEnoughData)."""

    out = []
    for section in sections:
        feature = section["feature"]
        body = (BODIES[feature](section["result"]) if section["status"] == "ok"
                else f"<p class=\"note\">{esc(REASONS.get(section['reason'] or '', 'No result.'))}</p>")
        outdated = ('<p class="ai-warn" style="color:#b45309">Out of date: the drawing or the findings this result was written '
                    'from changed afterwards. Run the function again for a current result.</p>') if section.get("stale") else ""
        out.append(
            f'<section class="ai" aria-label="AI-generated"><div class="ai-head"><span class="ai-badge">AI-generated</span>'
            f"<h2>{esc(TITLES[feature])}</h2></div><p class=\"ai-warn\">{esc(DISCLAIMER)}</p>{outdated}{body}{_provenance(section)}</section>")
    return "".join(out)
