"""
Draw a Studio workflow as an HTML+SVG concept figure from the graph the
workbench stores (GET /api/workflows/{id}), so the figure shows the saved
workflow rather than a redrawing of it. Node positions come from the graph.
"""
import html
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import harness as H  # noqa: E402

ROOT = H.ROOT
FIG = ROOT / "paper" / "figures"
# What each node runs, named by its backend rather than by the Studio's label.
KIND = {"prompt": "input text", "pysdk": "Python script", "merge": "joins outputs", "chart": "bar chart",
        "table": "table", "output": "text output", "llm": "NeMo Guardrails chat call",
        "guard": "guard model call"}
CLASS = {"prompt": "src", "pysdk": "run", "merge": "node", "chart": "out", "table": "out", "output": "out",
         "llm": "chk", "guard": "chk"}
W, BOXW, BOXH = 1100, 170, 104


def draw(key, name):
    rec = json.loads((ROOT / "results" / "studio_workflows.json").read_text())[str(key)]
    wf = H.get_json(H.CFG["workbench_base"] + f"/api/workflows/{rec['workflow_id']}")
    xs = [n["x"] for n in wf["nodes"]]
    ys = [n["y"] for n in wf["nodes"]]
    sx = (W - 40 - BOXW) / max(1, max(xs) - min(xs))
    sy = min(1.0, 300 / max(1, max(ys) - min(ys)))
    pos = {n["id"]: (20 + (n["x"] - min(xs)) * sx, 20 + (n["y"] - min(ys)) * sy) for n in wf["nodes"]}
    H_ = int(max(p[1] for p in pos.values()) + BOXH + 30)
    wires = []
    for e in wf["edges"]:
        sx0, sy0 = pos[e["from"]]
        tx0, ty0 = pos[e["to"]]
        if abs(sx0 - tx0) < BOXW:
            # Same column: leave the bottom of the source, enter the top of the target.
            x1, y1 = sx0 + BOXW / 2, sy0 + BOXH
            x2, y2 = tx0 + BOXW / 2, ty0 - 2
            wires.append(f'<path d="M{x1:.0f},{y1:.0f} L{x2:.0f},{y2:.0f}" '
                         'stroke="#8a93a2" stroke-width="2" fill="none" marker-end="url(#ag)"/>')
            continue
        x1, y1, x2, y2 = sx0 + BOXW, sy0 + BOXH / 2, tx0 - 2, ty0 + BOXH / 2
        # The curve ends in a straight stub so the arrowhead points along the
        # stroke that reaches it, not along a tangent the eye cannot see.
        xs = x2 - 18
        mx = (x1 + xs) / 2
        wires.append(f'<path d="M{x1:.0f},{y1:.0f} C{mx:.0f},{y1:.0f} {mx:.0f},{y2:.0f} {xs:.0f},{y2:.0f} L{x2:.0f},{y2:.0f}" '
                     'stroke="#8a93a2" stroke-width="2" fill="none" marker-end="url(#ag)"/>')
    boxes = []
    for n in wf["nodes"]:
        x, y = pos[n["id"]]
        title = re.sub(r"\d", "", n.get("title") or KIND[n["type"]]).strip()
        boxes.append(f'<div class="box {CLASS[n["type"]]}" style="left:{x:.0f}px;top:{y:.0f}px;width:{BOXW}px;'
                     f'height:{BOXH}px">\n  <div class="t">{html.escape(title)}</div>\n  <div class="s">'
                     f'{KIND[n["type"]]}</div>\n</div>')
    page = f"""<style>
  *{{box-sizing:border-box;margin:0;padding:0;font-family:Arial,Helvetica,sans-serif}}
  @page{{size:{W}px {H_}px;margin:0}}
  body{{width:{W}px;height:{H_}px;position:relative;background:#fff;color:#1f2733}}
  .box{{position:absolute;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(20,28,43,.10);
       padding:6px 8px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}}
  .t{{font-size:16px;font-weight:bold;line-height:1.15}}
  .s{{font-size:15px;color:#6b7280;line-height:1.2;margin-top:2px}}
  .node{{background:#f6f7f9;border:2px solid #d9dde4}}
  .src{{background:#fdeef1;border:2px solid #e4485f}}
  .run{{background:#eef5fc;border:2px solid #245b81}}
  .chk{{background:#e9f5ef;border:2px solid #2f7d68}}
  .out{{background:#fff7eb;border:2px solid #d97706}}
  svg.wires{{position:absolute;inset:0;width:{W}px;height:{H_}px}}
</style>

<svg class="wires">
  <defs><marker id="ag" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8a93a2"/></marker></defs>
  {chr(10).join('  ' + w for w in wires)}
</svg>

{chr(10).join(boxes)}
"""
    (FIG / f"{name}.html").write_text(page)
    print(f"wrote {name}.html from workflow {rec['workflow_id']}")


if __name__ == "__main__":
    draw(2, "fig_studio")
