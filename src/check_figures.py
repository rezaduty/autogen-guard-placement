"""
Geometry check for the HTML concept diagrams.

Label overflow and collisions are not visible in a build log, so they are
checked here instead of by eye. For each diagram this script parses the
absolutely positioned elements and fails if:

  1. any element extends past the canvas
  2. any two boxes overlap
  3. a box sits outside the zone it belongs to
  4. the estimated text height exceeds the box height, meaning the label
     would be clipped or would spill over its border
  5. the rendered PDF would place any text below a minimum point size

Text height is estimated from the font size, line height, and an average
glyph width for the sans-serif face, which is conservative enough to catch
a label that does not fit.

Exits non-zero on any failure.
"""
import math
import pathlib
from html.parser import HTMLParser
import re
import sys

FIGURES = pathlib.Path("paper/figures")

# Width of the rendered figure on the page, in inches, per diagram. Used to
# convert a CSS pixel size into the point size the reader actually sees.
RENDER_WIDTH_IN = {
    "fig_placements": 7.10,
    "fig_frameworks": 7.10,
    "fig_protocol": 7.10,
    "fig_studio": 7.10,
}
MIN_POINT_SIZE = 6.5
AVG_GLYPH_RATIO = 0.52   # average advance width as a fraction of font size
PAD_X = 10               # .box horizontal padding, each side
PAD_Y = 8                # .box vertical padding, each side

FAILURES = []


def style_dict(style):
    out = {}
    for part in style.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def px(v):
    m = re.match(r"(-?[\d.]+)px", (v or "").strip())
    return float(m.group(1)) if m else None


class _Boxes(HTMLParser):
    """Collects every absolutely positioned div with the text of its .t and
    .s children. A nested-tag parser rather than a regex: a regex that stops
    at the first closing div loses the title of any box that also has a
    subtitle, and then skips that box's fit check silently."""

    def __init__(self):
        super().__init__()
        self.stack, self.els = [], []

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            owner = next((e for _, e in reversed(self.stack) if e is not None), None)
            if owner is not None:
                owner["segs"].append("")
            return
        if tag != "div":
            return
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        st = style_dict(a.get("style") or "")
        el = None
        if "left" in st and "top" in st:
            el = {"classes": cls, "x": px(st.get("left")), "y": px(st.get("top")), "w": px(st.get("width")),
                  "h": px(st.get("height")), "title": "", "sub": "", "text": "",
                  "is_box": "box" in cls, "is_zone": "zone" in cls, "is_head": "phead" in cls, "segs": [""],
                  "iw": float(a.get("data-iw") or 0), "ih": float(a.get("data-ih") or 0)}
            self.els.append(el)
        self.stack.append((cls, el))

    def handle_endtag(self, tag):
        if tag == "div" and self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if not self.stack:
            return
        owner = next((e for _, e in reversed(self.stack) if e is not None), None)
        if owner is None:
            return
        cls = self.stack[-1][0]
        if "t" in cls:
            owner["title"] += data.strip()
        elif "s" in cls:
            owner["sub"] += data.strip()
        owner["text"] = (owner["text"] + " " + data.strip()).strip()
        owner["segs"][-1] = (owner["segs"][-1] + " " + data.strip()).strip()


def parse(html):
    """Pull out canvas size, css font sizes, and positioned elements."""
    canvas = re.search(r"body\{width:(\d+)px;height:(\d+)px", html)
    W, H = (int(canvas.group(1)), int(canvas.group(2))) if canvas else (0, 0)

    fonts = {}
    for cls, size in re.findall(r"\.([a-zA-Z]+)\{[^}]*font-size:([\d.]+)px", html):
        fonts[cls] = float(size)

    body = re.sub(r"<style>.*?</style>|<svg.*?</svg>", "", html, flags=re.S)
    p = _Boxes()
    p.feed(body)
    return W, H, fonts, p.els


def wrapped_lines(text, font_px, inner_w):
    if not text or not inner_w:
        return 0
    per_line = max(1, int(inner_w / (font_px * AVG_GLYPH_RATIO)))
    words, lines, cur = text.split(), 1, 0
    for word in words:
        add = len(word) + (1 if cur else 0)
        if cur + add > per_line:
            lines += 1
            cur = len(word)
        else:
            cur += add
    return lines


def check_figure(name):
    path = FIGURES / f"{name}.html"
    if not path.exists():
        FAILURES.append(f"{name}: html source missing")
        return
    html = path.read_text()
    W, H, fonts, els = parse(html)
    if not W or not H:
        FAILURES.append(f"{name}: could not read canvas size")
        return

    scale = RENDER_WIDTH_IN.get(name, 7.10) * 72.0 / W
    for cls, size in sorted(fonts.items()):
        pt = size * scale
        if pt < MIN_POINT_SIZE:
            FAILURES.append(f"{name}: class .{cls} renders at {pt:.1f}pt, "
                            f"below the {MIN_POINT_SIZE}pt floor")

    boxes = [e for e in els if e["is_box"]]
    zones = [e for e in els if e["is_zone"]]

    # 1. inside the canvas
    for e in els:
        if e["w"] is None or e["h"] is None:
            continue
        if e["x"] < 0 or e["y"] < 0 or e["x"] + e["w"] > W or e["y"] + e["h"] > H:
            FAILURES.append(
                f"{name}: element {e['title'] or e['text'][:28]!r} at "
                f"({e['x']},{e['y']},{e['w']},{e['h']}) leaves the "
                f"{W}x{H} canvas")

    # 2. no two boxes overlap
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
                    and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]):
                FAILURES.append(
                    f"{name}: boxes {a['title']!r} and {b['title']!r} overlap")

    # 3. a box that sits within a zone's span must sit within it fully
    for bx in boxes:
        for z in zones:
            cx, cy = bx["x"] + bx["w"] / 2, bx["y"] + bx["h"] / 2
            inside_centre = (z["x"] <= cx <= z["x"] + z["w"]
                             and z["y"] <= cy <= z["y"] + z["h"])
            if not inside_centre:
                continue
            if not (bx["x"] >= z["x"] and bx["y"] >= z["y"]
                    and bx["x"] + bx["w"] <= z["x"] + z["w"]
                    and bx["y"] + bx["h"] <= z["y"] + z["h"]):
                FAILURES.append(
                    f"{name}: box {bx['title']!r} crosses its zone border")

    # 4. text must fit inside the box
    ft = fonts.get("t", 13.0)
    fs = fonts.get("s", 10.0)
    for bx in boxes:
        if not bx["w"] or not bx["h"]:
            continue
        # an icon beside the text narrows it, an icon above it adds height
        inner_w = bx["w"] - 2 * PAD_X - bx.get("iw", 0)
        h = bx.get("ih", 0)
        if bx["title"]:
            h += wrapped_lines(bx["title"], ft, inner_w) * ft * 1.25
        if bx["sub"]:
            h += 5 + wrapped_lines(bx["sub"], fs, inner_w) * fs * 1.3
        need = max(h, bx.get("iw", 0) - 10) + 2 * PAD_Y
        if need > bx["h"]:
            FAILURES.append(
                f"{name}: label in box {bx['title']!r} needs about "
                f"{need:.0f}px but the box is {bx['h']:.0f}px tall")

    # 6. a panel heading, wrapped at its width, must clear every box below it
    fh = fonts.get("phead", 21.0)
    for hd in (e for e in els if e.get("is_head")):
        fq = fonts.get("pq", fh)

        def seg_lines(seg):
            # the parenthesised qualifier is set in the smaller .pq size
            title, _, qual = seg.partition("(")
            width = len(title) * fh * AVG_GLYPH_RATIO + (len(qual) + 1 if qual else 0) * fq * AVG_GLYPH_RATIO
            return max(1, math.ceil(width / hd["w"]))
        lines = sum(seg_lines(seg) for seg in hd["segs"] if seg)
        bottom = hd["y"] + lines * fh * 1.2
        for bx in boxes:
            if (bx["x"] < hd["x"] + hd["w"] and hd["x"] < bx["x"] + bx["w"]
                    and hd["y"] < bx["y"] + bx["h"] and bx["y"] < bottom):
                FAILURES.append(f"{name}: heading {hd['text']!r} runs into box {bx['title']!r}")

    print(f"  {name}: {len(boxes)} boxes, {len(zones)} zones, "
          f"canvas {W}x{H}, smallest type "
          f"{min(fonts.values()) * scale:.1f}pt")


def _pt(t, p0, p1, p2, p3):
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1])


def _angle(a, b):
    na, nb = math.hypot(*a), math.hypot(*b)
    if not na or not nb:
        return 0.0
    c = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(c))


def sample_path(d):
    """Points along an M/C/L path, for the pass-through check."""
    toks = re.findall(r"[MCL]|-?[\d.]+", d)
    pts, cur, i = [], None, 0
    while i < len(toks):
        c = toks[i]
        if c == "M":
            cur = (float(toks[i + 1]), float(toks[i + 2]))
            pts.append(cur)
            i += 3
        elif c == "L":
            nxt = (float(toks[i + 1]), float(toks[i + 2]))
            pts += [(cur[0] + (nxt[0] - cur[0]) * t / 10, cur[1] + (nxt[1] - cur[1]) * t / 10) for t in range(1, 11)]
            cur = nxt
            i += 3
        elif c == "C":
            p1 = (float(toks[i + 1]), float(toks[i + 2]))
            p2 = (float(toks[i + 3]), float(toks[i + 4]))
            p3 = (float(toks[i + 5]), float(toks[i + 6]))
            pts += [_pt(t / 20, cur, p1, p2, p3) for t in range(1, 21)]
            cur = p3
            i += 7
        else:
            i += 1
    return pts


def arrows(html):
    """Every wire drawn with an arrowhead: its end point, the direction the
    arrowhead points (the final tangent), and the direction the visible
    stroke arrives from (the chord over its last part)."""
    out = []
    for m in re.finditer(r'<line x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"[^>]*marker-end', html):
        x1, y1, x2, y2 = map(float, m.groups())
        d = (x2 - x1, y2 - y1)
        pts = [(x1 + (x2 - x1) * t / 20, y1 + (y2 - y1) * t / 20) for t in range(21)]
        out.append({"end": (x2, y2), "head": d, "stroke": d, "len": math.hypot(*d), "src": m.group(0)[:60], "pts": pts})
    for m in re.finditer(r'<path d="([^"]+)"[^>]*marker-end', html):
        nums = [float(v) for v in re.findall(r"-?[\d.]+", m.group(1))]
        cmds = re.findall(r"[MCL]", m.group(1))
        if cmds[-1] == "C":
            p0, p1, p2, p3 = (nums[-8], nums[-7]), (nums[-6], nums[-5]), (nums[-4], nums[-3]), (nums[-2], nums[-1])
            head = (p3[0] - p2[0], p3[1] - p2[1])
            q = _pt(0.8, p0, p1, p2, p3)
            stroke = (p3[0] - q[0], p3[1] - q[1])
            ln = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        else:
            p0, p3 = (nums[-4], nums[-3]), (nums[-2], nums[-1])
            head = stroke = (p3[0] - p0[0], p3[1] - p0[1])
            ln = math.hypot(*head)
        out.append({"end": p3, "head": head, "stroke": stroke, "len": ln, "src": m.group(1)[:60],
                    "pts": sample_path(m.group(1))})
    return out


def check_arrows(name):
    """An arrowhead must point the way its stroke arrives, and must land on
    the edge of a box rather than short of it, past it, or inside it."""
    html = (FIGURES / f"{name}.html").read_text()
    _, _, _, els = parse(html)
    boxes = [e for e in els if e["is_box"] and e["w"] and e["h"]]
    for a in arrows(html):
        ang = _angle(a["head"], a["stroke"])
        if ang > 20:
            FAILURES.append(f"{name}: arrowhead turned {ang:.0f} degrees from its stroke ({a['src']})")
        if a["len"] < 14:
            FAILURES.append(f"{name}: arrow too short to read ({a['src']})")
        x, y = a["end"]
        best = None
        for b in boxes:
            dx = max(b["x"] - x, 0, x - (b["x"] + b["w"]))
            dy = max(b["y"] - y, 0, y - (b["y"] + b["h"]))
            outside = math.hypot(dx, dy)
            inside = 0.0 if outside else min(x - b["x"], b["x"] + b["w"] - x, y - b["y"], b["y"] + b["h"] - y)
            gap = outside if outside else -inside
            if best is None or abs(gap) < abs(best):
                best = gap
        # The stroke must not pass through a box other than the one it leaves
        # and the one it enters: a wire drawn behind a box looks like it
        # connects something it does not.
        def inside(b, pt, m=5):
            return b["x"] + m < pt[0] < b["x"] + b["w"] - m and b["y"] + m < pt[1] < b["y"] + b["h"] - m
        ends = [b for b in boxes if any(math.hypot(max(b["x"] - q[0], 0, q[0] - b["x"] - b["w"]),
                                                   max(b["y"] - q[1], 0, q[1] - b["y"] - b["h"])) <= 9
                                        for q in (a["pts"][0], a["end"]))]
        crossed = [b["title"] for b in boxes if b not in ends and any(inside(b, q) for q in a["pts"])]
        if crossed:
            FAILURES.append(f"{name}: wire passes through box {crossed[0]!r} ({a['src']})")
        # A wire leaves its source outward and enters its target from outside.
        # One that doubles back through its own end boxes is hidden behind them.
        k = max(1, len(a["pts"]) // 6)
        for b in ends:
            if any(inside(b, q, m=3) for q in a["pts"][k:-k]):
                FAILURES.append(f"{name}: wire runs inside its own end box {b['title']!r} ({a['src']})")
        if best is None or best > 9 or best < -3:
            FAILURES.append(f"{name}: arrow ends {best if best is not None else 'nowhere'} px from the nearest box edge ({a['src']})")


def check_text_rules(name):
    """No numbers and no dashes or arrows in visible figure text. Numbers
    belong in captions and macros, where the verifier can cross-check them."""
    html = (FIGURES / f"{name}.html").read_text()
    body = re.sub(r"<style>.*?</style>", "", html, flags=re.S)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    visible = re.sub(r"<[^>]+>", " ", body)
    for m in re.finditer(r"\d", visible):
        FAILURES.append(f"{name}: digit in visible text near {visible[max(0, m.start() - 20):m.start() + 20]!r}")
        break
    for pat in ("\u2014", "\u2013", "->", "\u2192", "&mdash;", "&ndash;", ";"):
        if pat in visible:
            FAILURES.append(f"{name}: forbidden {pat!r} in visible text")


def main():
    print("Checking concept diagram geometry")
    # The concept-diagram sources are not distributed with the artifact. When
    # the generator is absent, a figure without its HTML is checked only for
    # its rendered PDF. With the generator present a missing HTML fails.
    full = pathlib.Path("src/make_diagrams.py").exists()
    for name in sorted(RENDER_WIDTH_IN):
        if not full and not (FIGURES / f"{name}.html").exists():
            if not (FIGURES / f"{name}.pdf").exists():
                FAILURES.append(f"{name}: neither source nor rendered PDF present")
            print(f"  {name}: source not distributed, rendered PDF present")
            continue
        check_figure(name)
        check_text_rules(name)
        check_arrows(name)
    if FAILURES:
        print(f"FIGURE CHECK FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("FIGURE CHECK PASSED")


if __name__ == "__main__":
    main()
