"""Programmatic answer checker for E_reason rejection sampling (rule-based, no LLM judge).

Errs on the side of rejecting: a false negative only drops a training example, while a false
positive would teach a wrong solution. v2 fixes v1 false negatives seen in the first 300 samples:
scientific notation written as "a \\times 10^b", units/\\text{} around numbers, "+ C" constants,
unicode symbols (θ, √, μ), "#...#" / "$...$" wrappers, and "y = ..." left-hand sides.
"""

import re

CHECKER_VERSION = "v2"


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k].strip()
    return None


_SCI = re.compile(r"\s*(?:\\times|\\cdot|×|x|\*)\s*10\s*\^\s*\{?\s*([-+−]?\d+)\s*\}?")
_NUM = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _numbers(s: str) -> list[float]:
    s = s.replace("−", "-").replace("\\!", "").replace("{,}", "")
    s = _SCI.sub(lambda m: "e" + m.group(1).replace("−", "-"), s)  # "8.9 \times 10^{5}" -> "8.9e5"
    s = re.sub(r"\\(?:text|mathrm|operatorname)\{[^{}]*\}", " ", s)  # units
    s = re.sub(r"(\d)\s+(?=\d{3}\b)", r"\1", s)  # "6 878" thousands separator
    s = s.replace("\\frac{", "(").replace("}{", ")/(")
    out = []
    for m in _NUM.findall(s):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    # a lone fraction like "\frac{3}{4}" or "3/4"
    frac = re.fullmatch(r"\s*\(?\s*([-+]?\d+(?:\.\d+)?)\s*\)?\s*/\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?\s*", s)
    if frac and float(frac.group(2)) != 0:
        return [float(frac.group(1)) / float(frac.group(2))]
    return out


def _norm_expr(s: str) -> str:
    s = s.strip().strip("#$").strip()
    for a, b in {"θ": "\\theta ", "√": "\\sqrt", "μ": "\\mu ", "π": "\\pi ", "α": "\\alpha ", "β": "\\beta ",
                 "ω": "\\omega ", "λ": "\\lambda ", "Δ": "\\Delta ", "−": "-", "·": "*", "\\left": "", "\\right": "",
                 "\\displaystyle": "", "\\,": "", "\\;": "", "\\!": "", "dfrac": "frac"}.items():
        s = s.replace(a, b)
    s = re.sub(r"\+\s*C'?\s*$", "", s)  # integration constant
    s = re.sub(r"^\s*[a-zA-Z](?:\([a-z]\))?\s*=\s*", "", s)  # "y(x) = ..."
    return s.strip()


def _math_equal(a: str, b: str) -> bool:
    try:
        from math_verify import parse, verify
        pa, pb = parse(f"${a}$"), parse(f"${b}$")
        return bool(pa) and bool(pb) and bool(verify(pa, pb))
    except Exception:
        return False


def is_correct(pred: str | None, gold: str, answer_type: str) -> bool:
    if not pred:
        return False
    if answer_type in {"Float", "Integer", "Percentage", "Fraction"}:
        g, p = _numbers(gold), _numbers(pred)
        if g and p:
            gv = g[0]
            # 1% relative tolerance covers rounding to 3 significant figures (2% let 6878 vs 7.00e3 pass); compare against any number in pred
            # only when pred is short (avoid "x = 3 or 5" style lists matching by luck)
            cands = p if len(p) <= 2 else p[:1]
            if any(abs(gv - pv) <= max(1e-9, 0.01 * abs(gv)) for pv in cands):
                return True
    if answer_type == "Multiple Choice":
        gl = re.match(r"\s*\(?([A-Ja-j])\)?(?:[\s.:)]|$)", gold)
        pl = re.match(r"\s*(?:\\text\{|\\textbf\{|\\mathrm\{)?\s*\(?([A-Ja-j])\)?(?:[\s.:)}]|$)", pred)
        if gl and pl and gl.group(1).lower() == pl.group(1).lower():
            return True
    if answer_type == "Boolean":
        norm = lambda s: re.sub(r"[^a-z]", "", s.lower().split(",")[0].split(".")[0].replace("\\text", ""))
        yes, no = {"yes", "true", "correct"}, {"no", "false", "incorrect"}
        g, p = norm(gold), norm(pred)
        return (g in yes and p in yes) or (g in no and p in no)
    g, p = _norm_expr(gold), _norm_expr(pred)
    if _math_equal(g, p):
        return True
    squash = lambda s: re.sub(r"[\s{}\\]|mathrm|text", "", s.lower())
    return squash(g) == squash(p) and len(squash(g)) > 0
