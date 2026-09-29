import ast, json, os, re, sqlite3, urllib.request
from flask import Flask, jsonify, request, session, render_template, g
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

# Explicitly set template_folder to the absolute path of the templates directory
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
DB = os.path.join(os.path.dirname(__file__), "reviewer.db")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROVIDER = os.getenv("AI_PROVIDER", "ollama")  # ollama (free, local) | openai
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
MAX_CHARS = 20000

SYSTEM = """You are a senior engineer doing code review. Analyze the user's code and
reply with ONLY a JSON object of this exact shape:
{"summary": str,
 "quality_score": int 0-100,
 "issues": [{"severity": "high|medium|low", "line": int or null, "description": str, "fix": str}],
 "time_complexity": {"value": "e.g. O(n log n)", "explanation": str},
 "space_complexity": {"value": str, "explanation": str},
 "improvements": [str],
 "optimized_code": str}
Be specific and honest. If the code is already good, say so and keep issues short."""


def call_ollama(code, lang):
    body = json.dumps({"model": OLLAMA_MODEL, "stream": False, "format": "json",
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": f"Language: {lang}\n\n{code}"}]}).encode()
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def call_openai(code, lang):
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Language: {lang}\n\n{code}"}])
    return json.loads(resp.choices[0].message.content)


def normalize(r, code):
    """Small local models sometimes skip fields; fill them so the UI never breaks."""
    cx = lambda k: r.get(k) if isinstance(r.get(k), dict) else {"value": str(r.get(k) or "unknown"), "explanation": ""}
    return {"summary": str(r.get("summary", "")), "quality_score": r.get("quality_score", 0),
            "issues": [i for i in r.get("issues", []) if isinstance(i, dict)],
            "time_complexity": cx("time_complexity"), "space_complexity": cx("space_complexity"),
            "improvements": [str(t) for t in r.get("improvements", [])],
            "optimized_code": str(r.get("optimized_code") or code)}


def rule_based(code, lang):
    """No-model fallback: static checks + loop-nesting complexity estimate."""
    issues, depth, recursion, sorts, allocs = [], 0, False, False, False
    add = lambda sev, line, d, f: issues.append({"severity": sev, "line": line, "description": d, "fix": f})
    tree = None
    if lang.lower() in ("python", "auto-detect"):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None
    if tree:
        def loops(n):
            best = 0
            for c in ast.iter_child_nodes(n):
                inc = 1 if isinstance(c, (ast.For, ast.While)) else \
                    len(c.generators) if isinstance(c, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) else 0
                best = max(best, inc + loops(c))
            return best
        depth = loops(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.ExceptHandler) and n.type is None:
                add("medium", n.lineno, "Bare except hides real errors.", "Catch specific exceptions.")
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("eval", "exec"):
                add("high", n.lineno, f"{n.func.id}() runs arbitrary code.", "Use ast.literal_eval or explicit parsing.")
            if isinstance(n, ast.Call) and (getattr(n.func, "id", "") == "sorted" or getattr(n.func, "attr", "") == "sort"):
                sorts = True
            if isinstance(n, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
                allocs = True
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in n.args.defaults):
                    add("medium", n.lineno, f"Mutable default argument in {n.name}().", "Default to None and create the object inside.")
                if any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == n.name for c in ast.walk(n)):
                    recursion = True
    else:
        stack = []
        for line in code.splitlines():
            if re.match(r"\s*(for|while)\b", line):
                ind = len(line) - len(line.lstrip())
                while stack and ind <= stack[-1]:
                    stack.pop()
                stack.append(ind)
                depth = max(depth, len(stack))
        sorts = bool(re.search(r"\bsort(ed)?\s*\(|\.sort\(", code))
        allocs = bool(re.search(r"\[\]|\bnew\s|\{\}|\bvector<|\bArrayList", code))
        if re.search(r"\beval\s*\(", code):
            add("high", None, "eval() runs arbitrary code.", "Avoid eval; parse input explicitly.")
    for i, line in enumerate(code.splitlines(), 1):
        if len(line) > 100:
            add("low", i, "Line is over 100 characters.", "Break it into smaller statements.")
        if re.search(r"TODO|FIXME", line):
            add("low", i, "Unfinished TODO/FIXME left in code.", "Resolve it or track it as an issue.")
    if recursion:
        t = ("O(2^n) to O(n)", "Recursive calls found. Cost depends on how many calls each level makes and whether results are cached.")
    elif depth:
        t = (f"O(n^{depth})" if depth > 1 else "O(n)", f"Loops are nested {depth} deep, so work grows with input size raised to that power.")
    elif sorts:
        t = ("O(n log n)", "A sort call dominates.")
    else:
        t = ("O(1)", "No loops or recursion detected.")
    sp = ("O(n)", "Builds lists or dicts that can grow with input.") if allocs or recursion else ("O(1)", "No growing data structures detected.")
    tips = []
    if depth >= 2:
        tips.append("Nested loops: try a dictionary or set for lookups to drop one level.")
    if recursion:
        tips.append("Cache repeated recursive calls (memoization) or convert to a loop.")
    sev = lambda s: sum(i["severity"] == s for i in issues)
    score = max(30, 100 - 15 * sev("high") - 8 * sev("medium") - 3 * sev("low") - (5 if depth >= 3 else 0))
    return {"summary": "Checked with built-in rules, not an AI model. Start Ollama for a full review and optimized code.",
            "quality_score": score, "issues": issues,
            "time_complexity": {"value": t[0], "explanation": t[1]},
            "space_complexity": {"value": sp[0], "explanation": sp[1]},
            "improvements": tips or ["No obvious problems found by the rule checks."], "optimized_code": code}


def run_analysis(code, lang):
    try:
        raw = call_openai(code, lang) if PROVIDER == "openai" else call_ollama(code, lang)
        r = normalize(raw, code)
        r["engine"] = f"AI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini') if PROVIDER == 'openai' else OLLAMA_MODEL})"
    except Exception:
        r = rule_based(code, lang)
        r["engine"] = "Built-in rules (AI model unavailable)"
    return r


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()


def init_db():
    with sqlite3.connect(DB) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS analyses(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            language TEXT, code TEXT NOT NULL, result TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP);""")


def login_required(f):
    def wrapper(*a, **k):
        if "uid" not in session:
            return jsonify(error="Log in to continue."), 401
        return f(*a, **k)
    wrapper.__name__ = f.__name__
    return wrapper


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/register")
def register():
    d = request.get_json(force=True)
    u, p = (d.get("username") or "").strip(), d.get("password") or ""
    if len(u) < 3 or len(p) < 6:
        return jsonify(error="Username needs 3+ characters and password 6+."), 400
    try:
        cur = db().execute("INSERT INTO users(username,password_hash) VALUES(?,?)",
                           (u, generate_password_hash(p)))
        db().commit()
    except sqlite3.IntegrityError:
        return jsonify(error="That username is taken."), 409
    session["uid"], session["username"] = cur.lastrowid, u
    return jsonify(username=u)


@app.post("/api/login")
def login():
    d = request.get_json(force=True)
    row = db().execute("SELECT * FROM users WHERE username=?", (d.get("username", ""),)).fetchone()
    if not row or not check_password_hash(row["password_hash"], d.get("password", "")):
        return jsonify(error="Wrong username or password."), 401
    session["uid"], session["username"] = row["id"], row["username"]
    return jsonify(username=row["username"])


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@app.get("/api/me")
def me():
    return jsonify(username=session.get("username"))


@app.post("/api/analyze")
@login_required
def analyze():
    d = request.get_json(force=True)
    code, lang = (d.get("code") or "").strip(), d.get("language") or "auto-detect"
    if not code:
        return jsonify(error="Paste or upload some code first."), 400
    if len(code) > MAX_CHARS:
        return jsonify(error=f"Code is over {MAX_CHARS} characters. Submit a smaller piece."), 400
    result = run_analysis(code, lang)
    cur = db().execute("INSERT INTO analyses(user_id,language,code,result) VALUES(?,?,?,?)",
                       (session["uid"], lang, code, json.dumps(result)))
    db().commit()
    return jsonify(id=cur.lastrowid, result=result)


@app.get("/api/history")
@login_required
def history():
    rows = db().execute(
        "SELECT id, language, created_at, substr(code,1,60) AS preview, result FROM analyses "
        "WHERE user_id=? ORDER BY id DESC LIMIT 50", (session["uid"],)).fetchall()
    out = []
    for r in rows:
        res = json.loads(r["result"])
        out.append(dict(id=r["id"], language=r["language"], created_at=r["created_at"],
                        preview=r["preview"], score=res.get("quality_score"),
                        time=(res.get("time_complexity") or {}).get("value")))
    return jsonify(out)


@app.get("/api/history/<int:aid>")
@login_required
def history_item(aid):
    r = db().execute("SELECT * FROM analyses WHERE id=? AND user_id=?", (aid, session["uid"])).fetchone()
    if not r:
        return jsonify(error="Not found."), 404
    return jsonify(id=r["id"], language=r["language"], code=r["code"], result=json.loads(r["result"]))


@app.delete("/api/history/<int:aid>")
@login_required
def delete_item(aid):
    db().execute("DELETE FROM analyses WHERE id=? AND user_id=?", (aid, session["uid"]))
    db().commit()
    return jsonify(ok=True)


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)