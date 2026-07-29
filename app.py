import os
import re
import time
import json as _json
import urllib.request
import urllib.error
from datetime import datetime
from functools import wraps
from urllib.parse import quote_plus, quote as url_quote

# Load .env file BEFORE anything reads os.environ
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
from flask_pymongo import PyMongo
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
from bson.objectid import ObjectId

# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

# ── Security: load secrets from environment, never hardcode ──
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production-use-env-var")

# ── CSRF Protection ──
csrf = CSRFProtect(app)
# Allow CSRF token via X-CSRFToken header (for AJAX/fetch calls)
app.config["WTF_CSRF_HEADERS"] = ["X-CSRFToken"]

# ── Rate Limiting ──
# Use Redis in production for accurate limits across multiple workers.
# Set REDIS_URL env var to switch: e.g. "redis://localhost:6379"
_redis_url = os.environ.get("REDIS_URL", "memory://")
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "60 per hour"],
    storage_uri=_redis_url
)

# ── Secure session cookies ──
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Uncomment the line below when running with HTTPS in production:
# app.config["SESSION_COOKIE_SECURE"] = True

# ── MongoDB URI from environment ──
app.config["MONGO_URI"] = os.environ.get(
    "MONGO_URI", "mongodb://localhost:27017/skillmatch_db"
)
mongo = PyMongo(app)
users_collection = mongo.db.users
applications_collection = mongo.db.applications

# ── File uploads ──
UPLOAD_FOLDER = "resumes"
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Whitelist of toggleable user settings (prevents field injection) ──
ALLOWED_TOGGLE_KEYS = {
    "job_alerts", "skill_tips", "interview_reminders", "application_updates"
}

# ── Gemini API config ──
GEMINI_MAX_RETRIES = 3          # how many times to retry on 429
GEMINI_RETRY_BASE_WAIT = 60      # seconds (doubles each attempt: 2, 4, 8)
GEMINI_TIMEOUT = 120             # request timeout in seconds

# ── URL filter for templates ──
@app.template_filter("urlencode")
def urlencode_filter(s):
    return url_quote(str(s))

# =========================================================
# HELPERS
# =========================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(path):
    text = ""
    ext = path.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif ext == "docx":
        doc = Document(path)
        for para in doc.paragraphs:
            text += para.text + " "
    return text.lower()


def extract_skills(text):
    SKILLS_DB = [
        # Languages
        "python", "java", "javascript", "typescript", "c++", "c", "c#", "go",
        "rust", "ruby", "php", "scala", "r", "matlab", "swift", "kotlin",
        # Web Frontend
        "html", "css", "react", "angular", "vue", "nextjs", "svelte",
        "tailwind", "bootstrap", "sass", "webpack",
        # Backend / Frameworks
        "flask", "django", "fastapi", "spring", "node", "express", "laravel",
        "rails", "graphql", "rest", "grpc",
        # Databases
        "mongodb", "sql", "mysql", "postgresql", "sqlite", "redis", "cassandra",
        "elasticsearch", "dynamodb", "oracle",
        # Cloud & DevOps
        "aws", "gcp", "azure", "docker", "kubernetes", "linux", "terraform",
        "ansible", "jenkins", "ci/cd", "github actions", "nginx",
        # AI/ML
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "keras",
        "huggingface", "langchain", "openai", "mlops", "spark",
        # Mobile
        "android", "ios", "flutter", "react native", "firebase",
        # Tools
        "git", "jira", "agile", "scrum", "figma", "postman",
    ]
    found = []
    for s in SKILLS_DB:
        pattern = r'(?<![a-z0-9])' + re.escape(s) + r'(?![a-z0-9])'
        if re.search(pattern, text):
            found.append(s)
    return [s.title() if len(s) > 2 else s.upper() for s in found]


# =========================================================
# GEMINI API — with retry & 429 handling
# =========================================================

def call_gemini_with_retry(url: str, payload: bytes,
                            max_retries: int = GEMINI_MAX_RETRIES,
                            base_wait: int = GEMINI_RETRY_BASE_WAIT) -> dict:
    """
    Call the Gemini API with exponential-backoff retry on HTTP 429.
    Raises urllib.error.HTTPError for non-429 HTTP errors.
    Raises RuntimeError after exhausting all retries.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as resp:
                return _json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Check for Retry-After header from Gemini
                retry_after = e.headers.get("Retry-After")
                if retry_after:
                    wait = int(retry_after)
                else:
                    wait = base_wait * (2 ** attempt)   # 2s, 4s, 8s
                print(f"[Gemini 429] Rate limited. Attempt {attempt + 1}/{max_retries}. "
                      f"Retrying in {wait}s...")
                last_error = e
                time.sleep(wait)
            else:
                # Non-429 HTTP error — don't retry, raise immediately
                raise

        except urllib.error.URLError as e:
            # Network-level error (timeout, DNS, etc.)
            wait = base_wait * (2 ** attempt)
            print(f"[Gemini URLError] {e}. Attempt {attempt + 1}/{max_retries}. "
                  f"Retrying in {wait}s...")
            last_error = e
            time.sleep(wait)

    raise RuntimeError(
        f"Gemini API unavailable after {max_retries} retries. "
        f"Last error: {last_error}"
    )


# ============================================================
# REAL ATS SCORING ENGINE
# ============================================================

def compute_ats_score(text, skills, job_fit):
    text_lower = text.lower()
    word_count = len(text.split())
    breakdown = {}

    # ── 1. SKILLS MATCH (25 pts) ──────────────────────────────
    required = ROLE_REQUIREMENTS.get(job_fit, ROLE_REQUIREMENTS.get("Backend Developer", []))
    required_names_lower = [r["name"].lower() for r in required]
    user_skills_lower = [s.lower() for s in skills]
    matched_required = sum(1 for r in required_names_lower if r in user_skills_lower)
    skills_score = min(int((matched_required / max(len(required_names_lower), 1)) * 25), 25)
    bonus_skills = min(len(skills) - matched_required, 5)
    skills_score = min(skills_score + bonus_skills, 25)
    breakdown["Skills Match"] = {
        "score": skills_score, "max": 25,
        "detail": f"{matched_required}/{len(required_names_lower)} required skills found, {len(skills)} total skills detected"
    }

    # ── 2. KEYWORDS & JOB-SPECIFIC TERMS (20 pts) ────────────
    JOB_KEYWORDS = {
        "Backend Developer":   ["api", "rest", "microservice", "server", "backend", "database", "endpoint", "deployment", "scalab"],
        "Full Stack Engineer": ["frontend", "backend", "full stack", "api", "database", "deploy", "responsive", "ui", "ux"],
        "Frontend Developer":  ["ui", "ux", "responsive", "component", "interface", "design", "accessibility", "performance"],
        "Data Scientist":      ["model", "analysis", "dataset", "predict", "machine learning", "data", "insight", "statistic", "visualiz"],
        "DevOps Engineer":     ["pipeline", "deploy", "infrastructure", "monitoring", "automat", "container", "cluster", "ci/cd"],
        "ML Engineer":         ["model", "training", "inference", "neural", "deep learning", "ml", "accuracy", "dataset", "pipeline"],
        "Android Developer":   ["android", "mobile", "app", "kotlin", "java", "ui", "material design", "api", "sdk"],
        "Cyber Security Engineer": ["security", "vulnerability", "penetrat", "firewall", "encrypt", "threat", "audit", "compliance"],
        "Software Engineer":   ["software", "develop", "engineer", "system", "application", "solution", "architecture"],
    }
    keywords = JOB_KEYWORDS.get(job_fit, JOB_KEYWORDS["Software Engineer"])
    kw_hits = sum(1 for kw in keywords if kw in text_lower)
    kw_score = min(int((kw_hits / max(len(keywords), 1)) * 20), 20)
    breakdown["Keywords"] = {
        "score": kw_score, "max": 20,
        "detail": f"{kw_hits}/{len(keywords)} job-specific keywords found"
    }

    # ── 3. EXPERIENCE & ACHIEVEMENTS (20 pts) ────────────────
    quant_hits = len(re.findall(
        r'\d+[%x]|\d+\s*(million|billion|k\b|\+|users|requests|accuracy|latency|reduction|improvement|increase|decrease)',
        text_lower
    ))
    quant_score = min(quant_hits * 4, 12)
    ACTION_VERBS = [
        "developed", "built", "designed", "implemented", "led", "managed",
        "created", "deployed", "optimized", "reduced", "increased", "improved",
        "automated", "integrated", "architected", "launched", "delivered",
        "collaborated", "mentored", "migrated", "scaled", "engineered"
    ]
    verb_hits = sum(1 for v in ACTION_VERBS if v in text_lower)
    verb_score = min(verb_hits, 8)
    exp_score = min(quant_score + verb_score, 20)
    breakdown["Experience Quality"] = {
        "score": exp_score, "max": 20,
        "detail": f"{quant_hits} quantified achievements, {verb_hits} action verbs detected"
    }

    # ── 4. RESUME STRUCTURE & SECTIONS (15 pts) ──────────────
    SECTIONS = {
        "contact":    ["email", "phone", "linkedin", "github", "portfolio"],
        "summary":    ["summary", "objective", "profile", "about"],
        "experience": ["experience", "work history", "employment", "internship", "intern"],
        "education":  ["education", "degree", "university", "college", "b.tech", "b.e", "b.sc", "m.tech", "mca"],
        "skills":     ["skills", "technical skills", "technologies", "tools"],
        "projects":   ["project", "built", "developed a", "created a"],
    }
    sections_found = 0
    section_names_found = []
    for sec, clues in SECTIONS.items():
        if any(c in text_lower for c in clues):
            sections_found += 1
            section_names_found.append(sec)
    struct_score = min(int((sections_found / len(SECTIONS)) * 15), 15)
    breakdown["Structure & Sections"] = {
        "score": struct_score, "max": 15,
        "detail": f"{sections_found}/{len(SECTIONS)} sections found: {', '.join(section_names_found)}"
    }

    # ── 5. EDUCATION (10 pts) ─────────────────────────────────
    EDU_SIGNALS = [
        "b.tech", "b.e", "b.sc", "bachelor", "master", "m.tech", "mca",
        "mba", "phd", "degree", "university", "college", "cgpa", "gpa",
        "10th", "12th", "sslc", "hsc"
    ]
    edu_hits = sum(1 for e in EDU_SIGNALS if e in text_lower)
    edu_score = min(edu_hits * 2, 10)
    breakdown["Education"] = {
        "score": edu_score, "max": 10,
        "detail": "Education details detected" if edu_score > 0 else "No clear education section found"
    }

    # ── 6. FORMATTING & ATS-FRIENDLINESS (10 pts) ────────────
    fmt_score = 0
    if 300 <= word_count <= 900:
        fmt_score += 4
    elif word_count > 200:
        fmt_score += 2
    garbage_ratio = len(re.findall(r'[\x00-\x08\x0e-\x1f\x7f-\x9f]', text)) / max(len(text), 1)
    if garbage_ratio < 0.01:
        fmt_score += 3
    has_email = bool(re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text_lower))
    has_phone = bool(re.search(r'[+\d][\d\s\-()]{8,}', text))
    if has_email:
        fmt_score += 2
    if has_phone:
        fmt_score += 1
    fmt_score = min(fmt_score, 10)
    breakdown["ATS Formatting"] = {
        "score": fmt_score, "max": 10,
        "detail": f"Word count: {word_count} | Email: {'✓' if has_email else '✗'} | Phone: {'✓' if has_phone else '✗'}"
    }

    total = skills_score + kw_score + exp_score + struct_score + edu_score + fmt_score
    total = max(min(total, 100), 0)
    return total, breakdown


def get_ats_grade(score):
    if score >= 85:
        return ("Excellent", "#34d399", "🏆 Your resume is highly ATS-optimized. Strong chance of shortlisting.")
    if score >= 70:
        return ("Good", "#6366f1", "✅ Good score. A few tweaks will push you to the top tier.")
    if score >= 55:
        return ("Average", "#fbbf24", "⚠️ Average score. Improve keywords and add quantified achievements.")
    return ("Needs Work", "#f87171", "❌ Low ATS score. Restructure your resume with the suggestions below.")


def determine_job_fit(skills):
    """Score all roles and return the best match."""
    s = [x.lower() for x in skills]
    role_signals = {
        "Data Scientist":          ["tensorflow", "pandas", "numpy", "scikit-learn", "scikit"],
        "DevOps Engineer":         ["docker", "kubernetes", "aws", "linux", "gcp", "azure", "terraform", "ansible"],
        "Android Developer":       ["flutter", "kotlin", "swift", "android", "ios", "firebase"],
        "ML Engineer":             ["pytorch", "tensorflow", "keras", "huggingface", "mlops", "langchain"],
        "Cyber Security Engineer": ["security", "penetration", "firewall", "encrypt"],
        "Full Stack Engineer":     ["react", "angular", "vue", "python", "django", "flask", "node", "java", "spring"],
        "Frontend Developer":      ["react", "angular", "vue", "css", "html", "typescript"],
        "Backend Developer":       ["python", "django", "flask", "java", "spring", "node", "mongodb", "sql"],
    }
    scores = {role: sum(1 for sig in sigs if sig in s) for role, sigs in role_signals.items()}
    best_role = max(scores, key=scores.get)
    if scores[best_role] == 0:
        return "Software Engineer"
    return best_role


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


def get_current_user():
    user = users_collection.find_one({"email": session["user"]})
    if user:
        user.setdefault("resume_uploaded", False)
        user.setdefault("skills", [])
        user.setdefault("missing_skills", [])
        user.setdefault("skill_levels", [])
        user.setdefault("resume_score", 0)
        user.setdefault("job_fit", "Job Seeker")
        user.setdefault("interview_practiced", 0)
        user.setdefault("streak", 1)
        user.setdefault("ai_era_gaps", [])
        user.setdefault("readiness", 0)
        user.setdefault("selected_role", user.get("job_fit", "Job Seeker"))
        user.setdefault("ats_breakdown", {})
        user.setdefault("ats_grade", "N/A")
        user.setdefault("ats_color", "#6366f1")
        user.setdefault("ats_message", "Upload your resume to get your ATS score.")
    return user


def get_greeting():
    h = datetime.now().hour
    if h < 12:
        return "Good morning"
    if h < 17:
        return "Good afternoon"
    return "Good evening"


def get_now_str():
    return datetime.now().strftime("%a, %d %b %Y · %I:%M %p")


def get_app_counts(email):
    apps = list(applications_collection.find({"user_email": email}))
    counts = {"applied": 0, "interview": 0, "offer": 0, "rejected": 0, "saved": 0}
    for a in apps:
        s = a.get("status", "applied")
        counts[s] = counts.get(s, 0) + 1
    total = sum(v for k, v in counts.items() if k != "saved")
    return counts, total, apps

# =========================================================
# JOB DATA
# =========================================================

ALL_JOBS = [
    {
        "emoji": "🏢", "title": "Backend Developer", "company": "Infosys",
        "location": "Hyderabad", "type": "Full-time", "salary": "₹4–8 LPA",
        "skills": ["Python", "Django", "MongoDB", "SQL"],
        "desc": "Build scalable REST APIs and microservices using Python/Django stack with MongoDB.",
        "color": "linear-gradient(135deg,#6366f1,#4f46e5)",
        "company_career_url": "https://career.infosys.com/joblist",
        "naukri_search": "https://www.naukri.com/backend-developer-jobs-in-hyderabad",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Backend+Developer+Python+Django&location=Hyderabad",
        "role_tags": ["Backend Developer", "Full Stack Engineer", "Software Engineer"]
    },
    {
        "emoji": "🌐", "title": "Full Stack Engineer", "company": "Wipro",
        "location": "Bengaluru", "type": "Full-time", "salary": "₹6–12 LPA",
        "skills": ["React", "Node", "MongoDB", "JavaScript"],
        "desc": "Develop full-stack web applications with React frontend and Node.js backend.",
        "color": "linear-gradient(135deg,#22d3ee,#0891b2)",
        "company_career_url": "https://careers.wipro.com/careers-home/",
        "naukri_search": "https://www.naukri.com/full-stack-developer-jobs-in-bangalore",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Full+Stack+Engineer+React+Node&location=Bengaluru",
        "role_tags": ["Full Stack Engineer", "Frontend Developer", "Backend Developer"]
    },
    {
        "emoji": "🔧", "title": "Python Developer", "company": "TCS",
        "location": "Chennai", "type": "Full-time", "salary": "₹4–7 LPA",
        "skills": ["Python", "Flask", "SQL", "Git"],
        "desc": "Develop automation scripts and data-driven applications using Python and Flask.",
        "color": "linear-gradient(135deg,#f472b6,#db2777)",
        "company_career_url": "https://ibegin.tcs.com/iBegin/",
        "naukri_search": "https://www.naukri.com/python-developer-jobs-in-chennai",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Python+Developer+Flask&location=Chennai",
        "role_tags": ["Backend Developer", "Full Stack Engineer", "Data Scientist"]
    },
    {
        "emoji": "☁️", "title": "Cloud / DevOps Engineer", "company": "Accenture",
        "location": "Pune", "type": "Hybrid", "salary": "₹8–15 LPA",
        "skills": ["AWS", "Docker", "Kubernetes", "Linux"],
        "desc": "Manage cloud infrastructure on AWS, deploy containerized workloads with Kubernetes.",
        "color": "linear-gradient(135deg,#34d399,#059669)",
        "company_career_url": "https://www.accenture.com/in-en/careers/jobsearch",
        "naukri_search": "https://www.naukri.com/devops-engineer-jobs-in-pune",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=DevOps+Engineer+AWS+Kubernetes&location=Pune",
        "role_tags": ["DevOps Engineer"]
    },
    {
        "emoji": "📱", "title": "Frontend Developer", "company": "Cognizant",
        "location": "Remote", "type": "Remote", "salary": "₹5–10 LPA",
        "skills": ["React", "JavaScript", "CSS", "HTML"],
        "desc": "Build pixel-perfect, responsive UIs using React.js and modern CSS frameworks.",
        "color": "linear-gradient(135deg,#fbbf24,#d97706)",
        "company_career_url": "https://careers.cognizant.com/global/en/search-results",
        "naukri_search": "https://www.naukri.com/frontend-developer-jobs",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Frontend+Developer+React&f_WT=2",
        "role_tags": ["Frontend Developer", "Full Stack Engineer"]
    },
    {
        "emoji": "🤖", "title": "ML / AI Engineer", "company": "HCL Technologies",
        "location": "Hyderabad", "type": "Full-time", "salary": "₹8–18 LPA",
        "skills": ["Python", "TensorFlow", "Pandas", "NumPy"],
        "desc": "Design and deploy machine learning models for real-world business problems.",
        "color": "linear-gradient(135deg,#a78bfa,#7c3aed)",
        "company_career_url": "https://www.hcltech.com/careers",
        "naukri_search": "https://www.naukri.com/machine-learning-engineer-jobs-in-hyderabad",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=ML+Engineer+TensorFlow+Python&location=Hyderabad",
        "role_tags": ["ML Engineer", "Data Scientist"]
    },
    {
        "emoji": "📊", "title": "Data Analyst / Scientist", "company": "Capgemini",
        "location": "Mumbai", "type": "Hybrid", "salary": "₹6–12 LPA",
        "skills": ["Python", "SQL", "Pandas", "NumPy"],
        "desc": "Analyze large datasets, build dashboards, and produce actionable business insights.",
        "color": "linear-gradient(135deg,#22d3ee,#6366f1)",
        "company_career_url": "https://www.capgemini.com/in-en/careers/",
        "naukri_search": "https://www.naukri.com/data-scientist-jobs-in-mumbai",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Data+Scientist+Python+SQL&location=Mumbai",
        "role_tags": ["Data Scientist", "ML Engineer"]
    },
    {
        "emoji": "🎓", "title": "Software Engineer Intern", "company": "Startup (Various)",
        "location": "Remote / Hybrid", "type": "Internship", "salary": "₹15k–30k/month",
        "skills": ["Python", "JavaScript", "Git", "HTML"],
        "desc": "6-month internship opportunity for final year students across product startups.",
        "color": "linear-gradient(135deg,#f472b6,#fbbf24)",
        "company_career_url": "https://internshala.com/internships/computer-science-internship",
        "naukri_search": "https://internshala.com/internships/computer-science-internship",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Software+Engineer+Intern&f_E=1",
        "role_tags": ["Backend Developer", "Frontend Developer", "Full Stack Engineer", "Software Engineer"]
    },
    {
        "emoji": "🔒", "title": "Cyber Security Engineer", "company": "IBM India",
        "location": "Bengaluru", "type": "Full-time", "salary": "₹8–16 LPA",
        "skills": ["Linux", "Python", "SQL", "Docker"],
        "desc": "Protect systems and networks by implementing security measures, monitoring threats.",
        "color": "linear-gradient(135deg,#f87171,#dc2626)",
        "company_career_url": "https://www.ibm.com/in-en/employment/",
        "naukri_search": "https://www.naukri.com/cyber-security-engineer-jobs-in-bangalore",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Cyber+Security+Engineer&location=Bengaluru",
        "role_tags": ["Cyber Security Engineer", "DevOps Engineer"]
    },
    {
        "emoji": "📲", "title": "Android Developer", "company": "Mindtree",
        "location": "Hyderabad", "type": "Full-time", "salary": "₹5–10 LPA",
        "skills": ["Android", "Kotlin", "Java", "Firebase"],
        "desc": "Build feature-rich native Android apps with clean architecture and Jetpack Compose.",
        "color": "linear-gradient(135deg,#34d399,#6366f1)",
        "company_career_url": "https://www.ltimindtree.com/careers/",
        "naukri_search": "https://www.naukri.com/android-developer-jobs-in-hyderabad",
        "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Android+Developer+Kotlin&location=Hyderabad",
        "role_tags": ["Android Developer"]
    },
]


def compute_match(job_skills, user_skills):
    if not user_skills:
        return 40
    user_lower = [s.lower() for s in user_skills]
    matched = sum(1 for s in job_skills if s.lower() in user_lower)
    base = int((matched / max(len(job_skills), 1)) * 100)
    return max(min(base + 20, 99), 35)


def get_jobs_for_user(user_skills, job_fit=None):
    jobs = []
    top_skills = user_skills[:4] if user_skills else []
    for j in ALL_JOBS:
        match = compute_match(j["skills"], user_skills)
        if job_fit and job_fit in j["role_tags"]:
            match = min(match + 15, 99)
        job = dict(j)
        job["match"] = match
        user_lower = [s.lower() for s in user_skills]
        matched_skills = [s for s in j["skills"] if s.lower() in user_lower]
        query_skills = matched_skills[:2] if matched_skills else top_skills[:2]
        combined_enc = quote_plus(j["title"] + " " + " ".join(query_skills))
        loc_enc = quote_plus(j["location"].split("/")[0].strip())
        job["live_naukri"] = f"https://www.naukri.com/jobs?k={combined_enc}&l={loc_enc}"
        job["live_linkedin"] = f"https://www.linkedin.com/jobs/search/?keywords={combined_enc}&location={loc_enc}"
        job["live_indeed"] = f"https://www.indeed.com/jobs?q={combined_enc}&l=India"
        jobs.append(job)
    jobs.sort(key=lambda x: x["match"], reverse=True)
    return jobs

# =========================================================
# SKILL GAP DATA per role
# =========================================================

ROLE_REQUIREMENTS = {
    "Backend Developer": [
        {"name": "Python", "importance": 95, "level": "Advanced", "resource": "python.org/docs", "resource_url": "https://docs.python.org/3/tutorial/",
         "youtube": [{"title": "Python Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=rfscVS0vtbw"},
                     {"title": "Python for Beginners - Programming with Mosh", "url": "https://www.youtube.com/watch?v=kqtD5dpn9C8"}]},
        {"name": "Django", "importance": 85, "level": "Intermediate", "resource": "djangoproject.com", "resource_url": "https://docs.djangoproject.com/en/4.2/",
         "youtube": [{"title": "Django Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=e1IyzVyrLSU"},
                     {"title": "Django REST Framework Full Course", "url": "https://www.youtube.com/watch?v=c708Nf0cHrs"}]},
        {"name": "MongoDB", "importance": 80, "level": "Intermediate", "resource": "mongodb.com/docs", "resource_url": "https://www.mongodb.com/docs/",
         "youtube": [{"title": "MongoDB Crash Course - Web Dev Simplified", "url": "https://www.youtube.com/watch?v=ofme2o29ngU"},
                     {"title": "MongoDB Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=ExcRbA7fy_A"}]},
        {"name": "SQL", "importance": 78, "level": "Intermediate", "resource": "w3schools SQL", "resource_url": "https://www.w3schools.com/sql/",
         "youtube": [{"title": "SQL Tutorial Full - freeCodeCamp", "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY"},
                     {"title": "SQL for Data Analysis - Corey Schafer", "url": "https://www.youtube.com/watch?v=9yeOJ0ZMUYw"}]},
        {"name": "Docker", "importance": 70, "level": "Beginner", "resource": "docs.docker.com", "resource_url": "https://docs.docker.com/get-started/",
         "youtube": [{"title": "Docker Tutorial for Beginners - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=3c-iBn73dDE"},
                     {"title": "Docker Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=Kyx2PsuwomE"}]},
        {"name": "Git", "importance": 90, "level": "Intermediate", "resource": "git-scm.com", "resource_url": "https://git-scm.com/doc",
         "youtube": [{"title": "Git & GitHub Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=SWYqp7iY_Tc"},
                     {"title": "Git Tutorial for Beginners - Programming with Mosh", "url": "https://www.youtube.com/watch?v=8JJ101D3knE"}]},
        {"name": "Redis", "importance": 60, "level": "Beginner", "resource": "redis.io/docs", "resource_url": "https://redis.io/docs/",
         "youtube": [{"title": "Redis Crash Course - Web Dev Simplified", "url": "https://www.youtube.com/watch?v=jgpVdJB2sKQ"},
                     {"title": "Redis Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=XCsS_NVAa1g"}]},
    ],
    "Full Stack Engineer": [
        {"name": "React", "importance": 92, "level": "Advanced", "resource": "react.dev", "resource_url": "https://react.dev/learn",
         "youtube": [{"title": "React Full Course 2024 - Dave Gray", "url": "https://www.youtube.com/watch?v=RVFAyFWO4go"},
                     {"title": "React JS Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=w7ejDZ8SWv8"}]},
        {"name": "Node", "importance": 88, "level": "Intermediate", "resource": "nodejs.org", "resource_url": "https://nodejs.org/en/docs/",
         "youtube": [{"title": "Node.js and Express.js Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=Oe421EPjeBE"},
                     {"title": "Node.js Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=fBNz5xF-Kx4"}]},
        {"name": "MongoDB", "importance": 80, "level": "Intermediate", "resource": "mongodb.com/docs", "resource_url": "https://www.mongodb.com/docs/",
         "youtube": [{"title": "MongoDB Crash Course - Web Dev Simplified", "url": "https://www.youtube.com/watch?v=ofme2o29ngU"},
                     {"title": "MERN Stack Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=7CqJlxBYj-M"}]},
        {"name": "TypeScript", "importance": 75, "level": "Intermediate", "resource": "typescriptlang.org", "resource_url": "https://www.typescriptlang.org/docs/",
         "youtube": [{"title": "TypeScript Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=30LWjhZzg50"},
                     {"title": "TypeScript Tutorial - Net Ninja", "url": "https://www.youtube.com/watch?v=2pZmKW9-I_k"}]},
        {"name": "Docker", "importance": 65, "level": "Beginner", "resource": "docs.docker.com", "resource_url": "https://docs.docker.com/get-started/",
         "youtube": [{"title": "Docker Tutorial for Beginners - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=3c-iBn73dDE"},
                     {"title": "Docker Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=Kyx2PsuwomE"}]},
        {"name": "Git", "importance": 90, "level": "Intermediate", "resource": "git-scm.com", "resource_url": "https://git-scm.com/doc",
         "youtube": [{"title": "Git & GitHub Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=SWYqp7iY_Tc"},
                     {"title": "Git Tutorial for Beginners - Programming with Mosh", "url": "https://www.youtube.com/watch?v=8JJ101D3knE"}]},
    ],
    "Frontend Developer": [
        {"name": "React", "importance": 95, "level": "Advanced", "resource": "react.dev", "resource_url": "https://react.dev/learn",
         "youtube": [{"title": "React Full Course 2024 - Dave Gray", "url": "https://www.youtube.com/watch?v=RVFAyFWO4go"},
                     {"title": "React JS Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=w7ejDZ8SWv8"}]},
        {"name": "TypeScript", "importance": 85, "level": "Intermediate", "resource": "typescriptlang.org", "resource_url": "https://www.typescriptlang.org/docs/",
         "youtube": [{"title": "TypeScript Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=30LWjhZzg50"},
                     {"title": "TypeScript Tutorial - Net Ninja", "url": "https://www.youtube.com/watch?v=2pZmKW9-I_k"}]},
        {"name": "CSS", "importance": 80, "level": "Advanced", "resource": "css-tricks.com", "resource_url": "https://css-tricks.com/",
         "youtube": [{"title": "CSS Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=OXGznpKZ_sA"},
                     {"title": "Flexbox & Grid - Kevin Powell", "url": "https://www.youtube.com/watch?v=u044iM9xsWU"}]},
        {"name": "JavaScript", "importance": 95, "level": "Advanced", "resource": "javascript.info", "resource_url": "https://javascript.info/",
         "youtube": [{"title": "JavaScript Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=PkZNo7MFNFg"},
                     {"title": "JavaScript Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=hdI2bqOjy3c"}]},
        {"name": "Git", "importance": 85, "level": "Intermediate", "resource": "git-scm.com", "resource_url": "https://git-scm.com/doc",
         "youtube": [{"title": "Git & GitHub Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=SWYqp7iY_Tc"},
                     {"title": "Git Tutorial for Beginners - Programming with Mosh", "url": "https://www.youtube.com/watch?v=8JJ101D3knE"}]},
    ],
    "Data Scientist": [
        {"name": "Python", "importance": 98, "level": "Advanced", "resource": "python.org", "resource_url": "https://docs.python.org/3/",
         "youtube": [{"title": "Python for Data Science - freeCodeCamp", "url": "https://www.youtube.com/watch?v=LHBE6Q9XlzI"},
                     {"title": "Python Data Science Handbook", "url": "https://www.youtube.com/watch?v=vmEHCJofslg"}]},
        {"name": "Pandas", "importance": 92, "level": "Advanced", "resource": "pandas.pydata.org", "resource_url": "https://pandas.pydata.org/docs/",
         "youtube": [{"title": "Pandas Full Course - Keith Galli", "url": "https://www.youtube.com/watch?v=vmEHCJofslg"},
                     {"title": "Pandas Tutorial - Corey Schafer", "url": "https://www.youtube.com/watch?v=ZyhVh-qRZPA"}]},
        {"name": "NumPy", "importance": 88, "level": "Intermediate", "resource": "numpy.org/doc", "resource_url": "https://numpy.org/doc/",
         "youtube": [{"title": "NumPy Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=QUT1VHiLmmI"},
                     {"title": "NumPy Tutorial - CS Dojo", "url": "https://www.youtube.com/watch?v=GB9ByFAIAH4"}]},
        {"name": "TensorFlow", "importance": 80, "level": "Intermediate", "resource": "tensorflow.org", "resource_url": "https://www.tensorflow.org/learn",
         "youtube": [{"title": "TensorFlow 2.0 Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=tpCFfeUEGs8"},
                     {"title": "Deep Learning with TensorFlow - Sentdex", "url": "https://www.youtube.com/watch?v=wQ8BIBpya2k"}]},
        {"name": "SQL", "importance": 75, "level": "Intermediate", "resource": "mode analytics SQL", "resource_url": "https://mode.com/sql-tutorial/",
         "youtube": [{"title": "SQL Tutorial Full - freeCodeCamp", "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY"},
                     {"title": "SQL for Data Analysis - Corey Schafer", "url": "https://www.youtube.com/watch?v=9yeOJ0ZMUYw"}]},
    ],
    "DevOps Engineer": [
        {"name": "Docker", "importance": 95, "level": "Advanced", "resource": "docs.docker.com", "resource_url": "https://docs.docker.com/",
         "youtube": [{"title": "Docker Tutorial for Beginners - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=3c-iBn73dDE"},
                     {"title": "Docker in 1 Hour - Programming with Mosh", "url": "https://www.youtube.com/watch?v=pTFZFxd5hgI"}]},
        {"name": "Kubernetes", "importance": 90, "level": "Intermediate", "resource": "kubernetes.io/docs", "resource_url": "https://kubernetes.io/docs/home/",
         "youtube": [{"title": "Kubernetes Full Course - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=X48VuDVv0do"},
                     {"title": "Kubernetes Crash Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=s_o8dwzRlu4"}]},
        {"name": "AWS", "importance": 88, "level": "Intermediate", "resource": "aws.amazon.com/training", "resource_url": "https://aws.amazon.com/training/",
         "youtube": [{"title": "AWS Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=ubCNZFQZZVQ"},
                     {"title": "AWS Certified Cloud Practitioner - Andrew Brown", "url": "https://www.youtube.com/watch?v=SOTamWNgDKc"}]},
        {"name": "Linux", "importance": 85, "level": "Advanced", "resource": "linuxcommand.org", "resource_url": "https://linuxcommand.org/",
         "youtube": [{"title": "Linux Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=sWbUDq4S6Y8"},
                     {"title": "Linux Command Line Tutorial", "url": "https://www.youtube.com/watch?v=v_1zB2WNN14"}]},
        {"name": "Python", "importance": 70, "level": "Intermediate", "resource": "python.org", "resource_url": "https://docs.python.org/3/",
         "youtube": [{"title": "Python Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=rfscVS0vtbw"},
                     {"title": "Python for DevOps - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=4SZl1r2O_bY"}]},
        {"name": "Git", "importance": 88, "level": "Intermediate", "resource": "git-scm.com", "resource_url": "https://git-scm.com/doc",
         "youtube": [{"title": "Git & GitHub Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=SWYqp7iY_Tc"},
                     {"title": "Git Tutorial for Beginners - Programming with Mosh", "url": "https://www.youtube.com/watch?v=8JJ101D3knE"}]},
    ],
    "ML Engineer": [
        {"name": "Python", "importance": 98, "level": "Advanced", "resource": "python.org", "resource_url": "https://docs.python.org/3/",
         "youtube": [{"title": "Python for ML - Sentdex", "url": "https://www.youtube.com/watch?v=OGxgnH8y2NM"},
                     {"title": "Python for Data Science - freeCodeCamp", "url": "https://www.youtube.com/watch?v=LHBE6Q9XlzI"}]},
        {"name": "TensorFlow", "importance": 92, "level": "Advanced", "resource": "tensorflow.org", "resource_url": "https://www.tensorflow.org/learn",
         "youtube": [{"title": "TensorFlow 2.0 Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=tpCFfeUEGs8"},
                     {"title": "Neural Networks with TensorFlow - Sentdex", "url": "https://www.youtube.com/watch?v=wQ8BIBpya2k"}]},
        {"name": "NumPy", "importance": 88, "level": "Intermediate", "resource": "numpy.org", "resource_url": "https://numpy.org/doc/",
         "youtube": [{"title": "NumPy Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=QUT1VHiLmmI"},
                     {"title": "NumPy Tutorial - CS Dojo", "url": "https://www.youtube.com/watch?v=GB9ByFAIAH4"}]},
        {"name": "Pandas", "importance": 85, "level": "Intermediate", "resource": "pandas.pydata.org", "resource_url": "https://pandas.pydata.org/docs/",
         "youtube": [{"title": "Pandas Full Course - Keith Galli", "url": "https://www.youtube.com/watch?v=vmEHCJofslg"},
                     {"title": "Pandas Tutorial - Corey Schafer", "url": "https://www.youtube.com/watch?v=ZyhVh-qRZPA"}]},
        {"name": "Docker", "importance": 65, "level": "Beginner", "resource": "docs.docker.com", "resource_url": "https://docs.docker.com/get-started/",
         "youtube": [{"title": "Docker for ML Engineers - Abhishek Thakur", "url": "https://www.youtube.com/watch?v=0qG_0CPQhpg"},
                     {"title": "Docker Tutorial for Beginners - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=3c-iBn73dDE"}]},
    ],
    "Android Developer": [
        {"name": "Kotlin", "importance": 95, "level": "Advanced", "resource": "kotlinlang.org", "resource_url": "https://kotlinlang.org/docs/home.html",
         "youtube": [{"title": "Kotlin Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=F9UC9DY-vIU"},
                     {"title": "Kotlin for Android - Philipp Lackner", "url": "https://www.youtube.com/watch?v=SFGbMZsk4FI"}]},
        {"name": "Java", "importance": 80, "level": "Intermediate", "resource": "docs.oracle.com/java", "resource_url": "https://docs.oracle.com/en/java/",
         "youtube": [{"title": "Java Full Course - Programming with Mosh", "url": "https://www.youtube.com/watch?v=eIrMbAQSU34"},
                     {"title": "Java for Android - Coding in Flow", "url": "https://www.youtube.com/watch?v=fis26HvvDII"}]},
        {"name": "Firebase", "importance": 75, "level": "Intermediate", "resource": "firebase.google.com/docs", "resource_url": "https://firebase.google.com/docs",
         "youtube": [{"title": "Firebase Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=9kRgVxULbag"},
                     {"title": "Firebase with Android - Coding in Flow", "url": "https://www.youtube.com/watch?v=jbHfJpoOzkI"}]},
        {"name": "Android", "importance": 98, "level": "Advanced", "resource": "developer.android.com", "resource_url": "https://developer.android.com/docs",
         "youtube": [{"title": "Android Development for Beginners - freeCodeCamp", "url": "https://www.youtube.com/watch?v=fis26HvvDII"},
                     {"title": "Android Full Course - Philipp Lackner", "url": "https://www.youtube.com/watch?v=SFGbMZsk4FI"}]},
        {"name": "Git", "importance": 85, "level": "Intermediate", "resource": "git-scm.com", "resource_url": "https://git-scm.com/doc",
         "youtube": [{"title": "Git & GitHub Crash Course - Traversy Media", "url": "https://www.youtube.com/watch?v=SWYqp7iY_Tc"},
                     {"title": "Git Tutorial for Beginners - Programming with Mosh", "url": "https://www.youtube.com/watch?v=8JJ101D3knE"}]},
    ],
    "Cyber Security Engineer": [
        {"name": "Linux", "importance": 95, "level": "Advanced", "resource": "linuxcommand.org", "resource_url": "https://linuxcommand.org/",
         "youtube": [{"title": "Linux for Hackers - NetworkChuck", "url": "https://www.youtube.com/watch?v=VbEx7B_PTOE"},
                     {"title": "Linux Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=sWbUDq4S6Y8"}]},
        {"name": "Python", "importance": 85, "level": "Intermediate", "resource": "python.org", "resource_url": "https://docs.python.org/3/",
         "youtube": [{"title": "Ethical Hacking with Python - freeCodeCamp", "url": "https://www.youtube.com/watch?v=XWuP5Yf5ILI"},
                     {"title": "Python for Cybersecurity - TCM Security", "url": "https://www.youtube.com/watch?v=FxroHmHGDOY"}]},
        {"name": "SQL", "importance": 70, "level": "Intermediate", "resource": "w3schools SQL", "resource_url": "https://www.w3schools.com/sql/",
         "youtube": [{"title": "SQL Injection Tutorial - NetworkChuck", "url": "https://www.youtube.com/watch?v=1nJgupaUPEQ"},
                     {"title": "SQL Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY"}]},
        {"name": "Docker", "importance": 65, "level": "Beginner", "resource": "docs.docker.com", "resource_url": "https://docs.docker.com/",
         "youtube": [{"title": "Docker Tutorial for Beginners - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=3c-iBn73dDE"},
                     {"title": "Docker Security - NetworkChuck", "url": "https://www.youtube.com/watch?v=eGz9DS-aIeY"}]},
    ],
}

# =========================================================
# AI-ERA SKILL GAP DATA
# =========================================================

AI_ERA_SKILLS = {
    "Backend Developer": [
        {"name": "AI/ML API Integration", "why": "Modern backend apps must integrate AI APIs (OpenAI, Gemini, Claude) to add intelligent features.",
         "resource_url": "https://platform.openai.com/docs", "resource": "OpenAI Docs",
         "youtube": [{"title": "Build AI Apps with OpenAI API - freeCodeCamp", "url": "https://www.youtube.com/watch?v=c-g6epk3fFE"},
                     {"title": "FastAPI + LangChain Tutorial", "url": "https://www.youtube.com/watch?v=YFHEBIpF1nA"}]},
        {"name": "Vector Databases", "why": "Semantic search and RAG require vector DBs like Pinecone or Weaviate.",
         "resource_url": "https://docs.pinecone.io/", "resource": "Pinecone Docs",
         "youtube": [{"title": "Vector Databases Explained - Fireship", "url": "https://www.youtube.com/watch?v=klTvEwg3oJ4"},
                     {"title": "RAG with LangChain - Sam Witteveen", "url": "https://www.youtube.com/watch?v=sVcwVQRHIc8"}]},
        {"name": "Prompt Engineering", "why": "Backend devs must craft and manage system prompts for LLM pipelines.",
         "resource_url": "https://www.promptingguide.ai/", "resource": "Prompting Guide",
         "youtube": [{"title": "Prompt Engineering Guide - freeCodeCamp", "url": "https://www.youtube.com/watch?v=_ZvnD73m40o"},
                     {"title": "Advanced Prompt Engineering - Andrew Ng", "url": "https://www.youtube.com/watch?v=ahnGLM-RC1Y"}]},
    ],
    "Full Stack Engineer": [
        {"name": "AI/LLM Integration", "why": "Full stack apps now require integrating AI chat, summarization, and generation features.",
         "resource_url": "https://js.langchain.com/docs/", "resource": "LangChain JS Docs",
         "youtube": [{"title": "Build AI Fullstack App - Fireship", "url": "https://www.youtube.com/watch?v=ffEDkqfIzxM"},
                     {"title": "Next.js AI Chatbot - Vercel", "url": "https://www.youtube.com/watch?v=O7NnT_NFJZE"}]},
        {"name": "WebSockets & Realtime", "why": "AI apps require streaming responses and real-time collaboration.",
         "resource_url": "https://socket.io/docs/v4/", "resource": "Socket.IO Docs",
         "youtube": [{"title": "WebSocket Full Tutorial - Traversy Media", "url": "https://www.youtube.com/watch?v=pnj3Jbho5Ck"},
                     {"title": "Real-time Apps with Socket.IO - Fireship", "url": "https://www.youtube.com/watch?v=ZKEqqIO7n-k"}]},
        {"name": "Prompt Engineering", "why": "Building full-stack AI features demands skill in writing effective prompts.",
         "resource_url": "https://www.promptingguide.ai/", "resource": "Prompting Guide",
         "youtube": [{"title": "Prompt Engineering Guide - freeCodeCamp", "url": "https://www.youtube.com/watch?v=_ZvnD73m40o"},
                     {"title": "Prompt Engineering Tips - Matt Wolfe", "url": "https://www.youtube.com/watch?v=1bUy-1hGZpI"}]},
    ],
    "Frontend Developer": [
        {"name": "AI UI Components", "why": "Modern frontends include AI chat interfaces, auto-complete, and generative UI elements.",
         "resource_url": "https://sdk.vercel.ai/docs", "resource": "Vercel AI SDK",
         "youtube": [{"title": "Build AI Chat UI with Next.js - Vercel", "url": "https://www.youtube.com/watch?v=O7NnT_NFJZE"},
                     {"title": "React AI Components - Fireship", "url": "https://www.youtube.com/watch?v=ffEDkqfIzxM"}]},
        {"name": "Accessibility (a11y)", "why": "AI tools are used by everyone — accessibility is now a hiring differentiator.",
         "resource_url": "https://web.dev/accessibility/", "resource": "web.dev Accessibility",
         "youtube": [{"title": "Web Accessibility Tutorial - freeCodeCamp", "url": "https://www.youtube.com/watch?v=e2nkq3h1P68"},
                     {"title": "A11y for Frontend Devs - Kevin Powell", "url": "https://www.youtube.com/watch?v=qr0ujkLLgmE"}]},
        {"name": "Performance Optimization", "why": "AI-heavy apps demand advanced performance skills: lazy loading, caching, Web Workers.",
         "resource_url": "https://web.dev/performance/", "resource": "web.dev Performance",
         "youtube": [{"title": "Web Performance 2024 - Fireship", "url": "https://www.youtube.com/watch?v=0fONene3OIA"},
                     {"title": "React Performance Optimization - Jack Herrington", "url": "https://www.youtube.com/watch?v=VYkMnKpEuAk"}]},
    ],
    "Data Scientist": [
        {"name": "LLMs & Foundation Models", "why": "Data scientists must now fine-tune, evaluate, and deploy large language models.",
         "resource_url": "https://huggingface.co/docs", "resource": "HuggingFace Docs",
         "youtube": [{"title": "LLMs from Scratch - Sebastian Raschka", "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY"},
                     {"title": "Fine-tuning LLMs - HuggingFace", "url": "https://www.youtube.com/watch?v=eC6Hd1hFvos"}]},
        {"name": "MLOps", "why": "Deploying and monitoring ML models in production is now essential for every data scientist.",
         "resource_url": "https://mlflow.org/docs/latest/index.html", "resource": "MLflow Docs",
         "youtube": [{"title": "MLOps Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=9BgIDqAzfuA"},
                     {"title": "MLOps Explained - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=fXKmLfQk5Vw"}]},
        {"name": "Prompt Engineering for Data", "why": "Querying LLMs with text-to-SQL and data analysis prompts is a new core data skill.",
         "resource_url": "https://www.promptingguide.ai/", "resource": "Prompting Guide",
         "youtube": [{"title": "ChatGPT for Data Analysis - Alex the Analyst", "url": "https://www.youtube.com/watch?v=C75TROiiEa0"},
                     {"title": "Text-to-SQL with LLMs - Data with Mo", "url": "https://www.youtube.com/watch?v=GVSBvJBNFEg"}]},
    ],
    "DevOps Engineer": [
        {"name": "AI-Assisted DevOps (AIOps)", "why": "AI tools like GitHub Copilot and anomaly detection are transforming DevOps.",
         "resource_url": "https://github.com/features/copilot", "resource": "GitHub Copilot",
         "youtube": [{"title": "AIOps Explained - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=xhBmGLlJPZs"},
                     {"title": "GitHub Copilot for DevOps - DevOps Toolkit", "url": "https://www.youtube.com/watch?v=RDd71IUIgpg"}]},
        {"name": "Infrastructure as Code (Terraform)", "why": "Terraform is the gold standard for managing cloud infrastructure with code.",
         "resource_url": "https://developer.hashicorp.com/terraform/docs", "resource": "Terraform Docs",
         "youtube": [{"title": "Terraform Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=SLB_c_ayRMo"},
                     {"title": "Terraform Tutorial - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=l5k1ai_GBDE"}]},
        {"name": "Observability & Monitoring", "why": "Modern systems need distributed tracing, metrics, and logs at scale using tools like Grafana.",
         "resource_url": "https://grafana.com/docs/", "resource": "Grafana Docs",
         "youtube": [{"title": "Observability with Grafana - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=yWc3xgRsezA"},
                     {"title": "Prometheus & Grafana Tutorial - freeCodeCamp", "url": "https://www.youtube.com/watch?v=9TJx7QTrTyo"}]},
    ],
    "ML Engineer": [
        {"name": "LLMs & Transformers", "why": "Transformer architecture and large language models are the backbone of modern AI systems.",
         "resource_url": "https://huggingface.co/docs/transformers", "resource": "HuggingFace Transformers",
         "youtube": [{"title": "Transformers from Scratch - Andrej Karpathy", "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY"},
                     {"title": "LLM Fine-tuning Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=eC6Hd1hFvos"}]},
        {"name": "MLOps & Model Deployment", "why": "ML engineers must ship models to production — containerization, serving, and monitoring.",
         "resource_url": "https://mlflow.org/docs/latest/index.html", "resource": "MLflow Docs",
         "youtube": [{"title": "MLOps Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=9BgIDqAzfuA"},
                     {"title": "Deploy ML Models - Krish Naik", "url": "https://www.youtube.com/watch?v=IbHJ4pY-Hg0"}]},
        {"name": "RAG & Vector Search", "why": "Retrieval-Augmented Generation is the dominant pattern for enterprise AI applications.",
         "resource_url": "https://python.langchain.com/docs/", "resource": "LangChain Docs",
         "youtube": [{"title": "RAG from Scratch - LangChain", "url": "https://www.youtube.com/watch?v=sVcwVQRHIc8"},
                     {"title": "Vector Databases & RAG - Fireship", "url": "https://www.youtube.com/watch?v=klTvEwg3oJ4"}]},
    ],
    "Android Developer": [
        {"name": "AI on Device (TFLite / ML Kit)", "why": "On-device AI for image, speech, and text processing is now expected in mobile apps.",
         "resource_url": "https://developers.google.com/ml-kit", "resource": "Google ML Kit",
         "youtube": [{"title": "ML Kit for Android - Android Developers", "url": "https://www.youtube.com/watch?v=ejrn_JHksws"},
                     {"title": "TensorFlow Lite Android - CodeWithChris", "url": "https://www.youtube.com/watch?v=R14wKkIHNRM"}]},
        {"name": "Jetpack Compose", "why": "Jetpack Compose is now the standard for building modern Android UIs.",
         "resource_url": "https://developer.android.com/jetpack/compose", "resource": "Compose Docs",
         "youtube": [{"title": "Jetpack Compose Full Course - Philipp Lackner", "url": "https://www.youtube.com/watch?v=cDabx3SjuOY"},
                     {"title": "Compose for Beginners - Android Developers", "url": "https://www.youtube.com/watch?v=qvDi0T3vph0"}]},
        {"name": "Kotlin Coroutines & Flow", "why": "Async programming with Coroutines is essential for smooth, performant Android apps.",
         "resource_url": "https://kotlinlang.org/docs/coroutines-overview.html", "resource": "Kotlin Coroutines",
         "youtube": [{"title": "Kotlin Coroutines Full Course - Philipp Lackner", "url": "https://www.youtube.com/watch?v=ShNhJ3wMpvQ"},
                     {"title": "Coroutines & Flow - Android Developers", "url": "https://www.youtube.com/watch?v=emk9_tVVLcc"}]},
    ],
    "Cyber Security Engineer": [
        {"name": "AI-Powered Threat Detection", "why": "Modern SOC teams use AI/ML to detect anomalies and automate threat response.",
         "resource_url": "https://www.coursera.org/learn/ai-for-cybersecurity", "resource": "AI for CyberSec - Coursera",
         "youtube": [{"title": "AI in Cybersecurity - David Bombal", "url": "https://www.youtube.com/watch?v=RXGHFr0d2WU"},
                     {"title": "Machine Learning for Security - freeCodeCamp", "url": "https://www.youtube.com/watch?v=YjVBZa5jGdA"}]},
        {"name": "Cloud Security (AWS/GCP/Azure)", "why": "Most infrastructure is cloud-based — securing cloud environments is a top priority skill.",
         "resource_url": "https://aws.amazon.com/security/", "resource": "AWS Security",
         "youtube": [{"title": "AWS Security Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=y8cbKJAo3B4"},
                     {"title": "Cloud Security Best Practices - NetworkChuck", "url": "https://www.youtube.com/watch?v=M4iAJNAJFIQ"}]},
        {"name": "Penetration Testing", "why": "Ethical hacking and pen testing skills are critical for modern security roles.",
         "resource_url": "https://www.offensive-security.com/", "resource": "Offensive Security",
         "youtube": [{"title": "Ethical Hacking Full Course - freeCodeCamp", "url": "https://www.youtube.com/watch?v=3Kq1MIfTWCE"},
                     {"title": "Pen Testing Tutorials - TCM Security", "url": "https://www.youtube.com/watch?v=fNzpcB7ODxQ"}]},
    ],
}

ROLE_ROADMAPS = {
    "Backend Developer": [
        {"week": "Week 1–2", "title": "🐍 Python & OOP", "desc": "Strengthen Python, OOP, file I/O, exceptions.", "color": "#6366f1",
         "resources": [{"name": "Python Docs", "url": "https://docs.python.org/3/tutorial/"}, {"name": "Automate the Boring Stuff", "url": "https://automatetheboringstuff.com/"}]},
        {"week": "Week 3–4", "title": "🌐 Django & REST APIs", "desc": "Build REST APIs with Django REST Framework.", "color": "#22d3ee",
         "resources": [{"name": "Django Docs", "url": "https://docs.djangoproject.com/"}, {"name": "DRF Tutorial", "url": "https://www.django-rest-framework.org/tutorial/quickstart/"}]},
        {"week": "Week 5–6", "title": "🗄️ MongoDB & SQL", "desc": "CRUD operations, aggregation, query optimization.", "color": "#34d399",
         "resources": [{"name": "MongoDB University", "url": "https://university.mongodb.com/"}, {"name": "SQLZoo", "url": "https://sqlzoo.net/"}]},
        {"week": "Week 7–8", "title": "🐳 Docker & Git", "desc": "Containerize apps, git branching strategies.", "color": "#f472b6",
         "resources": [{"name": "Docker Get Started", "url": "https://docs.docker.com/get-started/"}, {"name": "Git Branching", "url": "https://learngitbranching.js.org/"}]},
    ],
    "Full Stack Engineer": [
        {"week": "Week 1–2", "title": "⚛️ React Fundamentals", "desc": "Hooks, state, props, component lifecycle.", "color": "#61dafb",
         "resources": [{"name": "React Docs", "url": "https://react.dev/learn"}, {"name": "Full Stack Open", "url": "https://fullstackopen.com/"}]},
        {"week": "Week 3–4", "title": "🟢 Node.js & Express", "desc": "REST API, middleware, auth, JWT tokens.", "color": "#68a063",
         "resources": [{"name": "Node Docs", "url": "https://nodejs.org/en/docs/"}, {"name": "Express.js Guide", "url": "https://expressjs.com/en/guide/routing.html"}]},
        {"week": "Week 5–6", "title": "🗄️ Database & APIs", "desc": "MongoDB, Mongoose, connecting front+back.", "color": "#34d399",
         "resources": [{"name": "MongoDB Atlas", "url": "https://www.mongodb.com/atlas"}, {"name": "Mongoose Docs", "url": "https://mongoosejs.com/docs/"}]},
        {"week": "Week 7–8", "title": "🚀 Deploy & Interview", "desc": "Deploy on Vercel/Render, practice questions.", "color": "#6366f1",
         "resources": [{"name": "Vercel Deploy", "url": "https://vercel.com/docs"}, {"name": "System Design Primer", "url": "https://github.com/donnemartin/system-design-primer"}]},
    ],
    "Data Scientist": [
        {"week": "Week 1–2", "title": "🐍 Python for Data", "desc": "NumPy, Pandas, data cleaning, EDA.", "color": "#fbbf24",
         "resources": [{"name": "Pandas Docs", "url": "https://pandas.pydata.org/docs/"}, {"name": "Kaggle Learn", "url": "https://www.kaggle.com/learn"}]},
        {"week": "Week 3–4", "title": "📊 Statistics & SQL", "desc": "Hypothesis testing, A/B testing, SQL queries.", "color": "#22d3ee",
         "resources": [{"name": "Khan Academy Stats", "url": "https://www.khanacademy.org/math/statistics-probability"}, {"name": "Mode SQL Tutorial", "url": "https://mode.com/sql-tutorial/"}]},
        {"week": "Week 5–6", "title": "🤖 ML Algorithms", "desc": "Supervised, unsupervised, model evaluation.", "color": "#a78bfa",
         "resources": [{"name": "Scikit-learn Docs", "url": "https://scikit-learn.org/stable/"}, {"name": "Fast.ai", "url": "https://www.fast.ai/"}]},
        {"week": "Week 7–8", "title": "🧠 Deep Learning", "desc": "Neural nets, CNNs, TensorFlow basics.", "color": "#f472b6",
         "resources": [{"name": "TensorFlow Tutorials", "url": "https://www.tensorflow.org/tutorials"}, {"name": "Deep Learning Book", "url": "https://www.deeplearningbook.org/"}]},
    ],
}


def get_default_roadmap(role):
    if role in ROLE_ROADMAPS:
        return ROLE_ROADMAPS[role]
    return [
        {"week": "Week 1–2", "title": "📚 Core Fundamentals", "desc": "Strengthen your core skills for this role.", "color": "#6366f1", "resources": []},
        {"week": "Week 3–4", "title": "🔧 Build Projects", "desc": "Apply skills in real mini-projects.", "color": "#22d3ee", "resources": []},
        {"week": "Week 5–6", "title": "🧪 Practice Problems", "desc": "DSA, system design exercises.", "color": "#34d399", "resources": []},
        {"week": "Week 7–8", "title": "🎤 Mock Interviews", "desc": "Practice with real interview questions.", "color": "#f472b6", "resources": []},
    ]

# =========================================================
# INTERVIEW QUESTIONS
# =========================================================

INTERVIEW_QUESTIONS = {
    "Backend Developer": {
        "All Companies": [
            {"question": "What is REST and what are its core principles?", "hint": "Stateless, client-server, uniform interface, cacheable, layered system. REST uses HTTP verbs (GET, POST, PUT, DELETE).", "difficulty": "Easy", "category": "Web", "company": "All Companies"},
            {"question": "What is the difference between SQL and NoSQL databases? When would you use each?", "hint": "SQL: structured schema, ACID, relational (MySQL, PostgreSQL). NoSQL: flexible schema, horizontal scaling (MongoDB, Redis).", "difficulty": "Medium", "category": "Database", "company": "All Companies"},
            {"question": "Explain Python decorators with a practical example.", "hint": "Decorators wrap a function. Example: @login_required in Flask. They use closures and functools.wraps.", "difficulty": "Medium", "category": "Python", "company": "All Companies"},
            {"question": "What are ACID properties in databases? Why do they matter?", "hint": "Atomicity: all-or-nothing. Consistency: valid state always. Isolation: concurrent transactions don't interfere. Durability: committed data persists.", "difficulty": "Hard", "category": "Database", "company": "All Companies"},
            {"question": "How does indexing work in MongoDB? What types of indexes are available?", "hint": "Index stores subset of data in traversable form. Types: single field, compound, multikey, text, 2dsphere, hashed.", "difficulty": "Medium", "category": "MongoDB", "company": "All Companies"},
            {"question": "What is JWT? How does authentication work with JWT tokens?", "hint": "JSON Web Token has 3 parts: header.payload.signature. Server creates JWT on login, client sends in Authorization: Bearer header.", "difficulty": "Medium", "category": "Security", "company": "All Companies"},
        ],
        "Google": [
            {"question": "Design a system to handle 1 million requests per second. How would you scale?", "hint": "Load balancer → multiple app servers. Caching with Redis. Database read replicas. CDN for static assets. Message queues for async.", "difficulty": "Hard", "category": "System Design", "company": "Google", "year": "2024"},
            {"question": "Given an array of integers, find two numbers that add up to a target. Optimize for time complexity.", "hint": "Brute force O(n²). Optimal: HashMap approach O(n). Single pass: for each num, check if (target-num) in map.", "difficulty": "Easy", "category": "DSA", "company": "Google", "year": "2024"},
            {"question": "How would you design Google's URL shortener (bit.ly)?", "hint": "Hash long URL to 6-8 char code (base62). Store in DB. Redirect: lookup code → 301 redirect. Scale: Redis cache for hot URLs.", "difficulty": "Hard", "category": "System Design", "company": "Google", "year": "2023"},
            {"question": "Explain the difference between process and thread. How does Python handle concurrency?", "hint": "Process: separate memory space. Thread: shared memory. Python GIL limits true multi-threading. Use multiprocessing for CPU-bound, asyncio for I/O-bound.", "difficulty": "Medium", "category": "OS Concepts", "company": "Google", "year": "2023"},
        ],
        "Amazon": [
            {"question": "Tell me about a time you had to make a technical decision with incomplete information. (Leadership Principle: Bias for Action)", "hint": "Use STAR method: Situation, Task, Action, Result. Show you gathered available data, made a decision, measured outcome, iterated.", "difficulty": "Medium", "category": "Behavioral", "company": "Amazon", "year": "2024"},
            {"question": "Design Amazon's product recommendation system.", "hint": "Collaborative filtering. Real-time: event stream (purchases, clicks) → Kafka → ML model → Redis cache → API. Batch: daily retrain.", "difficulty": "Hard", "category": "System Design", "company": "Amazon", "year": "2024"},
            {"question": "How would you detect and prevent fraud in an e-commerce system at Amazon's scale?", "hint": "Real-time: rule engine (IP, velocity, device fingerprint) + ML model score. Batch: graph analysis for account networks.", "difficulty": "Hard", "category": "System Design", "company": "Amazon", "year": "2023"},
        ],
        "Microsoft": [
            {"question": "How would you implement a LRU Cache? Write the code.", "hint": "Use OrderedDict (Python) or doubly linked list + hashmap. O(1) get and put.", "difficulty": "Medium", "category": "DSA", "company": "Microsoft", "year": "2024"},
            {"question": "Design a distributed key-value store like Redis.", "hint": "In-memory storage. Persistence: RDB snapshots + AOF log. Replication: master-slave. Clustering: consistent hashing.", "difficulty": "Hard", "category": "System Design", "company": "Microsoft", "year": "2024"},
            {"question": "What is the difference between authentication and authorization? How would you implement RBAC?", "hint": "Auth: who are you (identity). Authz: what can you do (permissions). RBAC: User → Roles → Permissions.", "difficulty": "Medium", "category": "Security", "company": "Microsoft", "year": "2023"},
        ],
        "Infosys": [
            {"question": "Explain the MVC architecture pattern with a Django example.", "hint": "Django is actually MVT: Model (database), View (logic/controller), Template (presentation).", "difficulty": "Easy", "category": "Framework", "company": "Infosys", "year": "2024"},
            {"question": "What is the difference between GET and POST methods? When to use each?", "hint": "GET: retrieve data, params in URL, idempotent, cacheable. POST: send data in body, not idempotent, used for creating resources.", "difficulty": "Easy", "category": "Web", "company": "Infosys", "year": "2024"},
        ],
        "TCS": [
            {"question": "What are Python list comprehensions? Give examples with filtering and mapping.", "hint": "[expr for item in iterable if condition]. Filter: [x for x in nums if x > 0]. Map: [x*2 for x in nums].", "difficulty": "Easy", "category": "Python", "company": "TCS", "year": "2024"},
            {"question": "Explain the concept of ORM and how Django ORM simplifies database operations.", "hint": "ORM maps Python classes to DB tables. Handles connections, escaping (prevents SQL injection), migrations.", "difficulty": "Easy", "category": "Django", "company": "TCS", "year": "2024"},
        ],
        "Wipro": [
            {"question": "What is the difference between synchronous and asynchronous programming?", "hint": "Sync: blocking, one task completes before next. Async: non-blocking, tasks can run concurrently. Python asyncio: async def, await.", "difficulty": "Medium", "category": "Python", "company": "Wipro", "year": "2024"},
            {"question": "How would you handle errors and exceptions in a production Flask application?", "hint": "@app.errorhandler(404). Use try/except with specific exceptions. Log errors with Python logging module.", "difficulty": "Medium", "category": "Flask", "company": "Wipro", "year": "2023"},
        ],
    },
    "Full Stack Engineer": {
        "All Companies": [
            {"question": "What is the difference between SSR and CSR?", "hint": "SSR: HTML generated on server, better SEO, faster initial paint. CSR: React renders in browser, slower initial load but faster interactions.", "difficulty": "Medium", "category": "Architecture", "company": "All Companies"},
            {"question": "Explain the React virtual DOM and reconciliation process.", "hint": "Virtual DOM is a lightweight JS object copy of real DOM. On state change, React re-renders vDOM, diffs it, batches updates to real DOM.", "difficulty": "Medium", "category": "React", "company": "All Companies"},
            {"question": "What is CORS and how do you handle it?", "hint": "Cross-Origin Resource Sharing. Browser blocks requests to different origins. Server sets Access-Control-Allow-Origin header.", "difficulty": "Easy", "category": "Web", "company": "All Companies"},
        ],
        "Flipkart": [
            {"question": "Design Flipkart's product search with filters, sorting and pagination.", "hint": "Elasticsearch for full-text search + filtering. Cache popular queries in Redis.", "difficulty": "Hard", "category": "System Design", "company": "Flipkart", "year": "2024"},
            {"question": "How would you implement real-time inventory tracking at scale?", "hint": "Decrement on add-to-cart (with TTL for cart expiry). Use Redis atomic DECR for concurrency.", "difficulty": "Hard", "category": "System Design", "company": "Flipkart", "year": "2023"},
        ],
        "Swiggy": [
            {"question": "Design Swiggy's real-time order tracking system.", "hint": "GPS events from delivery partner app → WebSocket/SSE to customer. Store latest location in Redis. Pub/sub for updates.", "difficulty": "Hard", "category": "System Design", "company": "Swiggy", "year": "2024"},
        ],
        "Zomato": [
            {"question": "Design Zomato's restaurant recommendation system.", "hint": "Collaborative filtering. Location-based: nearby restaurants ranked by rating, delivery time. ML features: time of day, weather, past orders.", "difficulty": "Hard", "category": "System Design", "company": "Zomato", "year": "2024"},
        ],
    },
    "Frontend Developer": {
        "All Companies": [
            {"question": "What is the event loop in JavaScript? How does async/await work under the hood?", "hint": "JS is single-threaded. Event loop: call stack + callback queue + microtask queue. async/await is syntactic sugar for Promises.", "difficulty": "Hard", "category": "JavaScript", "company": "All Companies"},
            {"question": "Explain React's useEffect hook — when does it run and how to avoid infinite loops?", "hint": "useEffect(fn, deps). Runs after render. Empty array []: run once on mount. Deps array: run when deps change.", "difficulty": "Medium", "category": "React", "company": "All Companies"},
            {"question": "What is CSS specificity and how are conflicts resolved?", "hint": "Specificity order: !important > inline > ID(100) > class(10) > element(1). Same specificity: last rule wins.", "difficulty": "Easy", "category": "CSS", "company": "All Companies"},
        ],
        "Google": [
            {"question": "How would you optimize a React app with 10,000 list items for performance?", "hint": "Virtual scrolling (react-window). useMemo/useCallback. React.memo for pure components. Code splitting with lazy().", "difficulty": "Hard", "category": "Performance", "company": "Google", "year": "2024"},
        ],
    },
    "Data Scientist": {
        "All Companies": [
            {"question": "What is overfitting and how do you prevent it?", "hint": "Model memorizes training data, fails on test data. Prevention: regularization (L1/L2), dropout, cross-validation, more training data.", "difficulty": "Medium", "category": "ML", "company": "All Companies"},
            {"question": "Explain precision, recall, F1-score and when to prioritize each.", "hint": "Precision = TP/(TP+FP): when FP is costly. Recall = TP/(TP+FN): when FN is costly. F1 = harmonic mean: balanced.", "difficulty": "Medium", "category": "Metrics", "company": "All Companies"},
            {"question": "What is the bias-variance tradeoff?", "hint": "Bias: error from wrong assumptions (underfitting). Variance: error from sensitivity to training data (overfitting).", "difficulty": "Medium", "category": "ML Theory", "company": "All Companies"},
        ],
    },
    "DevOps Engineer": {
        "All Companies": [
            {"question": "What is the difference between Docker and a virtual machine?", "hint": "VM: full OS, hypervisor, heavy (GBs), slow startup. Docker: shares host OS kernel, lightweight (MBs), fast startup.", "difficulty": "Easy", "category": "Docker", "company": "All Companies"},
            {"question": "Explain CI/CD pipeline. What tools would you use?", "hint": "CI: lint, test, build on every commit. CD: deploy to staging (automated), deploy to prod. Tools: Docker, Kubernetes, Terraform, ArgoCD.", "difficulty": "Medium", "category": "CI/CD", "company": "All Companies"},
            {"question": "How does Kubernetes achieve high availability?", "hint": "Multiple master nodes with etcd quorum. ReplicaSets: maintain desired pod count. HPA: auto-scale on CPU/memory. Rolling updates.", "difficulty": "Hard", "category": "Kubernetes", "company": "All Companies"},
        ],
    },
    "ML Engineer": {
        "All Companies": [
            {"question": "Explain gradient descent and its variants (SGD, Adam, RMSprop).", "hint": "GD: update weights in direction of -gradient. Mini-batch: middle ground. Adam: adaptive learning rates + momentum.", "difficulty": "Medium", "category": "Optimization", "company": "All Companies"},
            {"question": "What is the difference between L1 and L2 regularization?", "hint": "L1 (Lasso): adds sum of |weights|, produces sparse models. L2 (Ridge): adds sum of weights², distributes weights evenly.", "difficulty": "Medium", "category": "Regularization", "company": "All Companies"},
        ],
    },
}


def get_questions(role, company, difficulty):
    role_qs = INTERVIEW_QUESTIONS.get(role, INTERVIEW_QUESTIONS.get("Backend Developer", {}))
    questions = list(role_qs.get("All Companies", []))
    if company != "All Companies":
        questions = list(role_qs.get(company, [])) + questions
    if difficulty != "All":
        questions = [q for q in questions if q.get("difficulty") == difficulty]
    return questions

# =========================================================
# PUBLIC ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        if users_collection.find_one({"email": email}):
            flash("Email already registered.", "error")
            return redirect("/register")
        users_collection.insert_one({
            "name": name, "email": email,
            "password": generate_password_hash(password),
            "resume_uploaded": False, "skills": [], "missing_skills": [],
            "skill_levels": [], "resume_score": 0, "job_fit": "Job Seeker",
            "bio": "", "preferred_location": "", "expected_salary": "",
            "preferred_job_type": "Any", "job_alerts": True, "skill_tips": True,
            "interview_reminders": True, "application_updates": True,
            "interview_practiced": 0, "streak": 1, "created_at": datetime.now()
        })
        flash("Account created! Please login.", "success")
        return redirect("/login")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = users_collection.find_one({"email": email})
        if not user or not check_password_hash(user["password"], password):
            flash("Invalid email or password.", "error")
            return redirect("/login")
        session["user"] = email
        return redirect("/dashboard")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
@login_required
def dashboard():
    user = get_current_user()
    app_counts, total_applications, _ = get_app_counts(session["user"])
    top_jobs = []
    if user["resume_uploaded"]:
        all_matched = get_jobs_for_user(user["skills"], user.get("job_fit"))
        top_jobs = all_matched[:3]
    return render_template("dashboard.html",
        user=user, total_applications=total_applications,
        app_counts=app_counts, top_jobs=top_jobs,
        greeting=get_greeting(), now=get_now_str()
    )

# =========================================================
# RESUME ANALYZER
# =========================================================

@app.route("/resume_analyzer")
@login_required
def resume_analyzer():
    user = get_current_user()
    return render_template("resume_analyzer.html", user=user)


@app.route("/upload_resume", methods=["GET", "POST"])
@login_required
@limiter.limit("10 per hour")
def upload_resume():
    user = get_current_user()
    if request.method == "POST":
        file = request.files.get("resume")
        if not file or file.filename == "":
            flash("No file selected.", "error")
            return redirect("/resume_analyzer")
        if not allowed_file(file.filename):
            flash("Only PDF, DOC, DOCX allowed.", "error")
            return redirect("/resume_analyzer")

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        try:
            text = extract_text_from_file(path)

            # ── Resume name validation (robust for short names) ──
            user_name = user.get("name", "").lower().strip()
            name_parts = [p for p in user_name.split() if len(p) > 2]
            text_lower_check = text[:2000].lower()
            if name_parts:
                name_matched = any(part in text_lower_check for part in name_parts)
                if not name_matched:
                    flash(
                        f"⚠️ Resume name mismatch! The resume doesn't seem to belong to "
                        f"'{user.get('name')}'. Please upload your own resume. "
                        f"(Tip: Your name should appear in the resume header.)",
                        "error"
                    )
                    return redirect("/resume_analyzer")

            skills = extract_skills(text)
            job_fit = determine_job_fit(skills)

            skill_levels = []
            for skill in skills:
                freq = text.lower().count(skill.lower())
                level = min(50 + (freq * 8), 95)
                skill_levels.append(level)

            required = ROLE_REQUIREMENTS.get(job_fit, [])
            user_lower = [s.lower() for s in skills]
            missing = [r["name"] for r in required if r["name"].lower() not in user_lower]

            ats_total, ats_breakdown = compute_ats_score(text, skills, job_fit)
            ats_grade, ats_color, ats_message = get_ats_grade(ats_total)

            users_collection.update_one(
                {"email": session["user"]},
                {"$set": {
                    "resume_uploaded": True, "skills": skills,
                    "missing_skills": missing, "skill_levels": skill_levels,
                    "resume_score": ats_total, "ats_breakdown": ats_breakdown,
                    "ats_grade": ats_grade, "ats_color": ats_color,
                    "ats_message": ats_message, "job_fit": job_fit,
                    "resume_filename": filename,
                    "resume_uploaded_at": datetime.now()
                }}
            )
            flash(f"Resume analyzed! {len(skills)} skills extracted. Check your AI-Era Skill Gap below!", "success")
            return redirect("/skill_gap")

        finally:
            if os.path.exists(path):
                os.remove(path)

    return render_template("upload_resume.html", user=user)

# =========================================================
# JOB MATCHES — with pagination
# =========================================================

@app.route("/job_matches")
@login_required
def job_matches():
    user = get_current_user()
    page = request.args.get("page", 1, type=int)
    per_page = 6
    all_jobs = get_jobs_for_user(user["skills"], user.get("job_fit"))
    total = len(all_jobs)
    jobs = all_jobs[(page - 1) * per_page: page * per_page]
    total_pages = (total + per_page - 1) // per_page
    return render_template("job_matches.html", user=user, jobs=jobs,
                           page=page, total_pages=total_pages)


@app.route("/apply_job", methods=["POST"])
@login_required
def apply_job():
    job_title = request.form.get("job_title")
    company = request.form.get("company")
    job_type = request.form.get("job_type", "Full-time")
    existing = applications_collection.find_one({
        "user_email": session["user"], "job_title": job_title, "company": company
    })
    if not existing:
        applications_collection.insert_one({
            "user_email": session["user"], "job_title": job_title,
            "company": company, "job_type": job_type, "status": "applied",
            "applied_date": datetime.now().strftime("%d %b %Y")
        })
    return redirect("/applications")


@app.route("/save_job", methods=["POST"])
@login_required
def save_job():
    job_title = request.form.get("job_title")
    company = request.form.get("company")
    existing = applications_collection.find_one({
        "user_email": session["user"], "job_title": job_title, "company": company
    })
    if not existing:
        applications_collection.insert_one({
            "user_email": session["user"], "job_title": job_title,
            "company": company, "job_type": "—", "status": "saved",
            "applied_date": datetime.now().strftime("%d %b %Y")
        })
    return redirect("/job_matches")

# =========================================================
# SKILL GAP
# =========================================================

@app.route("/skill_gap", methods=["GET", "POST"])
@login_required
def skill_gap():
    user = get_current_user()
    selected_role = request.form.get("target_role") or user.get("job_fit", "Backend Developer")
    gap_skills, readiness = [], 0
    role_skill_labels, user_radar_scores, required_radar_scores = [], [], []

    if user["resume_uploaded"]:
        required = ROLE_REQUIREMENTS.get(selected_role, ROLE_REQUIREMENTS["Backend Developer"])
        user_lower = [s.lower() for s in user["skills"]]
        gap_skills = [r for r in required if r["name"].lower() not in user_lower]
        matched = len(required) - len(gap_skills)
        readiness = int((matched / max(len(required), 1)) * 100)

        for skill_name in [r["name"] for r in required[:6]]:
            role_skill_labels.append(skill_name)
            req_skill = next((r for r in required if r["name"] == skill_name), None)
            required_radar_scores.append(req_skill["importance"] if req_skill else 80)
            if skill_name.lower() in user_lower:
                idx = user_lower.index(skill_name.lower())
                lv = user["skill_levels"][idx] if idx < len(user.get("skill_levels", [])) else 70
                user_radar_scores.append(lv)
            else:
                user_radar_scores.append(10)

        ai_era_names = [s["name"] for s in AI_ERA_SKILLS.get(selected_role, [])]
        users_collection.update_one(
            {"email": session["user"]},
            {"$set": {
                "missing_skills": [r["name"] for r in gap_skills],
                "ai_era_gaps": ai_era_names,
                "readiness": readiness,
                "selected_role": selected_role
            }}
        )
        user = get_current_user()

    return render_template("skill_gap.html",
        user=user, gap_skills=gap_skills, selected_role=selected_role,
        readiness=readiness, roadmap=get_default_roadmap(selected_role),
        role_skill_labels=role_skill_labels, user_radar_scores=user_radar_scores,
        required_radar_scores=required_radar_scores,
        ai_era_skills=AI_ERA_SKILLS.get(selected_role, [])
    )

# =========================================================
# INTERVIEWS
# =========================================================

@app.route("/interviews")
@login_required
def interviews():
    user = get_current_user()
    selected_role = request.args.get("role") or user.get("job_fit", "Backend Developer")
    selected_company = request.args.get("company", "All Companies")
    selected_diff = request.args.get("difficulty", "All")
    questions = get_questions(selected_role, selected_company, selected_diff)
    return render_template("interviews.html", user=user, questions=questions,
        selected_role=selected_role, selected_company=selected_company,
        selected_diff=selected_diff)


@app.route("/mark_practiced", methods=["POST"])
@login_required
def mark_practiced():
    users_collection.update_one({"email": session["user"]}, {"$inc": {"interview_practiced": 1}})
    return redirect(request.referrer or "/interviews")


@app.route("/mark_practiced_ajax", methods=["POST"])
@login_required
def mark_practiced_ajax():
    data = request.get_json()
    count = data.get("count", 1)
    users_collection.update_one({"email": session["user"]}, {"$inc": {"interview_practiced": count}})
    return jsonify({"ok": True})


@app.route("/ai_feedback", methods=["POST"])
@login_required
@limiter.limit("20 per hour")
def ai_feedback():
    """Proxy Gemini AI feedback for interview sessions."""
    data = request.get_json()
    answers_log = data.get("answersLog", [])
    selected_role = data.get("selectedRole", "Software Developer")
    session_time = data.get("sessionTime", "00:00")
    voice_used = data.get("voiceUsed", 0)
    tab_warnings = data.get("tabWarnings", 0)

    answered_count = sum(1 for a in answers_log if "[No answer given" not in a.get("answer", ""))
    skipped_count = len(answers_log) - answered_count

    answers_text = ""
    for i, a in enumerate(answers_log):
        answers_text += f"Q{i+1} [{a.get('difficulty','Medium')} - {a.get('category','Technical')}]: {a.get('question','')}\n"
        answers_text += f"Answer: {a.get('answer','[No answer given]')}\n"
        answers_text += f"Time spent: {a.get('timeSpent', 0)}s\n\n"

    if not answers_text.strip():
        answers_text = "No answers were recorded."

    prompt = f"""You are an expert technical interview coach. Analyze this mock interview session.

Role: {selected_role}
Questions answered: {answered_count}/{len(answers_log)}
Questions skipped: {skipped_count}
Voice responses used: {voice_used}
Tab switches detected: {tab_warnings}
Total session time: {session_time}

ACTUAL ANSWERS:
{answers_text}

Return ONLY valid JSON, no markdown, no backticks, no explanation — just the raw JSON object:
{{
  "score": <0-100 integer based on answer quality>,
  "strengths": ["<specific strength from their answers>", "<another>"],
  "mistakes": ["<specific mistake observed>", "<another>"],
  "improvements": ["<specific actionable improvement>", "<another>"],
  "summary": "<2-3 sentence honest assessment based on what they actually wrote>",
  "top_tip": "<the single most important thing to work on>"
}}

Be HONEST and SPECIFIC. If no answers given, score 0-10. Call out skipped questions."""

    api_key = os.environ.get("GEMINI_API_KEY", "")
    print(f"DEBUG: GEMINI_API_KEY loaded = {'YES, length=' + str(len(api_key)) if api_key else 'NOT FOUND'}")

    if not api_key:
        return jsonify({
            "ok": False,
            "error": "AI feedback is currently unavailable. Please try again later.",
            "no_key": True
        }), 503

    raw_text = ""
    try:
        url = (
           f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={api_key}"
            )
        payload = _json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1000}
        }).encode("utf-8")

        print(f"DEBUG: Calling Gemini API...")

        # ── Retry-aware Gemini call ──────────────────────────────
        result = call_gemini_with_retry(url, payload)

        print("DEBUG: Gemini API response received OK")

        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
        clean = raw_text.strip()

        # Strip markdown code fences if present
        if "```" in clean:
            parts = clean.split("```")
            if len(parts) >= 2:
                clean = parts[1]
                if clean.startswith("json"):
                    clean = clean[4:]
        clean = clean.strip()

        # Extract JSON object boundaries
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]

        feedback = _json.loads(clean)
        return jsonify({"ok": True, "feedback": feedback})

    except RuntimeError as e:
        # Exhausted all retries (Gemini kept returning 429)
        print(f"[ai_feedback] Gemini rate limit exhausted: {e}")
        return jsonify({
            "ok": False,
            "error": "The AI service is currently busy due to high demand. "
                     "Please wait a minute and try again.",
            "rate_limited": True
        }), 429

    except urllib.error.HTTPError as e:
        # Non-429 HTTP error from Gemini (400, 500, etc.)
        print(f"[ai_feedback] Gemini HTTP error {e.code}: {e.reason}")
        return jsonify({
            "ok": False,
            "error": f"AI service error (HTTP {e.code}). Please try again later."
        }), 502

    except _json.JSONDecodeError as e:
        print(f"[JSON parse error]: {e} — raw: {raw_text[:500]}")
        return jsonify({
            "ok": False,
            "error": f"Could not parse AI response: {str(e)}"
        }), 500

    except KeyError as e:
        print(f"[Gemini response structure error]: {e}")
        return jsonify({
            "ok": False,
            "error": f"Unexpected AI response format: {str(e)}"
        }), 500

    except Exception as e:
        import traceback
        print(f"[ai_feedback ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": f"{type(e).__name__}: {e}"
        }), 500

# =========================================================
# APPLICATIONS — with pagination
# =========================================================

@app.route("/applications")
@login_required
def applications():
    user = get_current_user()
    page = request.args.get("page", 1, type=int)
    per_page = 10
    app_counts, total_applications, all_apps = get_app_counts(session["user"])
    apps = all_apps[(page - 1) * per_page: page * per_page]
    total_pages = (total_applications + per_page - 1) // per_page
    return render_template("applications.html",
        user=user, applications=apps, app_counts=app_counts,
        total_applications=total_applications,
        page=page, total_pages=total_pages
    )


@app.route("/add_application", methods=["POST"])
@login_required
def add_application():
    applications_collection.insert_one({
        "user_email": session["user"],
        "job_title": request.form.get("job_title"),
        "company": request.form.get("company"),
        "job_type": request.form.get("job_type", "Full-time"),
        "status": request.form.get("status", "applied"),
        "applied_date": datetime.now().strftime("%d %b %Y")
    })
    return redirect("/applications")


@app.route("/update_application", methods=["POST"])
@login_required
def update_application():
    try:
        applications_collection.update_one(
            {"_id": ObjectId(request.form.get("app_id")), "user_email": session["user"]},
            {"$set": {"status": request.form.get("new_status")}}
        )
    except Exception:
        pass
    return redirect("/applications")

# =========================================================
# SETTINGS
# =========================================================

@app.route("/settings")
@login_required
def settings():
    user = get_current_user()
    return render_template("settings.html", user=user)


@app.route("/settings/profile", methods=["POST"])
@login_required
def settings_profile():
    new_email = request.form.get("email")
    existing = users_collection.find_one({"email": new_email})
    if existing and new_email != session["user"]:
        flash("That email is already in use by another account.", "error")
        return redirect("/settings")
    users_collection.update_one(
        {"email": session["user"]},
        {"$set": {
            "name": request.form.get("name"),
            "email": new_email,
            "job_fit": request.form.get("target_role"),
            "bio": request.form.get("bio", "")
        }}
    )
    session["user"] = new_email
    flash("Profile updated successfully!", "success")
    return redirect("/settings")


@app.route("/settings/password", methods=["POST"])
@login_required
def settings_password():
    user = get_current_user()
    current_pw = request.form.get("current_password")
    new_pw = request.form.get("new_password")
    confirm_pw = request.form.get("confirm_password")
    if not check_password_hash(user["password"], current_pw):
        flash("Current password is incorrect.", "error")
        return redirect("/settings")
    if new_pw != confirm_pw:
        flash("New passwords do not match.", "error")
        return redirect("/settings")
    if len(new_pw) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect("/settings")
    users_collection.update_one(
        {"email": session["user"]},
        {"$set": {"password": generate_password_hash(new_pw)}}
    )
    flash("Password updated!", "success")
    return redirect("/settings")


@app.route("/settings/toggle", methods=["POST"])
@login_required
def settings_toggle():
    key = request.form.get("setting_key")
    if key not in ALLOWED_TOGGLE_KEYS:
        flash("Invalid setting.", "error")
        return redirect("/settings")
    user = get_current_user()
    users_collection.update_one(
        {"email": session["user"]},
        {"$set": {key: not user.get(key, True)}}
    )
    return redirect("/settings")


@app.route("/settings/preferences", methods=["POST"])
@login_required
def settings_preferences():
    users_collection.update_one(
        {"email": session["user"]},
        {"$set": {
            "preferred_location": request.form.get("preferred_location", ""),
            "expected_salary": request.form.get("expected_salary", ""),
            "preferred_job_type": request.form.get("preferred_job_type", "Any")
        }}
    )
    flash("Preferences saved!", "success")
    return redirect("/settings")


@app.route("/delete_resume", methods=["POST"])
@login_required
def delete_resume():
    users_collection.update_one(
        {"email": session["user"]},
        {"$set": {
            "resume_uploaded": False, "skills": [], "missing_skills": [],
            "skill_levels": [], "resume_score": 0, "job_fit": "Job Seeker"
        }}
    )
    flash("Resume data deleted.", "success")
    return redirect("/settings")


@app.route("/delete_account", methods=["POST"])
@login_required
def delete_account():
    users_collection.delete_one({"email": session["user"]})
    applications_collection.delete_many({"user_email": session["user"]})
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)