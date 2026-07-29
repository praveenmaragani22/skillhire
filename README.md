# SkillHire

A Flask-based job recommendation and skill gap analysis platform for tech job seekers.

## Features
- Resume upload & ATS scoring
- AI-powered skill gap analysis
- Job matching with live Naukri / LinkedIn / Indeed links
- Mock interview practice with Gemini AI feedback
- Application tracker

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your actual values
```

### 3. Start MongoDB
Make sure MongoDB is running locally on port 27017, or set your `MONGO_URI` in `.env`.

### 4. Run the app
```bash
# Development
FLASK_DEBUG=true python job_recommendation/app.py

# Production (use a WSGI server like gunicorn)
gunicorn job_recommendation.app:app
```

## Environment Variables

| Variable | Description |
|---|---|
| `SECRET_KEY` | Flask session secret (generate with `python -c "import secrets; print(secrets.token_hex(32))"`) |
| `MONGO_URI` | MongoDB connection string |
| `GEMINI_API_KEY` | Google Gemini API key for AI feedback |
| `FLASK_DEBUG` | Set to `true` for development only |

## Security Notes
- Never commit your `.env` file to git
- Add `.env` to your `.gitignore`
- Set `SESSION_COOKIE_SECURE = True` when running over HTTPS
