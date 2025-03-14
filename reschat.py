from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings
import fitz  # PyMuPDF for PDF text extraction
import os
from groq import Groq
from typing import List
from dotenv import load_dotenv
import magic  # For file type validation
import logging
import asyncio
from slowapi import Limiter
from slowapi.util import get_remote_address

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Resume Analyzer API", description="API for uploading and comparing resumes using Groq AI.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Secure CORS setup (replace with your frontend URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Pydantic model for environment variables
class Settings(BaseSettings):
    groq_api_key: str

    class Config:
        env_file = ".env"

# Load settings
settings = Settings()
GROQ_API_KEY = settings.groq_api_key
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment variables!")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Pydantic model for input validation
class ResumeComparisonRequest(BaseModel):
    query: str
    resume_texts: List[str]

# Function to validate PDF file type
def validate_pdf(file_content: bytes):
    mime = magic.from_buffer(file_content, mime=True)
    if mime != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

# Function to extract text from PDF
def extract_text_from_pdf(file_stream: bytes) -> str:
    text = ""
    try:
        doc = fitz.open("pdf", file_stream)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF.")
    return text.strip()

# Upload and analyze multiple resumes
@app.post("/upload_resumes/")
@limiter.limit("5/minute")
async def upload_resumes(request: Request, files: List[UploadFile] = File(...)):
    """
    Upload and process multiple resumes in PDF format.
    - Validates file type and size.
    - Extracts text from each PDF.
    - Returns a list of extracted resume texts.
    """
    extracted_resumes = []

    for file in files:
        # Validate file extension
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext != "pdf":
            raise HTTPException(status_code=400, detail=f"Invalid file {file.filename}. Only PDFs are allowed.")

        # Validate file size (5MB limit)
        if file.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is too large (limit: 5MB).")

        # Read file content
        file_content = await file.read()

        # Validate file type
        validate_pdf(file_content)

        # Extract text from PDF
        resume_text = extract_text_from_pdf(file_content)

        extracted_resumes.append({"filename": file.filename, "resume_text": resume_text})

    return {"message": "Resumes uploaded successfully", "resumes": extracted_resumes}

# Compare resumes using Groq API
@app.post("/compare_resumes/")
@limiter.limit("5/minute")
async def compare_resumes(request: Request, request_data: ResumeComparisonRequest):
    """
    Compare multiple resumes based on a query.
    - Uses Groq API to analyze and compare resumes.
    - Returns a comparison response.
    """
    if len(request_data.resume_texts) < 2:
        return {"response": "Please upload at least two resumes for comparison."}

    # Create structured data for comparison
    resume_data = []
    for i, resume_text in enumerate(request_data.resume_texts):
        skills_count = {}
        common_skills = ["python", "java", "javascript", "react", "angular", "node", 
                        "sql", "nosql", "aws", "azure", "docker", "kubernetes", 
                        "machine learning", "ai", "data science", "full stack", "frontend",
                        "backend", "mobile", "android", "ios", "flutter", "product", "agile"]
        
        for skill in common_skills:
            count = resume_text.lower().count(skill)
            if count > 0:
                skills_count[skill] = count

        # Count experience years (simplified)
        exp_years = 0
        exp_indicators = ["years of experience", "years experience", "yr experience", "year exp"]
        for indicator in exp_indicators:
            if indicator in resume_text.lower():
                idx = resume_text.lower().find(indicator)
                context = resume_text[max(0, idx-20):idx]
                for char in context:
                    if char.isdigit():
                        exp_years = int(char)
                        break

        # Count projects (simplified)
        project_count = resume_text.lower().count("project") 

        resume_data.append({
            "id": i + 1,
            "skills": skills_count,
            "exp_years": exp_years,
            "project_count": project_count
        })

    # Convert to a minimal string representation
    structured_data = "RESUME COMPARISON DATA:\n"
    for r in resume_data:
        structured_data += f"Resume #{r['id']}\n"
        structured_data += f"Skills: {', '.join([f'{s}({c})' for s, c in r['skills'].items()])}\n"
        structured_data += f"Experience: ~{r['exp_years']} years, Projects: ~{r['project_count']}\n\n"

    # Use Groq API for comparison
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": "You are an HR assistant. Compare the candidates based on the structured data."},
                {"role": "user", "content": f"{structured_data}\n\nQuery: {request_data.query}"}
            ],
            temperature=0.7,
            max_completion_tokens=500
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to compare resumes due to an internal error.")

# Chat with resume using Groq API
@app.post("/chat/")
@limiter.limit("5/minute")
async def chat_with_resume(request: Request, request_data: ResumeComparisonRequest):
    """
    Chat with a resume based on a query.
    - Uses Groq API to analyze the resume and answer the query.
    - Returns a response.
    """
    if not request_data.resume_texts or not any(text.strip() for text in request_data.resume_texts):
        raise HTTPException(status_code=400, detail="Please upload at least one resume.")

    responses = []
    for i, resume_text in enumerate(request_data.resume_texts):
        try:
            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {"role": "system", "content": "Analyze this resume and answer the question."},
                    {"role": "user", "content": f"Resume: {resume_text[:4000]}"},  # Limit input size
                    {"role": "user", "content": request_data.query}
                ],
                temperature=0.7,
                max_completion_tokens=400
            )
            responses.append({"resume_index": i + 1, "response": response.choices[0].message.content})
        except Exception as e:
            logger.error(f"Groq API error for resume {i + 1}: {str(e)}")
            responses.append({"resume_index": i + 1, "response": f"Error processing this resume: {str(e)}"})

    return {"responses": responses}
