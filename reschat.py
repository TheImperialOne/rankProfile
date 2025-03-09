from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
import fitz  # PyMuPDF for PDF text extraction
import os
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Secure CORS setup (replace with your frontend URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Securely fetch API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment variables!")

# Set up Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to extract text from PDF
def extract_text_from_pdf(file_stream):
    text = ""
    with fitz.open("pdf", file_stream) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Upload and analyze multiple resumes
@app.post("/upload_resumes/")
async def upload_resumes(files: List[UploadFile] = File(...)):
    extracted_resumes = []

    for file in files:
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext != "pdf":
            raise HTTPException(status_code=400, detail=f"Invalid file {file.filename}. Only PDFs are allowed.")

        if file.size > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail=f"File {file.filename} is too large (limit: 5MB).")

        # Read file content correctly
        file_content = await file.read()
        
        # Process file in memory
        resume_text = extract_text_from_pdf(file_content)

        extracted_resumes.append({"filename": file.filename, "resume_text": resume_text})

    return {"message": "Resumes uploaded successfully", "resumes": extracted_resumes}

@app.post("/compare_resumes/")
async def compare_resumes(query: str = Form(...), resume_texts: List[str] = Form(...)):
    if len(resume_texts) < 2:
        return {"response": "Please upload at least two resumes for comparison."}
    
    # Step 1: Create a structured template for each resume to minimize tokens
    resume_data = []
    
    for i, resume_text in enumerate(resume_texts):
        # Extract candidate name (you could improve this with regex patterns)
        name_line = resume_text.split('\n')[0] if '\n' in resume_text else "Unknown"
        candidate_name = name_line[:30]  # Limit name length
        
        # Extremely minimal approach - just count occurrences of keywords
        skills_count = {}
        common_skills = ["python", "java", "javascript", "react", "angular", "node", 
                        "sql", "nosql", "aws", "azure", "docker", "kubernetes", 
                        "machine learning", "ai", "data science", "full stack", "frontend",
                        "backend", "mobile", "android", "ios", "flutter", "product", "agile"]
        
        for skill in common_skills:
            count = resume_text.lower().count(skill)
            if count > 0:
                skills_count[skill] = count
        
        # Count experience years (very simplified)
        exp_years = 0
        exp_indicators = ["years of experience", "years experience", "yr experience", "year exp"]
        for indicator in exp_indicators:
            if indicator in resume_text.lower():
                # Look for nearby digits
                idx = resume_text.lower().find(indicator)
                context = resume_text[max(0, idx-20):idx]
                for char in context:
                    if char.isdigit():
                        exp_years = int(char)
                        break
        
        # Count projects (very simplified)
        project_count = resume_text.lower().count("project") 
        
        # Create a minimal data structure
        resume_data.append({
            "id": i+1,
            "name": candidate_name,
            "skills": skills_count,
            "exp_years": exp_years,
            "project_count": project_count
        })
    
    # Step 2: Convert to a minimal string representation
    structured_data = "RESUME COMPARISON DATA:\n"
    for r in resume_data:
        structured_data += f"Resume #{r['id']} - {r['name']}\n"
        structured_data += f"Skills: {', '.join([f'{s}({c})' for s,c in r['skills'].items()])}\n"
        structured_data += f"Experience: ~{r['exp_years']} years, Projects: ~{r['project_count']}\n\n"
    
    # Step 3: Use a simpler model with the structured data and query
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",  # You could use a simpler model here
            messages=[
                {"role": "system", "content": "You are an HR assistant. Compare the candidates based on the structured data."},
                {"role": "user", "content": f"{structured_data}\n\nQuery: {query}"}
            ],
            temperature=0.7,
            max_completion_tokens=500
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": f"Error comparing resumes: {str(e)}"}

@app.post("/chat/")
async def chat_with_resume(query: str = Form(...), resume_texts: List[str] = Form(...)):
    if not resume_texts or not any(text.strip() for text in resume_texts):
        raise HTTPException(status_code=400, detail="Please upload at least one resume.")
    
    # Check if the query is asking for comparison/ranking
    is_comparison_query = any(keyword in query.lower() for keyword in 
                             ["compare", "rank", "better", "strongest", "weakest", "rate", "evaluate"])
    
    if is_comparison_query and len(resume_texts) > 1:
        # For comparison queries with multiple resumes, we'll use a special approach
        summaries = []
        
        # Create very concise summaries of each resume
        for i, resume_text in enumerate(resume_texts):
            try:
                # First summarize each resume to extract only key data points
                summary_response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {"role": "system", "content": 
                         "Analyse the given number of resumes and rank them against the provided job description."},
                        {"role": "user", "content": resume_text[:3000]}  # Limit input size
                    ],
                    temperature=0.3,
                    max_completion_tokens=150  # Very limited output
                )
                
                # Store the summary with candidate identifier
                summaries.append(f"Candidate {i+1}: {summary_response.choices[0].message.content}")
            except Exception as e:
                # Fallback if summarization fails
                summaries.append(f"Candidate {i+1}: [Resume processing error: {str(e)}]")
        
        # Combine summaries and perform comparison
        combined_summaries = "\n\n".join(summaries)
        
        try:
            # Make a single API call with the summarized information
            comparison_response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {"role": "system", "content": "Compare the candidates based only on the information provided."},
                    {"role": "user", "content": f"Based on these resume summaries:\n\n{combined_summaries}\n\n{query}"}
                ],
                temperature=0.7,
                max_completion_tokens=500
            )
            
            return {"response": comparison_response.choices[0].message.content}
        except Exception as e:
            return {"response": f"Error processing comparison: {str(e)}. Try comparing fewer resumes or simplify your query."}
    else:
        # For single resume queries or non-comparison queries, process each resume separately
        responses = []
        
        for i, resume_text in enumerate(resume_texts):
            try:
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {"role": "system", "content": "Analyze this resume and answer the question."},
                        {"role": "user", "content": f"Resume: {resume_text[:4000]}"},  # Limit input size
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7,
                    max_completion_tokens=400
                )
                responses.append({"resume_index": i + 1, "response": response.choices[0].message.content})
            except Exception as e:
                responses.append({"resume_index": i + 1, "response": f"Error processing this resume: {str(e)}"})
        
        return {"responses": responses}
    
    