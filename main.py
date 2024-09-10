from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, status
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import docx
import requests
from io import BytesIO
from transformers import pipeline, AutoTokenizer, TFMarianMTModel
import torch
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Mount static files (useful for serving HTML pages)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# # Language map for translation
# language_map = {
#     "af": "Afrikaans", "am": "Amharic", "an": "Aragonese", "ar": "Arabic",
#     "as": "Assamese", "az": "Azerbaijani", "be": "Belarusian", "bg": "Bulgarian",
#     "bn": "Bengali", "br": "Breton", "bs": "Bosnian", "ca": "Catalan",
#     # (Other languages omitted for brevity)
#     "zh": "Chinese", "zu": "Zulu"
# }

# src = "en"  # source language for translation

# class TranslationRequest(BaseModel):
#     target_lang: str

# def translate_text(text_chunk, target_lang):
#     """Helper function to translate a text chunk."""
#     batch = tokenizer([text_chunk], return_tensors="tf")
#     gen = model.generate(**batch)
#     translated = tokenizer.batch_decode(gen, skip_special_tokens=True)
#     return translated[0]


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the HTML login page."""
    print('Request for index page received')
    with open("login.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

# @app.get("/translate-document/")
# async def translate_document(
#     target_lang: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     """API endpoint to translate a DOCX document."""
#     try:
#         # Read the file
#         file_content = await file.read()
#         doc = docx.Document(BytesIO(file_content))
#         trg = next((code for code, language in language_map.items() if language.lower() == target_lang.lower()), None)

#         model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"

#         # Load model and tokenizer
#         global model
#         model = TFMarianMTModel.from_pretrained(model_name)
#         global tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(model_name)

#         # Translate paragraphs
#         for para in doc.paragraphs:
#             if para.text.strip():
#                 translated_text = translate_text(para.text, target_lang)
#                 para.text = translated_text

#         # Translate tables, headers, and footers (similar to above logic)
#         for table in doc.tables:
#             for row in table.rows:
#                 for cell in row.cells:
#                     for para in cell.paragraphs:
#                         translated_text = translate_text(para.text, target_lang)
#                         para.text = translated_text

#         for section in doc.sections:
#             for para in section.header.paragraphs:
#                 translated_text = translate_text(para.text, target_lang)
#                 para.text = translated_text

#             for para in section.footer.paragraphs:
#                 translated_text = translate_text(para.text, target_lang)
#                 para.text = translated_text

#         # Save the updated document to a BytesIO object
#         updated_doc_io = BytesIO()
#         doc.save(updated_doc_io)
#         updated_doc_io.seek(0)

#         # Return the translated document as a downloadable file
#         headers = {
#             'Content-Disposition': f'attachment; filename="translated_{file.filename}"',
#             'Content-Type': file.content_type
#         }
#         return StreamingResponse(updated_doc_io, headers=headers)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8008)
