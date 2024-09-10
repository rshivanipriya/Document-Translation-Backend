from fastapi import FastAPI, HTTPException,UploadFile,File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# import detectlanguage
from flask_cors import CORS
from flask import Flask, request
from flask_cors import CORS
import docx
from io import BytesIO
import docx2txt
from PyPDF2 import PdfReader
import chardet
 
import langid
 
app = FastAPI()
# app = Flask(__name__)
# CORS(app)
 
# Configure Detect Language API key
 
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
 
class TextPayload(BaseModel):
    text: str
 
# Mapping of language codes to full names
language_map = {
    "af": "Afrikaans",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dz": "Dzongkha",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "nb": "Norwegian Bokmål",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "or": "Oriya",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "se": "Northern Sami",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "vo": "Volapük",
    "wa": "Walloon",
    "xh": "Xhosa",
    "zh": "Chinese",
    "zu": "Zulu"
}

def extract_text_from_file(file: UploadFile) -> str:
    try:
        if file.content_type == "text/plain":
            content = file.file.read()
            result = chardet.detect(content)
            text = content.decode(result['encoding'])
        elif file.content_type == "application/pdf":
            pdf_reader = PdfReader(BytesIO(file.file.read()))
            text = "".join([page.extract_text() for page in pdf_reader.pages])
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(BytesIO(file.file.read()))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

@app.post("/detect-language/")
async def detect_language(file: UploadFile = File(...)):
    try:
        text = extract_text_from_file(file)
        if not text:
            raise HTTPException(status_code=400, detail="No text extracted from the file")

        language_code, confidence = langid.classify(text)
        if language_code in language_map:
            return {"language": language_map[language_code]}
        else:
            raise HTTPException(status_code=400, detail="No language detected")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000)
