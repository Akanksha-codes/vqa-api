from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from .model import vqa_pipeline
from .utils import preprocess_image

app = FastAPI()

@app.post("/vqa")
async def visual_qa(image: UploadFile = File(...), question: str = None):
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        image_content = await image.read()
        pil_image = Image.open(io.BytesIO(image_content))
        processed_image = preprocess_image(pil_image)
        
        result = vqa_pipeline(processed_image, question, top_k=1)
        return {"answer": result[0]['answer'], "confidence": result[0]['score']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
