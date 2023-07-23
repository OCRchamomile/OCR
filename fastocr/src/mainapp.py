from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn
import argparse
import shutil
import os

from mainmodel import scaleimg, create_df_ocr
from img2table.ocr import EasyOCR
import easyocr

import time
from functools import lru_cache

@app.get('/')
def main():
    html_content = """
            <html>
            <body>
            <form action="http://localhost:8000/ocr" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" />
            <input type="submit"/>
            </form>
            </body>
            </html>
            """
    return HTMLResponse(content=html_content)

@lru_cache(maxsize=1)
def load_ocr_model():
    model = easyocr.Reader(['ru'], gpu=False)
    return model

def invoke_ocr(doc, content_type):
    worker_pid = os.getpid()
    print(f"Handling OCR request with worker PID: {worker_pid}")
    start_time = time.time()

    model = load_ocr_model()

    format_img = "JPEG"
    if content_type == "image/png":
        format_img = "PNG"

    ocr_res = model.readtext(doc)
    df_res = pd.DataFrame(ocr_res, columns=['box', 'text', 'coef'])
    res = create_df_ocr(df_res)
    values = res.to_dict(orient="records")

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"OCR done, worker PID: {worker_pid}")

    return values, processing_time

@app.post("/ocr")
async def create_upload_file(file: UploadFile):
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    print("file saved at " + file_location)

    img_scale = scaleimg(file_location)

    result = None
    if file:
        if file.content_type in ["image/jpeg", "image/jpg", "image/png"]:
            result, processing_time = invoke_ocr(img_scale, file.content_type)
        else:
            return {"error": "Invalid file type. Only JPG/PNG images and PDF are allowed."}
      
    return JSONResponse(status_code=status.HTTP_200_OK, content=result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
