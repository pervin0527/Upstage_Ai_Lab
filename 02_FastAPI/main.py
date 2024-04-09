from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File

from predict import predict
from books import router as books_router

app = FastAPI()
app.include_router(books_router) ## BOOKS에 대한 라우터를 앱에 연결함.

@app.post('/predict/image')
async def predict_api(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    result = predict(image)

    return result


@app.get("/") ## root 경로. 127.0.0.1:8000/
def index():
    return {"Hello" : "World!"}


if __name__ == "__main__":
    import uvicorn ## ASGI 서버 실행(비동기 방식 서버 실행)
    uvicorn.run("main:app", reload=True) ## main.py에서 app이라는 형태를 띄고 있다.

    ## predict_api를 해보기 위해서는, 서버를 실행시키고 http://127.0.0.1:8000/docs로 이동.
    ## swagger 화면이 나올텐데 try it을 누르고 이미지를 제출하면 모델의 실행 결과를 보여준다.