from fastapi import FastAPI

app = FastAPI() ## app 객체는 하나의 웹 애플리케이션을 의미

@app.get('/') ## 특정 경로(현재 홈페이지 주소)에 대한 GET 요청을 처리하는 함수.
async def root():
    return {"message" : 'Hello world.'}

# @app.get("/items/{item_id}") ## example/com/items/{item_id}
# async def read_item(item_id):
#     return {"item_id": item_id}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


# @app.get("/users/{user_id}")
# async def read_user(user_id: str):
#     return {"read_user func": user_id}

@app.get("/users/me")
async def read_user_me():
    return {"read_user_me func": "the current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"read_user func": user_id}


from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

# @app.get("/models/{model_name}")
# async def get_model(model_name: str):
#     if model_name == "alexnet":
#         return {"model_name": model_name, "message": "Deep Learning FTW!"}
    
#     if model_name == 'lenet':
#         return {"model_name": model_name, "message": "LeCNN all the images"}
    
#     return {"model_name": model_name, "message": "Have some residuals"}

# @app.get("/files/{file_path:path}")
# async def read_file(file_path: str):
#     return {"file_path": file_path}

## /Users/pervin0527/test.txt
from fastapi import FastAPI, HTTPException

@app.get("/files/{file_path:path}")
async def read_file(file_path: str):    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            contents = file.read()
            return {"file_path": file_path, "content": contents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))