from typing import Annotated, Union

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

app = FastAPI()

items_db = [
    {
        "name": "Apple",
        "description": "Fresh and juicy red apple",
        "price": 0.99,
        "tax": 0.07
    },
    {
        "name": "Banana",
        "description": "Organic banana bunch",
        "price": 1.29,
        "tax": 0.09
    },
    {
        "name": "Orange",
        "description": "Citrus orange",
        "price": 1.49,
        "tax": 0.1
    }    
]

class Item(BaseModel):
    name: str = Field(examples=["NAME"])
    description: Union[str, None] = Field(default=None, examples=["ITEMS"])
    price: float = Field(examples=[77.7])
    tax: Union[float, None] = Field(default=None, examples=[77.77])

# class Item(BaseModel):
#     name: str
#     description: Union[str, None] = None
#     price: float
#     tax: Union[float, None] = None

#     model_config = {
#         "json_schema_extra": {
#             "examples": [
#                 {
#                     "name": "Foo",
#                     "description": "A very nice Item",
#                     "price": 35.4,
#                     "tax": 3.2,
#                 }
#             ]
#         }
#     }


@app.post("/items/")
async def create_item(item: Item):
    item_id = len(items_db) + 1
    items_db[item_id] = item
    return {"item_id": item_id, "item": item}


@app.put("/items/{item_id}")
async def update_item(
        item_id: int,
        item: Annotated[
        Item,
        Body(
            examples=[
                {
                    "name": "Foo",
                    "description": "A very nice Item",
                    "price": 35.4,
                    "tax": 3.2,
                }
            ],
        ),
    ],):
    if item_id in items_db:
        items_db[item_id] = item
        return {"item_id": item_id, "item": item}
    else:
        return {"error": "Item not found"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id in items_db:
        return items_db[item_id]
    else:
        return {"error": "Item not found"}


@app.get("/items/")
async def read_all_items():
    return list(items_db.values())


@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    if item_id in items_db:
        del items_db[item_id]
        return {"message": "Item deleted successfully"}
    else:
        return {"error": "Item not found"}