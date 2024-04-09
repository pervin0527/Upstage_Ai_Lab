## 책 데이터를 CRUD하는 REST API를 만든다.
from fastapi import APIRouter

## 하나의 DB라고 가정.
BOOKS = [
    {
        'id' : 1,
        'title' : '변하지 않는 원칙',
        'author' : '모건 하우절',
        'url' : 'http://yes24.com/변하지않는원칙'
    }
]

## Router -> url을 맵핑 시켜주는 역할을 한다.
router = APIRouter(prefix='/api/v1/books', tags=['books'])

## api/vi/books [GET]
## DB에 있는 모든 데이터를 가져오는 함수.
@router.get('/') ## Read
def get_all_books(): 
    return BOOKS

## id번째 책의 정보를 가져오고 싶을 때.
@router.get('/{book_id}')
def get_book(book_id: int):
    # for book in BOOKS:
    #     if book[id] == book_id:
    #         return book
    # return {"error" : f"Book Not Found ID : {book_id}"}
    
    book = next((book for book in BOOKS if book['id'] == book_id), None)
    if book:
        return book
    return {"error" : f"Book Not Found ID : {book_id}"}


## api/v1/books [POST]
## DB에 새로운 데이터를 추가하고 싶을 때.
@router.post('/') ## Create
def create_book(book : dict): ## book = {'id' : 2, 'title' : '', ...}
    BOOKS.append(book)

    return book

## api/v1/books [POST] ???
## DB에 저장되어 있는 특정 데이터를 수정(업데이트) 하려 할 때.
@router.put('/{book_id}')   ## Update
def update_book(book_id : int, book_update : dict):
    book = next((book for book in BOOKS if book['id'] == book_id), None) ## DB에서 대상 데이터 찾기.
    
    for key, value in book_update.items():
        if key in book:
            book[key] = value
    
    return book
    
## api/v1/books [POST]
## DB에 저장되어 있는 특정 데이터를 제거할 때.
@router.delete('/{book_id}') ## Delete
def delete_book(book_id : int):
    ## id로 찾아서 삭제해도 될텐데 굳이 이렇게 하는 이유가 뭐지??
    global BOOKS
    BOOKS =  [item for item in BOOKS if item['id'] != book_id]

    return {'message' : f'Success to delete book ID : {book_id}'}