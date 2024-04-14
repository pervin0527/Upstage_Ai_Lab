## uvicorn sql_app.main:app --reload
import pandas as pd

from sqlalchemy import func, text
from sqlalchemy.orm import Session
from fastapi import Depends, FastAPI, HTTPException

from . import crud, models, schemas
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# @app.on_event("startup")
# async def load_data():
#     db = next(get_db())
#     # 데이터 저장 전의 총 데이터 개수를 출력
#     initial_count = db.query(func.count(models.JobPost.id)).scalar()
#     print(f"데이터 저장 전 DB의 총 데이터 개수: {initial_count}")

#     if initial_count == 0:  # 데이터가 하나도 없을 때만 실행
#         csv_file = '/Users/pervin0527/Get_ME_AJob/outputs/jobkorea.csv'
#         df = pd.read_csv(csv_file)
#         df.columns = ['index', 'company_name', 'job_title', 'job_details', 'skills', 'link']
#         df.drop(columns=['index'], inplace=True)

#         try:
#             for index, row in df.iterrows():
#                 job_post = schemas.JobPostCreate(
#                     company_name=row['company_name'],
#                     job_title=row['job_title'],
#                     job_details=row['job_details'],
#                     skills=row['skills'],
#                     link=row['link']
#                 )
#                 crud.create_job_post(db=db, job_post=job_post)
#             db.commit()
#         except Exception as e:
#             db.rollback()
#             print(f"데이터 로드 중 오류 발생: {str(e)}")
#         # 데이터 저장 후의 총 데이터 개수를 출력
#         final_count = db.query(func.count(models.JobPost.id)).scalar()
#         print(f"데이터 저장 후 DB의 총 데이터 개수: {final_count}")
#     else:
#         print("DB에 이미 데이터가 존재합니다. 새로운 데이터 로드를 생략합니다.")

#     db.close()


@app.post("/jobposts/", response_model=schemas.JobPostRead)
def create_job_post(job_post: schemas.JobPostCreate, db: Session = Depends(get_db)):
    return crud.create_job_post(db=db, job_post=job_post)


@app.get("/jobposts/{job_post_id}", response_model=schemas.JobPostRead)
def read_job_post(job_post_id: int, db: Session = Depends(get_db)):
    db_job_post = crud.get_job_post(db, job_post_id=job_post_id)
    if db_job_post is None:
        raise HTTPException(status_code=404, detail="JobPost not found")
    return db_job_post


@app.get("/jobposts/", response_model=list[schemas.JobPostRead])
def read_job_posts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    job_posts = crud.get_job_posts(db, skip=skip, limit=limit)
    return job_posts


@app.put("/jobposts/{job_post_id}", response_model=schemas.JobPostRead)
def update_job_post(job_post_id: int, job_post: schemas.JobPostUpdate, db: Session = Depends(get_db)):
    return crud.update_job_post(db=db, job_post_id=job_post_id, updates=job_post)


@app.delete("/jobposts/{job_post_id}", response_model=dict)
def delete_job_post(job_post_id: int, db: Session = Depends(get_db)):
    if crud.delete_job_post(db=db, job_post_id=job_post_id):
        return {"detail": "JobPost deleted"}
    raise HTTPException(status_code=404, detail="JobPost not found")


@app.delete("/clear-data")
def clear_data(db: Session = Depends(get_db)):
    try:
        db.query(models.JobPost).delete()

        sequence_name = "job_posts_id_seq"
        db.execute(text(f"ALTER SEQUENCE {sequence_name} RESTART WITH 1"))

        db.commit()
        return {"message": "모든 데이터가 성공적으로 제거되었습니다."}
    except Exception as e:
        db.rollback()
        return {"error": str(e)}