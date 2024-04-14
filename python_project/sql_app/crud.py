from . import models, schemas
from sqlalchemy.orm import Session

def create_job_post(db: Session, job_post: schemas.JobPostCreate):
    db_job_post = models.JobPost(**job_post.dict())
    db.add(db_job_post)
    db.commit()
    db.refresh(db_job_post)
    return db_job_post

def get_job_post(db: Session, job_post_id: int):
    return db.query(models.JobPost).filter(models.JobPost.id == job_post_id).first()

def get_job_posts(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.JobPost).offset(skip).limit(limit).all()

def update_job_post(db: Session, job_post_id: int, updates: schemas.JobPostUpdate):
    db_job_post = db.query(models.JobPost).filter(models.JobPost.id == job_post_id).first()
    if db_job_post:
        update_data = updates.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_job_post, key, value)
        db.commit()
        db.refresh(db_job_post)
    return db_job_post

def delete_job_post(db: Session, job_post_id: int):
    db_job_post = db.query(models.JobPost).filter(models.JobPost.id == job_post_id).first()
    if db_job_post:
        db.delete(db_job_post)
        db.commit()
        return True
    return False
