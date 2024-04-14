from .database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text, JSON

class JobPost(Base):
    __tablename__ = 'job_posts'

    # id = Column(Integer, primary_key=True)
    # company_name = Column(String, index=True)  # 회사명
    # job_title = Column(String)  # 채용공고 제목
    # job_details = Column(Text)  # 채용공고 세부 사항
    # skills = Column(Text)  # 기술 세부 사항
    # link = Column(String)  # 링크

    id = Column(Integer, primary_key=True)
    main_field = Column(String, index=True)
    num_posts = Column(Integer)
    related_field = Column(JSON)
    

    def __repr__(self):
        return f"<JobPost(company_name='{self.company_name}', job_title='{self.job_title}')>"