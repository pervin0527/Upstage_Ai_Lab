from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# 기본 JobPost 정보를 위한 스키마
class JobPostBase(BaseModel):
    main_field: str
    num_posts: int
    related_field: Optional[List[Dict[str, Any]]] = None

# 채용공고 생성 요청을 위한 스키마 (읽기 전용 필드인 id 제외)
class JobPostCreate(JobPostBase):
    pass

# 채용공고 업데이트 요청을 위한 스키마 (선택적 업데이트를 위해 모든 필드를 Optional로 설정)
class JobPostUpdate(BaseModel):
    main_field: Optional[str] = None
    num_posts: Optional[int] = None
    related_field: Optional[List[Dict[str, Any]]] = None

# 채용공고 읽기 요청을 위한 스키마 (모든 필드 포함, ID 추가)
class JobPostRead(JobPostBase):
    id: int

    class Config:
        orm_mode = True
        # from_attributes = True  # 필요하지 않으면 이 줄은 제거 가능

# 응답을 위해 여러 채용공고를 포함할 수 있는 스키마
class JobPostList(BaseModel):
    items: List[JobPostRead]



# # 기본 JobPost 정보를 위한 스키마
# class JobPostBase(BaseModel):
#     company_name: str
#     job_title: str
#     job_details: Optional[str] = None
#     skills: Optional[str] = None
#     link: str

# # 채용공고 생성 요청을 위한 스키마 (읽기 전용 필드인 id, owner_id 제외)
# class JobPostCreate(JobPostBase):
#     pass

# # 채용공고 업데이트 요청을 위한 스키마 (선택적 업데이트를 위해 모든 필드를 Optional로 설정)
# class JobPostUpdate(JobPostBase):
#     company_name: Optional[str] = None
#     job_title: Optional[str] = None
#     job_details: Optional[str] = None
#     skills: Optional[str] = None
#     link: Optional[str] = None

# # 채용공고 읽기 요청을 위한 스키마 (모든 필드 포함, ID 추가)
# class JobPostRead(JobPostBase):
#     id: int

#     class Config:
#         orm_mode = True
#         # from_attributes = True

# # 응답을 위해 여러 채용공고를 포함할 수 있는 스키마
# class JobPostList(BaseModel):
#     items: List[JobPostRead]
