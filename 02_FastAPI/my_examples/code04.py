from typing import Union, List
from pydantic import BaseModel
from fastapi import FastAPI, Query, Path

app = FastAPI()
TEAMS = []

class Team(BaseModel):
    id : int
    team_name : str
    league : str


## 클라이언트가 홈페이지 데이터를 서버에게 요청 : GET
@app.get('/')
def root():
    return {"Hello" : "Welcome To My Web."}


## CRATE -  클라이언트가 서버로 데이터를 보낼 때 : POST
@app.post('/teams')
async def register_team(team : Team):
    TEAMS.append(team)

    return team

## READ - 클라이언트가 서버에게 id에 해당하는 데이터를 꺼내달라고 요청 : GET
@app.get('/teams')
def get_team(*, 
             team_name: Union[str, None] =  Query(default=None, min_length=1)):
    
    team = next((team for team in TEAMS if team['team_name'] == team_name), None)
    if team:
        return team
    
    return {"Error" : f"Team Not Found : {team}"}


## UPDATE -  데이터 업데이트
@app.put('/teams')
def update_team(*, 
                team_name: Union[str, None] =  Query(default=None, min_length=1),
                update_data: dict):
    
    team = next((team for team in TEAMS if team['team_name'] == team_name), None)
    for key, value in update_data:
        if key in team:
            team[key] = value

    return team


## Delete - 데이터 삭제
@app.delete('/teams')
def delete_team(team_name: str):
    for idx, team in enumerate(TEAMS):
        if team_name == team['team_name']:
            TEAMS.remove(idx)

    return {"Message" : f"Successfully removed {team_name}"}