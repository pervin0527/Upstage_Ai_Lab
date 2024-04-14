## Installation

    pip install sqlalchemy asyncp psycopg2

    ## macOS
    brew install postgresql@14
    brew services start postgresql@14

## Setup

    psql postgres
    CREATE ROLE postgres WITH LOGIN PASSWORD '비밀번호입력';

    brew services start postgresql
    brew services restart postgresql

    CREATE DATABASE jobpost; ## jobpost는 DB 테이블 이름.
    \connect jobpost;
