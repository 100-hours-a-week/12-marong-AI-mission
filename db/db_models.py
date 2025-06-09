from sqlalchemy import Column, BigInteger, String, Text
from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy import Column, Integer, String, Text

Base = declarative_base()

# Users
class Missions(Base):
    __tablename__ = "Missions"

    # 코랩 테스트용 SQLite DB 구조 정의
    # id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    
    # MySQL DB 구조 정의
    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    difficulty = Column(String(10), nullable=False)