from sqlalchemy import Column, BigInteger, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Users
class Missions(Base):
    __tablename__ = "Missions"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    difficulty = Column(String(10), nullable=False)