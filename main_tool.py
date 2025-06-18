from sqlalchemy.orm import Session
from datetime import datetime
from tool.get_week_index import GetWeekIndex
from db.db_models import Missions, Posts, PostLikes
from sqlalchemy import select, func, and_

def get_top_posts(session: Session, difficulty, limit: int=3):
    base_date = datetime(2025, 1, 6)
    today = datetime.today()
    week_index = GetWeekIndex(today, base_date).get()
    
    # 상, 중, 하 별로 별도로 피드 컨텐츠 조회
    stmt = (
        select(Posts.content)
        .join(PostLikes, Posts.id == PostLikes.post_id)
        .join(Missions, Posts.mission_id == Missions.id)
        .where(and_(Posts.week >= max(1, week_index - 2), Missions.difficulty == difficulty))
        .group_by(Posts.id)
        .order_by(func.count(PostLikes.id).desc())
        .limit(limit)
        
    )
    
    result = session.execute(stmt).scalars().all()
    return result