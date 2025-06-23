from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session
from datetime import datetime
from tool.get_week_index import GetWeekIndex
from db.db_models import Groups, Missions, Posts, PostLikes

# 그룹별 좋아요 가장 높은 피드 조회
def get_top_posts(session: Session, difficulty, group_id, limit: int=3):
    base_date = datetime(2025, 1, 6)
    today = datetime.today()
    week_index = GetWeekIndex(today, base_date).get()
    min_week = max(1, week_index - 3)
    print(f"week_index={week_index}, min_week={min_week}, difficulty={difficulty}, group_id={group_id}")

    # OUTER JOIN으로 좋아요 없는 게시글도 포함시킴
    stmt = (
        select(Posts.content)
        .outerjoin(PostLikes, Posts.id == PostLikes.post_id)
        .join(Missions, Posts.mission_id == Missions.id)
        .where(
            and_(
                Posts.week >= min_week,
                Missions.difficulty == difficulty,
                Posts.group_id == group_id
            )
        )
        .group_by(Posts.id)
        .order_by(func.count(PostLikes.id).desc())
        .limit(limit)
    )
    
    result = session.execute(stmt).scalars().all()
    print("쿼리 결과:", result)
    return result

# 그룹 설명이 의미있는지 판단하는 로직
def is_meaningful_description(desc):
    # None, 빈 문자열, 공백 문자열 체크
    if desc is None:
        return False
    if isinstance(desc, str):
        if desc.strip() == '':
            return False
        # 너무 짧은 설명 or 자주 나오는 의미 없는 패턴들 예시
        if desc.strip().lower() in {'none', '미정', 'no description', 'na', 'test', '테스트'}:
            return False
        if len(desc.strip()) < 3:
            return False
    return True

def largest_mission_id(session: Session):
    stmt = select(Missions.id)
    col = session.execute(stmt).scalars().all()
    max_id = max(col)
    
    return max_id

# DB 그룹 ID 및 설명 조회 (그룹별 미션 만들기 작업)
def get_group_info(session: Session):
    stmt = select(Groups.id, Groups.description)
    rows = session.execute(stmt).all()
    
    group_dict = dict()
    
    for group_id, group_description in rows:
        if is_meaningful_description(group_description):
            group_dict[group_id] = group_description
        else:
            group_dict[group_id] = '개발자들이 뭔가 새롭고 혁신적인 걸 만들어보는 공간'
            
    print(group_dict)
        
    return group_dict