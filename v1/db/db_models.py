from sqlalchemy import (
    Column, BigInteger, Integer, String, Text, ForeignKey, TIMESTAMP, func,
    Index, Boolean, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Missions
class Missions(Base):
    __tablename__ = "Missions"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    difficulty = Column(String(10), nullable=False)
    
    group_missions = relationship("GroupMissions", back_populates='mission')

# Users
class Users(Base):
    __tablename__ = "Users"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    provider_id = Column(String(100), nullable=False, unique=True)
    nickname = Column(String(200), nullable=False)
    provider_name = Column(String(100))
    profile_image_url = Column(Text)
    status = Column(String(40), default='active')
    has_completed_survey = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    deleted_at = Column(TIMESTAMP, nullable=True)

    # 관계 설정 (옵션)
    posts = relationship("Posts", back_populates="user", cascade="all, delete-orphan")
    post_likes = relationship("PostLikes", back_populates="user", cascade="all, delete-orphan")
    user_groups = relationship("UserGroups", back_populates="user", cascade="all, delete-orphan")


# Groups
class Groups(Base):
    __tablename__ = "Groups"

    id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False, comment="그룹 고유 식별자")
    name = Column(String(100), nullable=False, comment="그룹 표시용 이름")
    normalized_name = Column(String(100), nullable=False, comment="정규화된 이름")
    description = Column(Text, comment="그룹 설명")
    invite_code = Column(String(6), nullable=False, unique=True, comment="초대 코드")
    image_url = Column(Text, comment="그룹 대표 이미지 URL")

    __table_args__ = (
        Index("idx_groups_normalized_name", "normalized_name"),
    )

    posts = relationship("Posts", back_populates="group", cascade="all, delete-orphan")
    user_groups = relationship("UserGroups", back_populates="group", cascade="all, delete-orphan")
    group_missions = relationship("GroupMissions", back_populates='group')


# Posts
class Posts(Base):
    __tablename__ = "Posts"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey('Users.id'))
    group_id = Column(BigInteger, ForeignKey('Groups.id'))
    week = Column(Integer)
    mission_id = Column(BigInteger, ForeignKey('Missions.id'))
    content = Column(Text)
    created_at = Column(TIMESTAMP)

    # 올바른 관계 설정
    likes = relationship("PostLikes", back_populates="post", cascade="all, delete-orphan")
    user = relationship("Users", back_populates="posts")
    group = relationship("Groups", back_populates="posts")


# PostLikes
class PostLikes(Base):
    __tablename__ = "PostLikes"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey('Users.id'))
    post_id = Column(BigInteger, ForeignKey('Posts.id'))
    created_at = Column(TIMESTAMP)

    post = relationship("Posts", back_populates='likes')
    user = relationship("Users", back_populates='post_likes')

# GroupMissions
class GroupMissions(Base):
    __tablename__ = "GroupMissions"

    id = Column(BigInteger, nullable=False, autoincrement=True, primary_key=True)
    group_id = Column(BigInteger, ForeignKey('Groups.id'), nullable=False)
    mission_id = Column(BigInteger, ForeignKey('Missions.id'))
    week = Column(Integer, nullable=False)
    max_assignable = Column(Integer, default=0)
    remaining_count = Column(Integer, default=0)

    group = relationship("Groups", back_populates='group_missions')
    mission = relationship("Missions", back_populates='group_missions')
    
    __table_args__ = (
        Index("idx_group_week", "group_id", "week"),
        UniqueConstraint('group_id', 'mission_id', name='uq_user_mission'),
    )

# UserGroups
class UserGroups(Base):
    __tablename__ = "UserGroups"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("Users.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(BigInteger, ForeignKey("Groups.id", ondelete="CASCADE"), nullable=False)
    group_user_nickname = Column(String(100), default=None, comment="그룹 내 닉네임")
    normalized_nickname = Column(String(100), default=None, comment="정규화된 닉네임")
    group_user_profile_image_url = Column(Text, default=None)
    joined_at = Column(TIMESTAMP, server_default=func.now())
    is_owner = Column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint('user_id', 'group_id', name='uq_user_group'),
        Index("idx_user_groups_count", "user_id"),
        Index("idx_user_groups_group_normalized_nickname", "group_id", "normalized_nickname"),
    )

    user = relationship("Users", back_populates="user_groups")
    group = relationship("Groups", back_populates="user_groups")