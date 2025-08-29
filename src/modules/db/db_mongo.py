# src/modules/db/db_mongo.py

from __future__ import annotations

import os
import sys
from typing import Optional, Iterable, Dict, Any, List

import pandas as pd
from pymongo import MongoClient, UpdateOne, ASCENDING

# sys.path: 프로젝트 루트 올려두기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# env 기본값
DEFAULT_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB = os.getenv("MONGO_DB", "sns")

_CLIENT: Optional[MongoClient] = None


# --- 연결 핸들 ---
def get_client(uri: Optional[str] = None) -> MongoClient:
    """싱글턴 클라이언트"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = MongoClient(
            uri or DEFAULT_URI,
            connectTimeoutMS=10_000,
            serverSelectionTimeoutMS=5_000,
        )
    return _CLIENT


def get_db(name: Optional[str] = None):
    return get_client()[name or DEFAULT_DB]


def get_coll(db: Optional[str] = None, coll: str = "posts"):
    return get_db(db)[coll]


# --- 인덱스 ---
def ensure_index_unique(col, field: str, name: Optional[str] = None) -> None:
    # 유니크 보장
    col.create_index([(field, ASCENDING)], unique=True, name=name or f"uniq_{field}")


def ensure_index(col, field: str, direction: int = ASCENDING, unique: bool = False, name: Optional[str] = None) -> None:
    # 일반 인덱스
    col.create_index([(field, direction)], unique=unique, name=name)


# --- DataFrame I/O ---
def _to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.to_dict("records")


def upsert_many(col, records: Iterable[Dict[str, Any]], key: str, batch: int = 1000) -> None:
    """key 기준 대량 upsert"""
    ops: List[UpdateOne] = []
    for r in records:
        k = r.get(key)
        if not k:
            continue
        ops.append(UpdateOne({key: k}, {"$set": r}, upsert=True))
        if len(ops) >= batch:
            col.bulk_write(ops, ordered=False)
            ops = []
    if ops:
        col.bulk_write(ops, ordered=False)


def save_df_unique(df: pd.DataFrame, col, key: str = "post_id") -> None:
    """key 없으면 content로"""
    if key not in df.columns:
        key = "content"
    ensure_index_unique(col, key)
    upsert_many(col, _to_records(df), key)


def read_df(col, query: dict | None = None, projection: dict | None = None) -> pd.DataFrame:
    q = query or {}
    proj = projection or {}
    return pd.DataFrame(list(col.find(q, proj)))


# --- 중복 정리 ---
def remove_duplicates(col, field: str = "content") -> None:
    # 동일 field 그룹에서 첫 번째만 남기고 나머지 삭제
    pipeline = [
        {"$group": {"_id": f"${field}", "ids": {"$push": "$_id"}, "cnt": {"$sum": 1}}},
        {"$match": {"cnt": {"$gt": 1}}},
    ]
    for grp in col.aggregate(pipeline, allowDiskUse=True):
        dup_ids = grp["ids"][1:]
        if dup_ids:
            col.delete_many({"_id": {"$in": dup_ids}})


# --- 간단 유틸 ---
def count(col, query: dict | None = None) -> int:
    return col.count_documents(query or {})


def aggregate(col, pipeline: list) -> list:
    return list(col.aggregate(pipeline, allowDiskUse=True))