"""
database.py - MongoDB persistence for MachineGuard AI (PyMongo).

Collections: machines, predictions, prediction_details, maintenance_tasks, activity_log.
The database starts empty; nothing is pre-filled.
"""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

import config as cfg


class DatabaseError(Exception):
    """Raised when MongoDB is unavailable or an operation fails."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _key(name: Any) -> str:
    return str(name).replace(".", "_").lstrip("$")


def _clean(value: Any) -> Any:
    """Convert numpy / pandas values into BSON-friendly Python values (NaN -> None)."""
    if isinstance(value, dict):
        return {_key(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is pd.NaT:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    return value


class MachineGuardDB:
    """Thin wrapper around the MongoDB database used by the application."""

    def __init__(self, uri: str | None = None, db_name: str | None = None):
        self.uri = uri or cfg.MONGODB_URI
        self.db_name = db_name or cfg.DATABASE_NAME
        self.client = None
        self.db = None
        self.error: str | None = None
        self.connect()

    # ------------------------------------------------------------ connection
    def connect(self) -> bool:
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=cfg.MONGO_TIMEOUT_MS, tz_aware=True)
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self._initialise()
            self.error = None
        except Exception as exc:  # noqa: BLE001
            self.db = None
            self.error = (f"Cannot reach MongoDB at {self._safe_uri()} ({type(exc).__name__}). "
                          "Start MongoDB and check MONGODB_URI in .env.")
        return self.connected

    def _safe_uri(self) -> str:
        """URI without any credentials, for messages."""
        uri = self.uri
        if "@" in uri and "://" in uri:
            scheme, rest = uri.split("://", 1)
            uri = f"{scheme}://***@{rest.split('@', 1)[1]}"
        return uri

    @property
    def connected(self) -> bool:
        return self.db is not None

    def _initialise(self) -> None:
        existing = set(self.db.list_collection_names())
        for name in (cfg.COL_MACHINES, cfg.COL_PREDICTIONS, cfg.COL_DETAILS, cfg.COL_TASKS, cfg.COL_ACTIVITY):
            if name not in existing:
                self.db.create_collection(name)
        self.db[cfg.COL_MACHINES].create_index("machine_id", unique=True)
        self.db[cfg.COL_PREDICTIONS].create_index("prediction_id", unique=True)
        self.db[cfg.COL_PREDICTIONS].create_index([("created_at", DESCENDING)])
        self.db[cfg.COL_PREDICTIONS].create_index("machine_id")
        self.db[cfg.COL_DETAILS].create_index("prediction_id")
        self.db[cfg.COL_DETAILS].create_index([("model_key", ASCENDING), ("created_at", ASCENDING)])
        self.db[cfg.COL_TASKS].create_index("task_id", unique=True)
        self.db[cfg.COL_TASKS].create_index("machine_id")
        self.db[cfg.COL_ACTIVITY].create_index([("created_at", DESCENDING)])

    def _col(self, name: str):
        if not self.connected:
            raise DatabaseError(self.error or "MongoDB is not connected.")
        return self.db[name]

    @staticmethod
    def _find(collection, query=None, sort=None, limit=0, projection=None) -> list[dict]:
        try:
            cursor = collection.find(query or {}, {"_id": 0, **(projection or {})})
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
            return list(cursor)
        except PyMongoError as exc:
            raise DatabaseError(f"Database read failed: {exc}") from exc

    # -------------------------------------------------------------- activity
    def log_activity(self, action: str, entity_type: str, entity_id: str, message: str = "") -> None:
        try:
            self._col(cfg.COL_ACTIVITY).insert_one({
                "created_at": _now(), "action": action, "entity_type": entity_type,
                "entity_id": entity_id, "message": message,
            })
        except (DatabaseError, PyMongoError):
            pass  # activity logging must never break the main operation

    def list_activity(self, limit: int = 30) -> list[dict]:
        return self._find(self._col(cfg.COL_ACTIVITY), sort=[("created_at", DESCENDING)], limit=limit)

    # -------------------------------------------------------------- machines
    def add_machine(self, data: dict) -> dict:
        machine_id = str(data.get("machine_id", "")).strip()
        if not machine_id or not str(data.get("name", "")).strip():
            raise DatabaseError("Machine ID and name are required.")
        doc = _clean({**data, "machine_id": machine_id, "active": True,
                      "created_at": _now(), "updated_at": _now()})
        try:
            self._col(cfg.COL_MACHINES).insert_one(dict(doc))
        except DuplicateKeyError as exc:
            raise DatabaseError(f"Machine ID '{machine_id}' already exists.") from exc
        except PyMongoError as exc:
            raise DatabaseError(f"Could not save the machine: {exc}") from exc
        self.log_activity("machine_added", "machine", machine_id, f"Registered machine {doc['name']}")
        return doc

    def update_machine(self, machine_id: str, updates: dict) -> None:
        updates = {k: v for k, v in updates.items() if k not in ("machine_id", "_id")}
        updates["updated_at"] = _now()
        try:
            res = self._col(cfg.COL_MACHINES).update_one({"machine_id": machine_id}, {"$set": _clean(updates)})
        except PyMongoError as exc:
            raise DatabaseError(f"Could not update the machine: {exc}") from exc
        if res.matched_count == 0:
            raise DatabaseError(f"Machine '{machine_id}' was not found.")
        self.log_activity("machine_updated", "machine", machine_id, "Machine details edited")

    def set_machine_active(self, machine_id: str, active: bool) -> None:
        self.update_machine(machine_id, {"active": bool(active)})
        self.log_activity("machine_activated" if active else "machine_deactivated", "machine", machine_id,
                          "Machine reactivated" if active else "Machine deactivated")

    def get_machine(self, machine_id: str) -> dict | None:
        rows = self._find(self._col(cfg.COL_MACHINES), {"machine_id": machine_id}, limit=1)
        return rows[0] if rows else None

    def list_machines(self, active_only: bool = False) -> list[dict]:
        query = {"active": True} if active_only else {}
        return self._find(self._col(cfg.COL_MACHINES), query, sort=[("machine_id", ASCENDING)])

    # ----------------------------------------------------------- predictions
    def save_prediction(self, summary: dict, details: list[dict]) -> str:
        prediction_id = f"PRD-{_now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:5].upper()}"
        created = summary.get("created_at") or _now()
        doc = _clean({**summary, "prediction_id": prediction_id, "created_at": created,
                      "review_status": cfg.REVIEW_STATUSES[0], "review_note": "",
                      "stored_detail_rows": len(details)})
        try:
            self._col(cfg.COL_PREDICTIONS).insert_one(dict(doc))
            rows = [_clean({**d, "prediction_id": prediction_id, "machine_id": summary.get("machine_id"),
                            "model_key": summary.get("model_key"), "created_at": created}) for d in details]
            for start in range(0, len(rows), 1000):
                self._col(cfg.COL_DETAILS).insert_many(rows[start:start + 1000], ordered=False)
        except PyMongoError as exc:
            raise DatabaseError(f"Could not save the prediction: {exc}") from exc
        self.log_activity("prediction_saved", "prediction", prediction_id,
                          f"{summary.get('n_records', 0)} record(s) on {summary.get('machine_name', '')}")
        return prediction_id

    def list_predictions(self, machine_id: str | None = None, model_key: str | None = None,
                         input_method: str | None = None, date_from: datetime | None = None,
                         date_to: datetime | None = None, limit: int = 1000) -> list[dict]:
        query: dict = {}
        if machine_id:
            query["machine_id"] = machine_id
        if model_key:
            query["model_key"] = model_key
        if input_method:
            query["input_method"] = input_method
        if date_from or date_to:
            query["created_at"] = {}
            if date_from:
                query["created_at"]["$gte"] = date_from
            if date_to:
                query["created_at"]["$lte"] = date_to
        return self._find(self._col(cfg.COL_PREDICTIONS), query, sort=[("created_at", DESCENDING)], limit=limit)

    def get_prediction(self, prediction_id: str) -> dict | None:
        rows = self._find(self._col(cfg.COL_PREDICTIONS), {"prediction_id": prediction_id}, limit=1)
        return rows[0] if rows else None

    def get_details(self, prediction_id: str, predicted_class: int | None = None, limit: int = 0) -> list[dict]:
        query: dict = {"prediction_id": prediction_id}
        if predicted_class is not None:
            query["predicted_class"] = int(predicted_class)
        return self._find(self._col(cfg.COL_DETAILS), query, sort=[("row_number", ASCENDING)], limit=limit)

    def get_details_by_model(self, model_key: str, limit: int = 20000) -> list[dict]:
        return self._find(self._col(cfg.COL_DETAILS), {"model_key": model_key},
                          sort=[("created_at", ASCENDING), ("row_number", ASCENDING)], limit=limit)

    def update_review(self, prediction_id: str, status: str, note: str = "") -> None:
        try:
            res = self._col(cfg.COL_PREDICTIONS).update_one(
                {"prediction_id": prediction_id},
                {"$set": {"review_status": status, "review_note": note, "reviewed_at": _now()}})
        except PyMongoError as exc:
            raise DatabaseError(f"Could not update the review status: {exc}") from exc
        if res.matched_count == 0:
            raise DatabaseError("Prediction not found.")
        self.log_activity("review_updated", "prediction", prediction_id, f"Review status set to '{status}'")

    # ----------------------------------------------------------------- tasks
    def add_task(self, data: dict) -> dict:
        if not str(data.get("description", "")).strip():
            raise DatabaseError("A task description is required.")
        task_id = f"TASK-{uuid.uuid4().hex[:8].upper()}"
        doc = _clean({**data, "task_id": task_id, "created_at": _now(), "updated_at": _now(),
                      "completed_at": None})
        try:
            self._col(cfg.COL_TASKS).insert_one(dict(doc))
        except PyMongoError as exc:
            raise DatabaseError(f"Could not save the task: {exc}") from exc
        self.log_activity("task_created", "task", task_id, doc["description"][:80])
        return doc

    def list_tasks(self, machine_id: str | None = None, status: str | None = None,
                   priority: str | None = None, limit: int = 1000) -> list[dict]:
        query: dict = {}
        if machine_id:
            query["machine_id"] = machine_id
        if status:
            query["status"] = status
        if priority:
            query["priority"] = priority
        return self._find(self._col(cfg.COL_TASKS), query,
                          sort=[("scheduled_date", ASCENDING), ("created_at", DESCENDING)], limit=limit)

    def update_task(self, task_id: str, updates: dict) -> None:
        updates = {k: v for k, v in updates.items() if k not in ("task_id", "_id")}
        updates["updated_at"] = _now()
        if updates.get("status") == "Completed":
            updates["completed_at"] = _now()
        elif "status" in updates:
            updates["completed_at"] = None
        try:
            res = self._col(cfg.COL_TASKS).update_one({"task_id": task_id}, {"$set": _clean(updates)})
        except PyMongoError as exc:
            raise DatabaseError(f"Could not update the task: {exc}") from exc
        if res.matched_count == 0:
            raise DatabaseError("Task not found.")
        self.log_activity("task_updated", "task", task_id,
                          f"Status: {updates['status']}" if "status" in updates else "Task edited")

    def update_task_status(self, task_id: str, status: str) -> None:
        self.update_task(task_id, {"status": status})

    def delete_task(self, task_id: str) -> None:
        try:
            self._col(cfg.COL_TASKS).delete_one({"task_id": task_id})
        except PyMongoError as exc:
            raise DatabaseError(f"Could not delete the task: {exc}") from exc
        self.log_activity("task_deleted", "task", task_id, "Task deleted")

    # ------------------------------------------------------------- dashboard
    def dashboard_counts(self) -> dict:
        try:
            predictions = self._col(cfg.COL_PREDICTIONS)
            totals = list(predictions.aggregate([{"$group": {
                "_id": None, "records": {"$sum": "$n_records"}, "failures": {"$sum": "$n_failure"}}}]))
            return {
                "machines": self._col(cfg.COL_MACHINES).count_documents({"active": True}),
                "predictions": predictions.count_documents({}),
                "records": int(totals[0]["records"]) if totals else 0,
                "failures": int(totals[0]["failures"]) if totals else 0,
                "open_tasks": self._col(cfg.COL_TASKS).count_documents(
                    {"status": {"$in": cfg.OPEN_TASK_STATUSES}}),
            }
        except PyMongoError as exc:
            raise DatabaseError(f"Database read failed: {exc}") from exc
