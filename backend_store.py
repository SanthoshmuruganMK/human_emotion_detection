import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from werkzeug.security import check_password_hash


IMAGE_LIMIT_PER_EMOTION = 5


class BackendStore:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.database_path = os.path.join(self.root, "emotion_data.sqlite3")
        self.capture_dir = os.path.join(self.root, "captures")
        os.makedirs(self.capture_dir, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS devices (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analysis_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL REFERENCES devices(id),
                    captured_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    face_count INTEGER NOT NULL,
                    image_path TEXT
                );
                CREATE TABLE IF NOT EXISTS face_emotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id INTEGER NOT NULL REFERENCES analysis_frames(id) ON DELETE CASCADE,
                    emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    scores_json TEXT NOT NULL,
                    box_json TEXT NOT NULL,
                    image_saved INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS ix_analysis_frames_device_time
                    ON analysis_frames(device_id, captured_at DESC);
                CREATE INDEX IF NOT EXISTS ix_face_emotions_frame_emotion
                    ON face_emotions(frame_id, emotion);
                """
            )
            device_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(devices)")
            }
            if (
                "password_hash" not in device_columns
                and "access_code_hash" in device_columns
            ):
                connection.execute(
                    "ALTER TABLE devices RENAME COLUMN access_code_hash TO password_hash"
                )
            elif "password_hash" not in device_columns:
                connection.execute("ALTER TABLE devices ADD COLUMN password_hash TEXT")
            if "username" not in device_columns:
                connection.execute("ALTER TABLE devices ADD COLUMN username TEXT")
            if "login_count" not in device_columns:
                connection.execute(
                    "ALTER TABLE devices ADD COLUMN login_count INTEGER NOT NULL DEFAULT 0"
                )
            if "last_login_ip" not in device_columns:
                connection.execute("ALTER TABLE devices ADD COLUMN last_login_ip TEXT")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ix_devices_username
                ON devices(username COLLATE NOCASE)
                """
            )

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def register_user(self, username, password_hash):
        user_id = uuid.uuid4().hex
        now = self._now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO devices (
                        id, display_name, password_hash, created_at, last_seen_at, username
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        username,
                        password_hash,
                        now,
                        now,
                        username,
                    ),
                )
        except sqlite3.IntegrityError:
            return None
        return {"id": user_id, "username": username, "display_name": username}

    def authenticate_user(self, username, password, remote_ip=None):
        with self._connect() as connection:
            user = connection.execute(
                """
                SELECT id, display_name, username, password_hash
                FROM devices WHERE username = ? COLLATE NOCASE
                """,
                (username,),
            ).fetchone()
            if user is None or not check_password_hash(user["password_hash"], password):
                return None
            connection.execute(
                """
                UPDATE devices
                SET last_seen_at = ?, login_count = login_count + 1, last_login_ip = ?
                WHERE id = ?
                """,
                (self._now(), remote_ip, user["id"]),
            )
        return {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
        }

    def set_user_password(self, user_id, password_hash):
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE devices SET password_hash = ?
                WHERE id = ? AND username IS NOT NULL
                """,
                (password_hash, user_id),
            )
        return cursor.rowcount == 1

    def get_user(self, user_id):
        with self._connect() as connection:
            user = connection.execute(
                """
                SELECT id, display_name, username FROM devices
                WHERE id = ? AND username IS NOT NULL
                """,
                (user_id,),
            ).fetchone()
        return dict(user) if user else None

    def touch_user(self, user_id):
        now = datetime.now(timezone.utc)
        stale_before = (now.replace(microsecond=0) - timedelta(seconds=60)).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE devices SET last_seen_at = ?
                WHERE id = ? AND last_seen_at < ?
                """,
                (now.isoformat(timespec="seconds"), user_id, stale_before),
            )

    def record_analysis(self, user_id, source, width, height, faces, jpeg_bytes):
        emotions = sorted({face["emotion"] for face in faces})
        captured_at = self._now()
        frame_path = None
        saved_emotions = []
        saved_path = None

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for emotion in emotions:
                saved_count = connection.execute(
                    """
                    SELECT COUNT(DISTINCT frames.id)
                    FROM analysis_frames AS frames
                    JOIN face_emotions AS emotions ON emotions.frame_id = frames.id
                    WHERE frames.device_id = ? AND frames.image_path IS NOT NULL
                      AND emotions.emotion = ? AND emotions.image_saved = 1
                    """,
                    (user_id, emotion),
                ).fetchone()[0]
                if saved_count < IMAGE_LIMIT_PER_EMOTION:
                    saved_emotions.append(emotion)

            cursor = connection.execute(
                """
                INSERT INTO analysis_frames
                    (device_id, captured_at, source, width, height, face_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, captured_at, source, width, height, len(faces)),
            )
            frame_id = cursor.lastrowid
            if saved_emotions:
                user_capture_dir = os.path.join(self.capture_dir, user_id)
                os.makedirs(user_capture_dir, exist_ok=True)
                filename = f"{frame_id}-{uuid.uuid4().hex}.jpg"
                saved_path = os.path.join(user_capture_dir, filename)
                frame_path = os.path.relpath(saved_path, self.root)

            connection.execute(
                "UPDATE analysis_frames SET image_path = ? WHERE id = ?",
                (frame_path, frame_id),
            )
            for face in faces:
                connection.execute(
                    """
                    INSERT INTO face_emotions
                        (frame_id, emotion, confidence, scores_json, box_json, image_saved)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        frame_id,
                        face["emotion"],
                        face["confidence"],
                        face["scores_json"],
                        face["box_json"],
                        int(face["emotion"] in saved_emotions),
                    ),
                )

            try:
                if saved_path:
                    with open(saved_path, "xb") as image_file:
                        image_file.write(jpeg_bytes)
                        image_file.flush()
                        os.fsync(image_file.fileno())
                connection.commit()
            except Exception:
                connection.rollback()
                if saved_path and os.path.exists(saved_path):
                    os.remove(saved_path)
                raise

        return {"id": frame_id, "image_saved_for": saved_emotions}

    def owner_dashboard_data(self, recent_limit=250):
        with self._connect() as connection:
            totals = {
                "users": connection.execute(
                    "SELECT COUNT(*) FROM devices WHERE username IS NOT NULL"
                ).fetchone()[0],
                "analyses": connection.execute(
                    "SELECT COUNT(*) FROM analysis_frames WHERE image_path IS NOT NULL"
                ).fetchone()[0],
                "faces": connection.execute(
                    "SELECT COALESCE(SUM(face_count), 0) FROM analysis_frames"
                ).fetchone()[0],
                "images": connection.execute(
                    "SELECT COUNT(*) FROM analysis_frames WHERE image_path IS NOT NULL"
                ).fetchone()[0],
            }
            users = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT devices.id, devices.username, devices.display_name,
                           devices.created_at, devices.last_seen_at,
                           devices.login_count, devices.last_login_ip,
                           COUNT(frames.id) AS analysis_count,
                           COALESCE(SUM(frames.face_count), 0) AS face_count
                    FROM devices
                    LEFT JOIN analysis_frames AS frames ON frames.device_id = devices.id
                    GROUP BY devices.id
                    ORDER BY devices.last_seen_at DESC
                    LIMIT 200
                    """
                )
            ]
            captures = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT frames.id, frames.device_id, devices.username,
                           devices.display_name, frames.captured_at, frames.source,
                           frames.width, frames.height, frames.face_count,
                           frames.image_path,
                           GROUP_CONCAT(DISTINCT emotions.emotion) AS emotions
                    FROM analysis_frames AS frames
                    JOIN devices ON devices.id = frames.device_id
                    LEFT JOIN face_emotions AS emotions ON emotions.frame_id = frames.id
                    WHERE frames.image_path IS NOT NULL
                    GROUP BY frames.id
                    ORDER BY frames.captured_at DESC
                    LIMIT ?
                    """,
                    (recent_limit,),
                )
            ]
            capture_ids = [capture["id"] for capture in captures]
            face_details = {}
            if capture_ids:
                placeholders = ",".join("?" for _ in capture_ids)
                rows = connection.execute(
                    f"""
                    SELECT frame_id, emotion, confidence, scores_json, box_json
                    FROM face_emotions
                    WHERE frame_id IN ({placeholders})
                    ORDER BY frame_id, id
                    """,
                    capture_ids,
                )
                for row in rows:
                    face_details.setdefault(row["frame_id"], []).append(
                        {
                            "emotion": row["emotion"],
                            "confidence": row["confidence"],
                            "scores": json.loads(row["scores_json"]),
                            "box": json.loads(row["box_json"]),
                        }
                    )
            for capture in captures:
                capture["faces"] = face_details.get(capture["id"], [])
                capture["image_url"] = (
                    f"/owner/frames/{capture['id']}" if capture["image_path"] else None
                )

        return {"totals": totals, "users": users, "captures": captures}

    def delete_analysis(self, frame_id):
        with self._connect() as connection:
            frame = connection.execute(
                "SELECT image_path FROM analysis_frames WHERE id = ?",
                (frame_id,),
            ).fetchone()
            if frame is None:
                return False
            image_path = frame["image_path"]
            connection.execute("DELETE FROM face_emotions WHERE frame_id = ?", (frame_id,))
            connection.execute("DELETE FROM analysis_frames WHERE id = ?", (frame_id,))

        if image_path:
            self._remove_capture_file(image_path)
        return True

    def delete_user(self, user_id):
        with self._connect() as connection:
            user = connection.execute(
                "SELECT id FROM devices WHERE id = ?",
                (user_id,),
            ).fetchone()
            if user is None:
                return False
            frames = connection.execute(
                "SELECT id, image_path FROM analysis_frames WHERE device_id = ?",
                (user_id,),
            ).fetchall()
            frame_ids = [frame["id"] for frame in frames]
            if frame_ids:
                placeholders = ",".join("?" for _ in frame_ids)
                connection.execute(
                    f"DELETE FROM face_emotions WHERE frame_id IN ({placeholders})",
                    frame_ids,
                )
            connection.execute(
                "DELETE FROM analysis_frames WHERE device_id = ?",
                (user_id,),
            )
            connection.execute("DELETE FROM devices WHERE id = ?", (user_id,))

        for frame in frames:
            if frame["image_path"]:
                self._remove_capture_file(frame["image_path"])
        return True

    def _remove_capture_file(self, image_path):
        capture_path = os.path.abspath(os.path.join(self.root, image_path))
        if os.path.commonpath((self.capture_dir, capture_path)) != self.capture_dir:
            raise RuntimeError("Stored capture path is outside the backend capture directory")
        try:
            os.remove(capture_path)
        except FileNotFoundError:
            pass

    def get_capture_path(self, frame_id):
        with self._connect() as connection:
            frame = connection.execute(
                "SELECT image_path FROM analysis_frames WHERE id = ?",
                (frame_id,),
            ).fetchone()
        if frame is None or not frame["image_path"]:
            return None

        capture_path = os.path.abspath(os.path.join(self.root, frame["image_path"]))
        if os.path.commonpath((self.capture_dir, capture_path)) != self.capture_dir:
            raise RuntimeError("Stored capture path is outside the backend capture directory")
        return capture_path
