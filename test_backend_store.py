import json
import os
import sqlite3
import tempfile
import unittest

from backend_store import BackendStore, IMAGE_LIMIT_PER_EMOTION
from werkzeug.security import generate_password_hash


class BackendStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = BackendStore(self.temporary_directory.name)
        self.user = self.store.register_user(
            "test-user", generate_password_hash("test-password-123")
        )
        self.frame = b"jpeg-image-data"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_user_password_is_verified_and_only_its_hash_is_stored(self):
        self.assertIsNone(
            self.store.authenticate_user(
                "test-user", "not-the-password", "203.0.113.1"
            )
        )
        authenticated = self.store.authenticate_user(
            "TEST-USER", "test-password-123", "203.0.113.2"
        )
        self.assertEqual(authenticated["display_name"], "test-user")
        with self.store._connect() as connection:
            row = connection.execute(
                "SELECT username, password_hash FROM devices WHERE id = ?",
                (self.user["id"],),
            ).fetchone()
        self.assertEqual(row["username"], "test-user")
        self.assertNotEqual(row["password_hash"], "test-password-123")
        self.assertIsNotNone(self.store.get_user(self.user["id"]))
        account = self.store.owner_dashboard_data()["users"][0]
        self.assertEqual(account["login_count"], 1)
        self.assertEqual(account["last_login_ip"], "203.0.113.2")

    def test_successful_signins_increment_count_and_update_latest_ip(self):
        self.store.authenticate_user(
            "test-user", "test-password-123", "203.0.113.10"
        )
        self.store.authenticate_user(
            "test-user", "test-password-123", "203.0.113.11"
        )
        self.assertIsNone(
            self.store.authenticate_user(
                "test-user", "wrong-password", "203.0.113.12"
            )
        )

        account = self.store.owner_dashboard_data()["users"][0]
        self.assertEqual(account["login_count"], 2)
        self.assertEqual(account["last_login_ip"], "203.0.113.11")

    def test_usernames_are_unique_without_case_sensitivity(self):
        duplicate = self.store.register_user(
            "TEST-USER", generate_password_hash("another-password")
        )
        self.assertIsNone(duplicate)

    def test_owner_can_replace_account_password_hash(self):
        self.assertTrue(
            self.store.set_user_password(
                self.user["id"], generate_password_hash("replacement-password")
            )
        )
        self.assertIsNone(
            self.store.authenticate_user("test-user", "test-password-123")
        )
        self.assertIsNotNone(
            self.store.authenticate_user("test-user", "replacement-password")
        )
        self.assertFalse(
            self.store.set_user_password(
                "missing-user", generate_password_hash("replacement-password")
            )
        )

    def test_full_frames_are_capped_per_emotion_and_analysis_rows_are_retained(self):
        emotion = "Angry"
        for index in range(IMAGE_LIMIT_PER_EMOTION + 2):
            result = self.store.record_analysis(
                self.user["id"],
                "camera",
                1280,
                720,
                [
                    {
                        "emotion": emotion,
                        "confidence": 91.25,
                        "scores_json": json.dumps({emotion: 91.25}),
                        "box_json": json.dumps([5, 6, 70, 80]),
                    }
                ],
                self.frame,
            )
            self.assertEqual(
                bool(result["image_saved_for"]),
                index < IMAGE_LIMIT_PER_EMOTION,
            )

        with self.store._connect() as connection:
            capture_count = connection.execute(
                "SELECT COUNT(*) FROM analysis_frames WHERE device_id = ?",
                (self.user["id"],),
            ).fetchone()[0]
            saved_image_rows = connection.execute(
                """
                SELECT frames.image_path, emotions.emotion, emotions.box_json
                FROM analysis_frames AS frames
                JOIN face_emotions AS emotions ON emotions.frame_id = frames.id
                WHERE frames.device_id = ? AND frames.image_path IS NOT NULL
                  AND emotions.image_saved = 1
                ORDER BY frames.id DESC
                """,
                (self.user["id"],),
            ).fetchall()
            saved_emotion_count = connection.execute(
                """
                SELECT COUNT(DISTINCT frames.id)
                FROM analysis_frames AS frames
                JOIN face_emotions AS emotions ON emotions.frame_id = frames.id
                WHERE frames.device_id = ? AND frames.image_path IS NOT NULL
                  AND emotions.emotion = ? AND emotions.image_saved = 1
                """,
                (self.user["id"], emotion),
            ).fetchone()[0]
        self.assertEqual(capture_count, IMAGE_LIMIT_PER_EMOTION + 2)
        self.assertEqual(saved_emotion_count, IMAGE_LIMIT_PER_EMOTION)
        self.assertEqual(len(saved_image_rows), IMAGE_LIMIT_PER_EMOTION)
        self.assertTrue(
            all(
                os.path.isfile(os.path.join(self.store.root, row["image_path"]))
                for row in saved_image_rows
            )
        )
        self.assertEqual(saved_image_rows[0]["emotion"], emotion)
        self.assertEqual(json.loads(saved_image_rows[0]["box_json"]), [5, 6, 70, 80])
        dashboard = self.store.owner_dashboard_data()
        self.assertEqual(dashboard["totals"]["users"], 1)
        self.assertEqual(dashboard["totals"]["analyses"], IMAGE_LIMIT_PER_EMOTION)
        self.assertEqual(dashboard["totals"]["faces"], IMAGE_LIMIT_PER_EMOTION + 2)
        self.assertEqual(dashboard["totals"]["images"], IMAGE_LIMIT_PER_EMOTION)
        self.assertEqual(dashboard["users"][0]["username"], "test-user")
        self.assertEqual(len(dashboard["captures"]), IMAGE_LIMIT_PER_EMOTION)
        self.assertTrue(all(capture["image_url"] for capture in dashboard["captures"]))
        self.assertTrue(
            all("password_hash" not in item for item in dashboard["users"])
        )
        self.assertTrue(
            all("password_hash" not in item for item in dashboard["captures"])
        )

    def test_frame_without_faces_is_recorded_without_an_image(self):
        result = self.store.record_analysis(
            self.user["id"], "meet", 1920, 1080, [], self.frame
        )
        with self.store._connect() as connection:
            frame = connection.execute(
                """
                SELECT face_count, image_path FROM analysis_frames WHERE id = ?
                """,
                (result["id"],),
            ).fetchone()
        self.assertEqual(result["image_saved_for"], [])
        self.assertEqual(frame["face_count"], 0)
        self.assertIsNone(frame["image_path"])

    def test_delete_analysis_removes_record_faces_and_saved_image(self):
        result = self.store.record_analysis(
            self.user["id"],
            "camera",
            320,
            240,
            [
                {
                    "emotion": "Happy",
                    "confidence": 95.0,
                    "scores_json": '{"Happy":95}',
                    "box_json": "[1,2,30,40]",
                }
            ],
            self.frame,
        )
        image_path = self.store.get_capture_path(result["id"])
        self.assertTrue(os.path.isfile(image_path))

        self.assertTrue(self.store.delete_analysis(result["id"]))
        self.assertFalse(os.path.exists(image_path))
        self.assertFalse(self.store.delete_analysis(result["id"]))
        with self.store._connect() as connection:
            frame_count = connection.execute(
                "SELECT COUNT(*) FROM analysis_frames WHERE id = ?",
                (result["id"],),
            ).fetchone()[0]
            emotion_count = connection.execute(
                "SELECT COUNT(*) FROM face_emotions WHERE frame_id = ?",
                (result["id"],),
            ).fetchone()[0]
        self.assertEqual(frame_count, 0)
        self.assertEqual(emotion_count, 0)

    def test_delete_user_removes_account_analysis_and_images(self):
        result = self.store.record_analysis(
            self.user["id"],
            "camera",
            320,
            240,
            [
                {
                    "emotion": "Happy",
                    "confidence": 95.0,
                    "scores_json": '{"Happy":95}',
                    "box_json": "[1,2,30,40]",
                }
            ],
            self.frame,
        )
        image_path = self.store.get_capture_path(result["id"])
        self.assertTrue(os.path.isfile(image_path))

        self.assertTrue(self.store.delete_user(self.user["id"]))
        self.assertFalse(os.path.exists(image_path))
        self.assertIsNone(self.store.get_user(self.user["id"]))
        with self.store._connect() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM analysis_frames WHERE device_id = ?",
                    (self.user["id"],),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM face_emotions").fetchone()[0],
                0,
            )
        self.assertFalse(self.store.delete_user(self.user["id"]))

    def test_legacy_device_database_migrates_to_password_accounts(self):
        legacy_directory = tempfile.TemporaryDirectory()
        self.addCleanup(legacy_directory.cleanup)
        database_path = os.path.join(legacy_directory.name, "emotion_data.sqlite3")
        connection = sqlite3.connect(database_path)
        connection.execute(
            """
            CREATE TABLE devices (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                access_code_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO devices
                (id, display_name, access_code_hash, created_at, last_seen_at)
            VALUES ('legacy-user', 'Legacy user', 'old-hash', 'now', 'now')
            """
        )
        connection.commit()
        connection.close()

        store = BackendStore(legacy_directory.name)
        with store._connect() as migrated_connection:
            columns = {
                row["name"]
                for row in migrated_connection.execute("PRAGMA table_info(devices)")
            }
            migrated_hash = migrated_connection.execute(
                "SELECT password_hash FROM devices WHERE id = 'legacy-user'"
            ).fetchone()[0]
        self.assertIn("password_hash", columns)
        self.assertNotIn("access_code_hash", columns)
        self.assertIn("login_count", columns)
        self.assertIn("last_login_ip", columns)
        self.assertEqual(migrated_hash, "old-hash")
        self.assertIsNone(store.get_user("legacy-user"))

if __name__ == "__main__":
    unittest.main()
