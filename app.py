import base64
import binascii
import functools
import hmac
import ipaddress
import json
import os
import re
import secrets
import socket
import threading
import webbrowser
from datetime import timedelta
from urllib.parse import urlsplit

import cv2
import numpy as np
from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask.sessions import SecureCookieSessionInterface
from tensorflow.keras.models import load_model
from werkzeug.security import generate_password_hash

from backend_store import BackendStore, IMAGE_LIMIT_PER_EMOTION


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "emotion_model.h5")
CASCADE_PATH = os.path.join(PROJECT_DIR, "haarcascade_frontalface_default.xml")
LOGIN_DIR = os.path.join(PROJECT_DIR, "login")
BACKEND_DIR = os.path.abspath(
    os.environ.get("BACKEND_DIR", os.path.join(PROJECT_DIR, "backend"))
)
PUBLIC_BASE_URL = (
    os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL", "")
).rstrip("/")
if PUBLIC_BASE_URL:
    try:
        public_url_parts = urlsplit(PUBLIC_BASE_URL)
        public_url_parts.port
    except ValueError as error:
        raise RuntimeError("PUBLIC_BASE_URL must be a valid HTTPS origin") from error
    if (
        public_url_parts.scheme.lower() != "https"
        or not public_url_parts.netloc
        or public_url_parts.username is not None
        or public_url_parts.password is not None
        or public_url_parts.path not in ("", "/")
        or public_url_parts.query
        or public_url_parts.fragment
    ):
        raise RuntimeError("PUBLIC_BASE_URL must be a valid HTTPS origin")

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

app = Flask(__name__, template_folder=os.path.join(PROJECT_DIR, "templates") if os.path.isdir(os.path.join(PROJECT_DIR, "templates")) else PROJECT_DIR)
app.config.update(
    MAX_CONTENT_LENGTH=12 * 1024 * 1024,
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=bool(PUBLIC_BASE_URL),
    SESSION_COOKIE_SAMESITE="Lax",
)


class PortScopedSessionInterface(SecureCookieSessionInterface):
    def _is_owner_port(self):
        return (
            os.environ.get("LOCAL_DUAL_PORTS") == "1"
            and request.environ.get("SERVER_PORT") == "4000"
        )

    def get_cookie_name(self, app):
        if self._is_owner_port():
            return "emotion_owner_session"
        return super().get_cookie_name(app)

    def get_cookie_path(self, app):
        if self._is_owner_port():
            return "/owner"
        return super().get_cookie_path(app)


app.session_interface = PortScopedSessionInterface()


def get_secret_key():
    secret_key = os.environ.get("APP_SECRET_KEY")
    if secret_key:
        return secret_key

    os.makedirs(LOGIN_DIR, exist_ok=True)
    key_path = os.path.join(LOGIN_DIR, "session.key")
    try:
        with open(key_path, "x", encoding="ascii") as key_file:
            key_file.write(secrets.token_hex(32))
    except FileExistsError:
        pass

    with open(key_path, "r", encoding="ascii") as key_file:
        secret_key = key_file.read().strip()
    if not secret_key:
        raise RuntimeError(f"Flask session key file is empty: {key_path}")
    return secret_key


app.secret_key = get_secret_key()
backend_store = BackendStore(BACKEND_DIR)


def get_network_urls(port=None):
    port = port or int(os.environ.get("PORT", "5000"))
    addresses = set()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as discovery_socket:
        try:
            discovery_socket.connect(("192.0.2.1", 80))
            addresses.add(discovery_socket.getsockname()[0])
        except OSError:
            pass

    if not addresses:
        try:
            addresses.update(
                item[4][0]
                for item in socket.getaddrinfo(
                    socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM
                )
            )
        except socket.gaierror:
            pass

    return [
        f"http://{address}:{port}"
        for address in sorted(addresses)
        if not address.startswith(("127.", "169.254.", "0."))
    ]


def print_application_links(port):
    print("Emotion Studio links:")
    for network_url in get_network_urls(port):
        print(f"  Sign in: {network_url}/login")
    if len(os.environ.get("OWNER_ACCESS_CODE", "")) >= 16:
        print(f"  Private local data viewer: http://127.0.0.1:{port}/owner")
    if PUBLIC_BASE_URL:
        print(f"  Sign in: {PUBLIC_BASE_URL}/login")


def open_application_url(url):
    if not webbrowser.open(url):
        print(f"Could not open {url} in a browser. Open it manually.")


def current_user():
    if "user" not in g:
        user_id = session.get("user_id")
        user = backend_store.get_user(user_id) if user_id else None
        g.user = (
            {
                "id": user["id"],
                "username": user["username"],
                "display_name": user["display_name"],
            }
            if user
            else None
        )
    return g.user


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(*args, **kwargs):
        if current_user() is None:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Please sign in to continue"}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


@app.before_request
def protect_post_requests():
    if os.environ.get("LOCAL_DUAL_PORTS") == "1":
        port = request.environ.get("SERVER_PORT")
        is_owner_path = request.path.startswith("/owner")
        if (port == "4000") != is_owner_path or port not in {"4000", "5000"}:
            abort(404)

    if request.path.startswith("/owner"):
        try:
            remote_is_local = ipaddress.ip_address(request.remote_addr).is_loopback
            host = urlsplit(f"//{request.host}").hostname
            host_is_local = host == "localhost" or (
                host is not None and ipaddress.ip_address(host).is_loopback
            )
        except (TypeError, ValueError):
            remote_is_local = False
            host_is_local = False
        if not remote_is_local or not host_is_local:
            abort(404)

    token = session.get("csrf_token")
    if token is None:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token

    if request.method == "POST":
        submitted_token = request.headers.get("X-CSRF-Token")
        if submitted_token is None:
            submitted_token = request.form.get("csrf_token", "")
        if not hmac.compare_digest(token.encode("utf-8"), submitted_token.encode("utf-8")):
            abort(400, "Invalid or missing form security token")


face_classifier = cv2.CascadeClassifier(CASCADE_PATH)
if face_classifier.empty():
    raise FileNotFoundError(f"Unable to load Haar cascade from {CASCADE_PATH}")

classifier = load_model(MODEL_PATH, compile=False)
if classifier.output_shape[-1] != len(EMOTION_LABELS):
    raise ValueError(
        f"Expected a {len(EMOTION_LABELS)}-class model but found "
        f"{classifier.output_shape[-1]} classes. Please retrain the model first."
    )

profiles = {}


def get_profile(profile_id):
    profile = profiles.get(profile_id)
    if profile is None:
        profile = {
            "id": profile_id,
            "frames": 0,
            "faces_detected": 0,
            "counts": {},
            "most_common": "-",
        }
        profiles[profile_id] = profile
    return profile


def get_most_common(counts):
    if not counts:
        return "-"
    return max(counts.items(), key=lambda item: item[1])[0]


@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        account=current_user(),
        network_urls=get_network_urls(request.environ.get("SERVER_PORT")),
        user_url=f"{PUBLIC_BASE_URL}/login" if PUBLIC_BASE_URL else None,
        image_limit=IMAGE_LIMIT_PER_EMOTION,
    )


@app.route("/login")
def login():
    if current_user() is not None:
        return redirect(url_for("index"))
    return render_template(
        "auth.html",
        image_limit=IMAGE_LIMIT_PER_EMOTION,
    )


@app.route("/api/register", methods=["POST"])
def register():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid registration request"}), 400
    username = payload.get("username")
    password = payload.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        return jsonify({"error": "Enter a username and password"}), 400
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,32}", username):
        return jsonify(
            {
                "error": (
                    "Username must be 3–32 characters and use only letters, "
                    "numbers, dots, underscores, or hyphens"
                )
            }
        ), 400
    if len(password) < 8 or len(password) > 128:
        return jsonify({"error": "Password must be between 8 and 128 characters"}), 400
    if payload.get("consent") is not True:
        return jsonify(
            {"error": "Confirm that people being analyzed know images and results are saved"}
        ), 400

    user = backend_store.register_user(username, generate_password_hash(password))
    if user is None:
        return jsonify({"error": "That username is already in use"}), 409
    session.clear()
    session.permanent = True
    session["user_id"] = user["id"]
    session["csrf_token"] = secrets.token_urlsafe(32)
    return jsonify(
        {
            "redirect": url_for("index"),
        }
    )


@app.route("/api/login", methods=["POST"])
def login_user():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid login request"}), 400
    username = payload.get("username")
    password = payload.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        return jsonify({"error": "Enter your username and password"}), 400

    user = backend_store.authenticate_user(
        username.strip(), password, request.remote_addr
    )
    if user is None:
        return jsonify({"error": "Username or password is not valid"}), 401

    session.clear()
    session.permanent = True
    session["user_id"] = user["id"]
    session["csrf_token"] = secrets.token_urlsafe(32)
    return jsonify({"redirect": url_for("index")})


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/owner", methods=["GET", "POST"])
def owner_dashboard():
    owner_code = os.environ.get("OWNER_ACCESS_CODE", "")
    configured = len(owner_code) >= 16
    if not configured:
        return (
            render_template(
                "owner.html",
                authorized=False,
                configured=False,
                error=None,
            ),
            503,
        )

    if request.method == "POST":
        submitted_code = request.form.get("owner_code", "")
        if not hmac.compare_digest(
            owner_code.encode("utf-8"), submitted_code.encode("utf-8")
        ):
            return (
                render_template(
                    "owner.html",
                    authorized=False,
                    configured=True,
                    error="That owner passphrase is not valid.",
                ),
                401,
            )
        session["owner_authorized"] = True
        session["csrf_token"] = secrets.token_urlsafe(32)
        return redirect(url_for("owner_dashboard"))

    if not session.get("owner_authorized"):
        return render_template(
            "owner.html",
            authorized=False,
            configured=True,
            error=None,
        )
    return render_template(
        "owner.html",
        authorized=True,
        configured=True,
        data=backend_store.owner_dashboard_data(),
    )


@app.route("/owner/users/<user_id>/password", methods=["POST"])
def owner_reset_password(user_id):
    if (
        len(os.environ.get("OWNER_ACCESS_CODE", "")) < 16
        or not session.get("owner_authorized")
    ):
        abort(403)

    password = request.form.get("new_password", "")
    password_confirmation = request.form.get("confirm_password", "")
    if not 8 <= len(password) <= 128:
        flash("The new password must be between 8 and 128 characters.", "error")
    elif password != password_confirmation:
        flash("The new password and confirmation do not match.", "error")
    elif not backend_store.set_user_password(
        user_id, generate_password_hash(password)
    ):
        abort(404)
    else:
        flash("Password reset. The user can sign in with the new password.", "success")
    return redirect(url_for("owner_dashboard"))


@app.route("/owner/users/<user_id>/delete", methods=["POST"])
def owner_delete_user(user_id):
    if (
        len(os.environ.get("OWNER_ACCESS_CODE", "")) < 16
        or not session.get("owner_authorized")
    ):
        abort(403)
    if not backend_store.delete_user(user_id):
        abort(404)
    flash("Account and its stored analyses were deleted.", "success")
    return redirect(url_for("owner_dashboard"))


@app.route("/owner/analyses/<int:frame_id>/delete", methods=["POST"])
def owner_delete_analysis(frame_id):
    if (
        len(os.environ.get("OWNER_ACCESS_CODE", "")) < 16
        or not session.get("owner_authorized")
    ):
        abort(403)
    if not backend_store.delete_analysis(frame_id):
        abort(404)
    flash("Saved analysis and its image were deleted.", "success")
    return redirect(url_for("owner_dashboard"))


@app.route("/owner/logout", methods=["POST"])
def owner_logout():
    session.pop("owner_authorized", None)
    return redirect(url_for("owner_dashboard"))


@app.route("/owner/frames/<int:frame_id>")
def owner_frame(frame_id):
    if (
        len(os.environ.get("OWNER_ACCESS_CODE", "")) < 16
        or not session.get("owner_authorized")
    ):
        abort(403)
    capture_path = backend_store.get_capture_path(frame_id)
    if capture_path is None or not os.path.isfile(capture_path):
        abort(404)
    return send_file(capture_path, mimetype="image/jpeg", conditional=True)


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.route("/api/profile", methods=["POST"])
@login_required
def api_profile():
    user = current_user()
    profile_id = str(user["id"])
    backend_store.touch_user(profile_id)
    profile = get_profile(profile_id)
    profile["most_common"] = get_most_common(profile["counts"])
    return jsonify(profile)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not isinstance(image_data, str) or not image_data:
        return jsonify({"error": "No image provided"}), 400

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    source = payload.get("source", "camera")
    if not isinstance(source, str) or source not in {"camera", "meet", "photo"}:
        return jsonify({"error": "Unsupported capture source"}), 400

    try:
        image_bytes = base64.b64decode(image_data, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"error": "Invalid base64 image data"}), 400

    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode the image"}), 400
    if frame.shape[0] > 2160 or frame.shape[1] > 3840:
        return jsonify({"error": "Images must not exceed 3840×2160 pixels"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    detection_results = []
    primary_emotion = None
    max_area = 0

    for x, y, width, height in faces:
        roi_gray = gray[y : y + height, x : x + width]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if roi_gray.size == 0:
            continue

        roi = roi_gray.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = classifier.predict(roi, verbose=0)[0]
        prediction_index = int(np.argmax(prediction))
        emotion = EMOTION_LABELS.get(prediction_index, f"Class_{prediction_index}")
        confidence = round(float(np.max(prediction)) * 100.0, 2)

        scores = {
            label: round(float(prediction[idx]) * 100.0, 2)
            for idx, label in EMOTION_LABELS.items()
        }

        face_info = {
            "box": [int(x), int(y), int(width), int(height)],
            "emotion": emotion,
            "confidence": confidence,
            "scores": scores,
        }
        detection_results.append(face_info)

        area = width * height
        if area > max_area:
            max_area = area
            primary_emotion = emotion

    user = current_user()
    jpeg_ok, stored_image = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not jpeg_ok:
        raise RuntimeError("Could not encode the analyzed frame for storage")
    stored_faces = [
        {
            "emotion": face["emotion"],
            "confidence": face["confidence"],
            "scores_json": json.dumps(face["scores"], separators=(",", ":")),
            "box_json": json.dumps(face["box"], separators=(",", ":")),
        }
        for face in detection_results
    ]
    capture = backend_store.record_analysis(
        str(user["id"]),
        source,
        int(frame.shape[1]),
        int(frame.shape[0]),
        stored_faces,
        stored_image.tobytes(),
    )
    backend_store.touch_user(str(user["id"]))

    profile = get_profile(str(user["id"]))
    profile["frames"] = int(profile.get("frames", 0)) + 1
    profile["faces_detected"] = len(detection_results)

    if primary_emotion:
        profile["counts"][primary_emotion] = profile["counts"].get(primary_emotion, 0) + 1

    profile["most_common"] = get_most_common(profile["counts"])

    return jsonify(
        {
            "faces": detection_results,
            "profile": profile,
            "stored_images": bool(capture["image_saved_for"]),
        }
    )


if __name__ == "__main__":
    if os.environ.get("LOCAL_DUAL_PORTS") == "1":
        from werkzeug.serving import make_server

        owner_server = make_server("127.0.0.1", 4000, app, threaded=True)
        try:
            login_server = make_server("127.0.0.1", 5000, app, threaded=True)
        except Exception:
            owner_server.server_close()
            raise

        print("Emotion Studio links:")
        print("  Sign in:            http://127.0.0.1:5000/login")
        if len(os.environ.get("OWNER_ACCESS_CODE", "")) >= 16:
            print("  Private owner data: http://127.0.0.1:4000/owner")
        else:
            print("  Set OWNER_ACCESS_CODE to enable the private owner data page.")
        print(
            "Both local interfaces are bound to 127.0.0.1. "
            "Keep this terminal open while using the app."
        )
        threading.Timer(
            1, open_application_url, args=("http://127.0.0.1:5000/login",)
        ).start()
        owner_thread = threading.Thread(
            target=owner_server.serve_forever,
            name="emotion-owner-server",
            daemon=True,
        )
        owner_thread.start()
        try:
            login_server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping Emotion Studio.")
        finally:
            owner_server.shutdown()
            owner_server.server_close()
            login_server.server_close()
        raise SystemExit(0)

    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    print_application_links(port)
    print(
        "This Flask development server is for local-network testing only. "
        "Use a production WSGI server and HTTPS for public hosting."
    )
    application_url = f"http://127.0.0.1:{port}/login"
    threading.Timer(1, open_application_url, args=(application_url,)).start()
    app.run(debug=False, host=host, port=port)
