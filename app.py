import os
from collections import deque
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_NAME = "gemini-1.5-flash"
MAX_HISTORY_MESSAGES = 10

# In-memory store per user_id: { "xan123": deque([{"role": "user", "text": "..."}, ...], maxlen=10) }
user_chat_history = {}

SYSTEM_PERSONALITY = (
    "You are a super friendly, witty, and funny best friend chatting in Roman Bangla "
    "(Banglish, using English letters to write Bengali/Banglish conversation). "
    "Always address the user affectionately as 'Xan' and 'jan' naturally in conversation. "
    "Keep replies playful, supportive, and engaging, and include fun emojis in your messages "
    "so the chat feels lively. Never break character."
)


def get_gemini_client():
    """Initialize and return a Gemini client using GEMINI_API_KEY from environment."""
    api_key = os.environ.get("Gemini api key", "").strip()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def append_history(user_id, role, text):
    """Store a message in the user's rolling history of up to 10 messages."""
    if user_id not in user_chat_history:
        user_chat_history[user_id] = deque(maxlen=MAX_HISTORY_MESSAGES)
    user_chat_history[user_id].append({"role": role, "text": text})


def build_contents(user_id, current_message):
    """Build Gemini contents list from the last 10 messages plus the new message."""
    contents = []
    history = user_chat_history.get(user_id)
    if history:
        for item in history:
            gemini_role = "user" if item["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=item["text"])],
                )
            )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=current_message)],
        )
    )
    return contents


@app.route("/", methods=["GET"])
def index():
    """Health check endpoint."""
    return "AI API is Running", 200


@app.route("/demo", methods=["GET"])
def demo_page():
    """Serve the chat bubble demo UI."""
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Generate a Banglish friendly reply using Gemini and per-user history."""
    client = get_gemini_client()
    if client is None:
        return jsonify({
            "error": "GEMINI_API_KEY environment variable is missing.",
            "user_id": None,
        }), 500

    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or "").strip()
    user_id = str(payload.get("user_id") or "").strip()

    if not message or not user_id:
        return jsonify({
            "error": "Both 'message' and 'user_id' are required.",
            "user_id": user_id or None,
        }), 400

    try:
        contents = build_contents(user_id, message)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PERSONALITY,
                temperature=0.85,
            ),
        )

        reply_text = (response.text or "").strip()
        if not reply_text:
            reply_text = "Kemon acho Xan jan? Ektu network jhamela lagche, abar bol na!"

        # Save both user message and assistant reply (capped at last 10 entries per user_id)
        append_history(user_id, "user", message)
        append_history(user_id, "model", reply_text)

        return jsonify({
            "reply": reply_text,
            "user_id": user_id,
        }), 200

    except Exception as exc:
        return jsonify({
            "error": f"Failed to generate reply: {str(exc)}",
            "user_id": user_id,
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
