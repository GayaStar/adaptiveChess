from flask import Flask, request, jsonify, session, render_template_string

import chess
from chess_rl_agent import RLChessAgent
from flask_cors import CORS
import requests
import os
from hybrid_agent import HybridChessAgent


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback-secret")

frontend_url = os.environ.get("FRONTEND_URL", "https://adaptivechess.onrender.com")

CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": [frontend_url]}},
    origins=["https://adaptivechess.onrender.com"], 
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)

# Per-user RL agents
user_agents = {}
base_dir = os.path.dirname(os.path.abspath(__file__))
stockfish_path = os.path.join(base_dir, 'stockfish-ubuntu-x86-64-avx2')
model_path = os.path.join(base_dir, "model", "chess_rl_model_final1.pth")

def get_agent(user_id):
    if user_id not in user_agents:
        # ✅ Resolve absolute path to model
        base_dir = os.path.dirname(os.path.abspath(__file__))  # This is the folder of app.py
        model_path = os.path.join(base_dir, 'model', 'chess_rl_model_final1.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        user_agents[user_id] = RLChessAgent(model_path)
    return user_agents[user_id]

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 204  # Preflight response

    data = request.json
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400

    session['user_id'] = user_id
    return jsonify({'message': 'Logged in', 'user_id': user_id})

# Initialize the hybrid agent once
agent = HybridChessAgent(model_path, stockfish_path)

@app.route("/rl-move", methods=["POST"])
def rl_move():
    data = request.get_json()
    fen = data.get("fen")
    user_id = data.get("user_id")

    # TODO: Look up the user’s ELO from a database. For now, assume 800.
    elo = data.get("elo")
    agent.set_user_elo(elo)

    try:
        board = chess.Board(fen)
        move = agent.select_move(board)
        NODE_BACKEND_URL = os.environ.get("NODE_BACKEND_URL", "https://adaptivechess.onrender.com")

        response = requests.post(f"{NODE_BACKEND_URL}/update_rl_temperature", json={
            "userId": user_id,
            "temperature": agent.temperature
        })
        if response.status_code != 200:
            return jsonify({
            "move": {"from": move.uci()[:2], "to": move.uci()[2:], "promotion": "q"},
            "temperature": agent.temperature
        })



        return jsonify(response)
    except Exception as e:
        return f"Failed to select move: {str(e)}", 500

@app.route("/update-elo", methods=["POST"])
def update_elo():   
    data = request.json
    elo = data.get("elo")
    user_id = data.get("user_id")

    if elo is None or not user_id:
        return jsonify({"error": "Missing elo or user_id"}), 400

    try:
        agent = get_agent(user_id)
        agent.update_user_elo(elo)

        try:
            
            NODE_BACKEND_URL = os.environ.get("NODE_BACKEND_URL", "https://adaptivechess.onrender.com")

            response = requests.post(f"{NODE_BACKEND_URL}/update_rl_temperature", json={
                "userId": user_id,
                "temperature": agent.temperature
            })
            
        except Exception as err:
            error_message = f"[ERROR] While syncing to Node: {err}"
            return render_template_string("""
            <html>
            <body>
                <h1>Application Page</h1>
                <p style="color:red;">{{ error }}</p>
            </body>
            </html>
        """, error=error_message)


        return jsonify({"status": "updated", "temperature": agent.temperature})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_user_id', methods=['GET'])
def get_user_id():
    user_id = session.get('user_id')
    if user_id:
        return jsonify({'user_id': user_id})
    else:
        return jsonify({'error': 'Not logged in'}), 401

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Hybrid Chess API is running"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
