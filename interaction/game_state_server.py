from flask import Flask, request, jsonify, render_template_string, send_from_directory
import threading

app = Flask(__name__)

# This will be injected from the main program
game_state = None


def start_game_state_server(gs, port=8765):
    print("\nSTARTING GAME STATE SERVER...")
    global game_state
    game_state = gs

    thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            use_reloader=False
        ),
        daemon=True
    )
    thread.start()
    print(f"[STATE SERVER] Running on port {port}")


@app.route("/reveal", methods=["POST"])
def reveal():
    data = request.json
    idx = int(data["idx"])
    team = data["team"]

    ok = game_state.reveal_card(idx, team)
    return jsonify({"success": ok})


@app.route("/ping", methods=["GET"])
def ping():
    return "OK", 200


@app.route("/ui")
def ui():
    html = """
    <html>
    <head>
        <style>
            body {
                font-family: sans-serif;
            }

            .palette button {
                font-size: 18px;
                padding: 10px 18px;
                margin-right: 10px;
                cursor: pointer;
            }

            .card {
                position: relative;
                cursor: pointer;
            }

            .overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0.55;
                pointer-events: none;
            }

            .red { background: red; }
            .blue { background: blue; }
            .neutral { background: #e6d36f; }   /* yellowish */
            .assassin { background: black; }
        </style>
    </head>

    <body>
        <h2>Codenames Control</h2>

        <div class="palette" style="margin-bottom: 20px;">
            <button onclick="selectTeam('red')" style="background:red;">Red</button>
            <button onclick="selectTeam('blue')" style="background:blue; color:white;">Blue</button>
            <button onclick="selectTeam('neutral')" style="background:#e6d36f;">Neutral</button>
            <button onclick="selectTeam('assassin')" style="background:black; color:white;">Assassin</button>
            <button onclick="selectTeam('unreveal')" style="background:white; border:2px solid black;">
                Unreveal
            </button>
            <span id="current" style="margin-left:20px; font-size:18px;">
                Selected: none
            </span>
        </div>

        <div id="board"
             style="
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 15px;
                width: 1000px;
             ">
        </div>

        <script>
            let selectedTeam = null;
            const current = document.getElementById("current");
            const board = document.getElementById("board");

            const ROWS = 4;
            const COLS = 5;
            const NUM_CARDS = ROWS * COLS;

            const cards = [];

            function selectTeam(team) {
                selectedTeam = team;
                current.innerText = "Selected: " + team;
            }

            function applyOverlay(card, team) {
                let overlay = card.querySelector(".overlay");
                if (!overlay) {
                    overlay = document.createElement("div");
                    overlay.className = "overlay";
                    card.appendChild(overlay);
                }
                overlay.className = "overlay " + team;
            }

            function renderState(state) {
                for (const idx in state.revealed) {
                    applyOverlay(cards[idx], state.revealed[idx]);
                }
            }

            function fetchState() {
                fetch("/state")
                    .then(res => res.json())
                    .then(renderState);
            }

            for (let i = 0; i < NUM_CARDS; i++) {
                const container = document.createElement("div");
                container.className = "card";

                const img = document.createElement("img");
                img.src = "/card/" + i;
                img.style.width = "100%";

                container.onclick = () => {
                    if (!selectedTeam) return;
                
                    if (selectedTeam === "unreveal") {
                        fetch("/unreveal", {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({idx: i})
                        }).then(() => {
                            const overlay = container.querySelector(".overlay");
                            if (overlay) overlay.remove();
                        });
                        return;
                    }
                
                    fetch("/reveal", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({idx: i, team: selectedTeam})
                    }).then(() => {
                        applyOverlay(container, selectedTeam);
                    });
                };

                container.appendChild(img);
                board.appendChild(container);
                cards.push(container);
            }

            fetchState();
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/board_image")
def board_image():
    # Adjust path if needed
    return send_from_directory("../assets", game_state.board_image)


@app.route("/card/<int:idx>")
def card_image(idx):
    path = game_state.board[idx]
    return send_from_directory("../assets/cards", path)


@app.route("/state")
def state():
    return {
        "revealed": game_state.revealed
    }


@app.route("/unreveal", methods=["POST"])
def unreveal():
    data = request.json
    idx = int(data["idx"])

    ok = game_state.unreveal_card(idx)
    return {"success": ok}
