from flask import Flask

def create_app():
    app = Flask(__name__)
    # cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.route("/")
    def hello():
        return "Medical Merry"

    
    return app