from .main import app

@app.route('/test')
def test():
    return "TEst"