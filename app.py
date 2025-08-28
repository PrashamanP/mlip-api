# app.py
from flask import Flask, request, jsonify, render_template
from analyze import get_llm_response
import os

# Point Flask at the templates/ directory
app = Flask(__name__, template_folder='templates')
app.url_map.strict_slashes = False

@app.route("/")
def home():
    return render_template('index.html')   

@app.post("/api/v1/analyze")
def analyze_endpoint():
    try:
        image_data = request.get_data(cache=False)
        if not image_data:
            return jsonify({"error": "No image bytes in request body"}), 400

        result = get_llm_response(image_data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': f'Error retrieving response from LLM. Error: {e}'}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
