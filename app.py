import os
import warnings
# Suppress TensorFlow deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask import send_file, jsonify
import io
import zipfile

try:
    # import here so Flask can start even if deepface has missing optional deps
    from query_faces import query
except Exception as e:
    # Provide a clearer, actionable error message for common retinaface/tf-keras issue
    msg = str(e)
    helpful = (
        "Failed to import DeepFace/retinaface dependencies. Common fix:\n"
        " - Install tf-keras for TensorFlow 2.20+: `pip install tf-keras`\n"
        " - Or downgrade TensorFlow to a compatible version (e.g. 2.19.x): `pip install 'tensorflow==2.19.0'`\n"
        "Detailed error: " + msg
    )
    raise RuntimeError(helpful)

# Configuration
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

# Flask debug mode (set to False to prevent continuous reloading with TensorFlow)
DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route('/')
def index():
    return render_template('index.html', results=None)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/dataset/<path:filename>')
def dataset_file(filename):
    dataset_dir = os.path.join(BASE_DIR, 'dataset_images')
    return send_from_directory(dataset_dir, filename)


@app.route('/search', methods=['POST'])
def search():
    # index dir and params from form
    index_dir = request.form.get('index_dir') or os.path.join(BASE_DIR, 'face_index')
    top_k = int(request.form.get('top_k') or 20)
    threshold = float(request.form.get('threshold') or 0.45)

    file = request.files.get('query_image')
    if not file or file.filename == '':
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        return "Unsupported file type", 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # run search
    try:
        results = query(index_dir, save_path, top_k=top_k, threshold=threshold)
    except Exception as e:
        return f"Error running search: {e}", 500

    # For each result, build web URL for image display
    display_results = []
    for r in results:
        fname = r.get('file')
        score = r.get('score', 0.0)
        
        # Check if file exists in dataset_images folder
        dataset_path = os.path.join(BASE_DIR, 'dataset_images', fname)
        if os.path.exists(dataset_path):
            # Use web URL to serve the image
            web_url = url_for('dataset_file', filename=fname)
        else:
            # File not found, use None to show placeholder
            web_url = None

        display_results.append({'file': fname, 'score': score, 'path': web_url})

    return render_template('index.html', results=display_results, query_image=url_for('uploaded_file', filename=filename))


@app.route('/download_zip', methods=['POST'])
def download_zip():
    # Expect JSON body with 'files': list of filenames (relative names in dataset_images)
    data = request.get_json() or {}
    files = data.get('files') or []
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    # Create in-memory zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in files:
            # secure the filename to avoid path traversal
            safe = secure_filename(fname)
            dataset_path = os.path.join(BASE_DIR, 'dataset_images', safe)
            if os.path.exists(dataset_path):
                # add file under its basename
                zf.write(dataset_path, arcname=os.path.basename(safe))

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='results_images.zip')


if __name__ == '__main__':
    # Run Flask app without debug mode to prevent continuous reloading
    # The auto-reloader (watchdog) causes issues with TensorFlow/Keras imports
    # To enable debug mode, set environment variable: FLASK_DEBUG=true
    app.run(host='0.0.0.0', port=6070, debug=DEBUG_MODE, use_reloader=DEBUG_MODE)
