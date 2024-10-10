from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib
import re
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secret key of your choice

# File paths
model_path = 'offensive_language_model.pkl'
vectorizer_path = 'vectorizer.pkl'
offensive_texts_path = 'offensive_texts.txt'

# Variables to store the latest uploaded text
latest_text = None

# Load the model and vectorizer
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please check if the files exist in the specified path.")
    model = None
    vectorizer = None
except Exception as e:
    print(f"An error occurred: {e}")
    model = None
    vectorizer = None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def authenticate(username, password, login_type):
    if login_type == 'admin' and username == 'admin' and password == 'adminpass':
        return 'admin'
    elif login_type == 'user' and username == 'user1' and password == 'userpass':
        return 'user'
    elif login_type == 'user' and username == 'user2' and password == 'userpass':
        return 'user'
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        login_type = request.form['login_type']
        user_type = authenticate(username, password, login_type)
        if user_type:
            session['username'] = username
            session['user_type'] = user_type
            if user_type == 'admin':
                return redirect(url_for('view_offensive_texts'))  # Redirect admin to offensive texts page
            else:
                return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_type', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/check-text', methods=['POST'])
def check_text():
    global latest_text
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not loaded'}), 500

    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    user_input = data['text']
    user_input_clean = clean_text(user_input)
    user_input_vec = vectorizer.transform([user_input_clean])
    prediction = model.predict(user_input_vec)

    if prediction[0] == 1:
        # Create the file if it doesn't exist and append the offensive text and username
        if not os.path.exists(offensive_texts_path):
            with open(offensive_texts_path, 'w') as f:
                f.write("Offensive Texts and Usernames:\n")
        
        with open(offensive_texts_path, 'a') as f:
            f.write(f"User: {session.get('username', 'Unknown')} - Offensive Text: {user_input}\n")
        return jsonify({'offensive': True})
    else:
        latest_text = user_input
        return jsonify({'offensive': False, 'text': user_input})

@app.route('/latest-text')
def get_latest_text():
    if latest_text is None:
        return jsonify({'text': 'No text has been uploaded yet.'})
    return jsonify({'text': latest_text})

@app.route('/latest-text.html')
def latest_text_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('latest_text.html')

@app.route('/admin/offensive-texts')
def view_offensive_texts():
    if 'username' not in session or session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    try:
        with open(offensive_texts_path, 'r') as f:
            offensive_texts = f.read()
    except FileNotFoundError:
        offensive_texts = "No offensive texts found."

    return render_template('offensive_texts.html', texts=offensive_texts)

if __name__ == '__main__':
    app.run(debug=True)
