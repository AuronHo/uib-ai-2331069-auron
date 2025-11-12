import os
import pickle
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Load Model and Tokenizer (This runs only ONCE when server starts) ---

# Get the path to the current folder (my_ai_project/core/)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'sentiment_model.keras')
TOKENIZER_PATH = os.path.join(APP_DIR, 'tokenizer.pkl')

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load the tokenizer
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    tokenizer = None

# --- 2. Model Constants (Make sure these match your notebook!) ---
MAX_LEN = 200
THRESHOLD = 0.5

# --- 3. The View Function ---

def sentiment_analysis_view(request):
    """
    This view handles both showing the form (GET) 
    and processing the form data (POST).
    """
    
    context = {
        'sentiment': None,
        'review': None,
    }

    if request.method == 'POST':
        # Get the text from the form
        review_text = request.POST.get('review_text', '')

        if review_text and model and tokenizer:
            try:
                # 1. Preprocess the text (same as your notebook)
                sequence = tokenizer.texts_to_sequences([review_text])
                print(f"DEBUG: Sequence from tokenizer: {sequence}") # <-- ADD THIS
                padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
                print(f"DEBUG: Padded sequence: {padded_sequence}") # <-- ADD THIS

                # 2. Make prediction
                prediction = model.predict(padded_sequence)
                sentiment_score = prediction[0][0] # Get the single prediction value

                # 3. Determine sentiment
                sentiment = "positive" if sentiment_score > THRESHOLD else "negative"

                # 4. Send the result back to the HTML
                context['sentiment'] = sentiment
                context['review'] = review_text
            
            except Exception as e:
                print(f"Prediction error: {e}")
                context['error'] = "An error occurred during prediction."

    # Render the HTML page, passing in the 'context' (result)
    return render(request, 'core/index.html', context)