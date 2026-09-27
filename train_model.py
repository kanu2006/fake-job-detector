import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Dataset
data = {
    "text": [
        "We are looking for a Senior Software Engineer with 3 years experience in Python and FastAPI.",
        "Earn $500 daily with no experience required! Contact us on Telegram immediately.",
        "Data Analyst position open. Require knowledge of SQL, Excel, and Power BI.",
        "Urgent hiring! Pay refundable training fee of $100 to start earning daily income via WhatsApp.",
        "Frontend Developer wanted with expertise in React, JavaScript, and Tailwind CSS.",
        "Guaranteed daily payout for online simple form filling job. Join our crypto network on Zelle."
    ],
    "label": [0, 1, 0, 1, 0, 1]  # 0 = Real, 1 = Fake
}

df = pd.DataFrame(data)

# 2. TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
X_tfidf = tfidf.fit_transform(df["text"])
y = df["label"]

# 3. Model Training
model = LogisticRegression()
model.fit(X_tfidf, y)

# 4. Save model files
joblib.dump(model, "job_detector_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Models successfully trained and saved!")