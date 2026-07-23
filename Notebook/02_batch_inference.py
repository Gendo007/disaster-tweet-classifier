import os
import joblib
import pandas as pd
import torch
import torch.nn as nn


class DisasterTweetClassifier(nn.Module):

    def __init__(self, input_dim: int = 1000, hidden_dim: int = 16):
        super(DisasterTweetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def clean_text(text: str) -> str:
    text = str(text).lower()
    return "".join([c for c in text if c.isalnum() or c == " "])


def run_batch_inference(input_csv_path: str, output_csv_path: str):

    # Load artifacts
    vectorizer = joblib.load("../artifacts/tfidf_vectorizer.pkl")

    model = DisasterTweetClassifier(input_dim=1000, hidden_dim=16)
    model.load_state_dict(torch.load("../artifacts/disaster_model.pth"))
    model.eval()

    # Ingest test stream
    df = pd.read_csv(input_csv_path)

    # Feature transformation
    cleaned = df["text"].apply(clean_text)
    X_vec = vectorizer.transform(cleaned).toarray()
    X_tensor = torch.tensor(X_vec, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        probs = model(X_tensor).numpy().flatten()

    # Append predictions
    df["disaster_probability"] = probs.round(4)
    df["is_disaster"] = (df["disaster_probability"] >= 0.50).astype(int)
    df["status_flag"] = df["is_disaster"].map(
        {1: "CRITICAL_ALERT", 0: "NORMAL_CHATTER"}
    )

    # Export annotated dataset
    df.to_csv(output_csv_path, index=False)
    print(
        f"Inference complete. {len(df)} records processed. Saved -> {output_csv_path}"
    )


if __name__ == "__main__":
    run_batch_inference(
        input_csv_path="../data/test.csv",
        output_csv_path="../data/predicted_disaster_tweets.csv",
    )
