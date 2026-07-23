import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class DisasterTweetClassifier(nn.Module):
    """Simple Feedforward Neural Network for binary text classification."""

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
    """Standardizes text string by lowercasing and removing punctuation."""
    text = str(text).lower()
    return "".join([c for c in text if c.isalnum() or c == " "])


def main():
    # Setup paths
    os.makedirs("../artifacts", exist_ok=True)
    data_path = "../data/train.csv"

    # 1. Ingestion
    df = pd.read_csv(data_path)

    # 2. Preprocessing
    df["clean_text"] = df["text"].apply(clean_text)

    # 3. Feature Extraction
    vectorizer = TfidfVectorizer(max_features=1000)
    X_num = vectorizer.fit_transform(df["clean_text"]).toarray()
    y_num = df["target"].values

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y_num, test_size=0.2, random_state=42, stratify=y_num
    )

    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 5. Initialization
    model = DisasterTweetClassifier(input_dim=1000, hidden_dim=16)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 6. Optimization Loop
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Loss: {loss.item():.4f}")

    # 7. Model Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        binary_preds = (test_preds >= 0.5).float()
        acc = (binary_preds == y_test_t).sum().item() / y_test_t.size(0)

    print(f"\nFinal Out-of-Sample Accuracy: {acc * 100:.2f}%")

    # 8. Persist Artifacts
    torch.save(model.state_dict(), "../artifacts/disaster_model.pth")
    joblib.dump(vectorizer, "../artifacts/tfidf_vectorizer.pkl")
    print("Pipeline artifacts saved to ../artifacts/")


if __name__ == "__main__":
    main()
