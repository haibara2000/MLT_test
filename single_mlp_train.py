import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from model.single_mlp import SingleTaskDataset, SingleTaskMLP
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# 训练函数
def train_model(model, dataloader, criterion, optimizer, device, epochs=20):
    model.train()
    model.to(device)  # 将模型部署到 GPU
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)  # 数据转到 GPU
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有 GPU
    print(f"Using device: {device}")

    csv_train_file = 'data/train.csv'  # 替换为您的训练集 CSV 文件路径

    # 训练表情识别模型
    print("=== Training Emotion Recognition Model ===")
    emotion_train_dataset = SingleTaskDataset(csv_train_file, target_column='emotion')
    emotion_train_loader = DataLoader(emotion_train_dataset, batch_size=32, shuffle=True)

    emotion_input_dim = 58
    emotion_hidden_dim = 128
    emotion_output_dim = len(pd.read_csv(csv_train_file)['emotion'].unique())
    emotion_model = SingleTaskMLP(emotion_input_dim, emotion_hidden_dim, emotion_output_dim)
    emotion_criterion = nn.CrossEntropyLoss()
    emotion_optimizer = optim.Adam(emotion_model.parameters(), lr=0.001)

    train_model(emotion_model, emotion_train_loader, emotion_criterion, emotion_optimizer, device, epochs=20)
    torch.save(emotion_model.state_dict(), 'pth/emotion_model.pth')
    print("Emotion model saved as 'emotion_mlp_model.pth'")

    # 训练专注度识别模型
    print("\n=== Training Focus Recognition Model ===")
    focus_train_dataset = SingleTaskDataset(csv_train_file, target_column='if_focus')
    focus_train_loader = DataLoader(focus_train_dataset, batch_size=32, shuffle=True)

    focus_input_dim = 58
    focus_hidden_dim = 128
    focus_output_dim = len(pd.read_csv(csv_train_file)['if_focus'].unique())
    focus_model = SingleTaskMLP(focus_input_dim, focus_hidden_dim, focus_output_dim)
    focus_criterion = nn.CrossEntropyLoss()
    focus_optimizer = optim.Adam(focus_model.parameters(), lr=0.001)

    train_model(focus_model, focus_train_loader, focus_criterion, focus_optimizer, device, epochs=20)
    torch.save(focus_model.state_dict(), 'pth/focus_model.pth')
    print("Focus model saved as 'pth/focus_mlp_model.pth'")

if __name__ == '__main__':
    main()
