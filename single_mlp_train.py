import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from model.single_mlp import SingleTaskDataset, SingleTaskMLP
from sklearn.metrics import accuracy_score, f1_score


# 计算准确率和F1分数的函数
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # 评估时不需要计算梯度
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return accuracy, f1


# 训练函数
def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device, epochs=20):
    model.train()
    model.to(device)  # 将模型部署到 GPU
    for epoch in range(epochs):
        total_loss = 0
        all_labels = []
        all_predictions = []

        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)  # 数据转到 GPU
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算训练集的准确率
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # 计算训练集的准确率和F1分数
        train_accuracy = accuracy_score(all_labels, all_predictions)
        train_f1 = f1_score(all_labels, all_predictions, average='weighted')

        # 计算测试集的准确率和F1分数
        test_accuracy, test_f1 = evaluate_model(model, test_dataloader, device)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}, "
              f"Train Accuracy: {train_accuracy * 100:.2f}%, Train F1: {train_f1:.4f}, "
              f"Test Accuracy: {test_accuracy * 100:.2f}%, Test F1: {test_f1:.4f}")


# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有 GPU
    print(f"Using device: {device}")

    csv_train_file = 'data/normalized_train.csv'
    csv_test_file = 'data/normalized_test.csv'  # 测试集文件路径

    # 训练表情识别模型
    print("=== Training Emotion Recognition Model ===")
    emotion_train_dataset = SingleTaskDataset(csv_train_file, target_column='emotion')
    emotion_test_dataset = SingleTaskDataset(csv_test_file, target_column='emotion')  # 测试集
    emotion_train_loader = DataLoader(emotion_train_dataset, batch_size=32, shuffle=True)
    emotion_test_loader = DataLoader(emotion_test_dataset, batch_size=32, shuffle=False)

    emotion_input_dim = 58
    emotion_hidden_dim = 128
    emotion_output_dim = len(pd.read_csv(csv_train_file)['emotion'].unique())
    emotion_model = SingleTaskMLP(emotion_input_dim, emotion_hidden_dim, emotion_output_dim)
    emotion_criterion = nn.CrossEntropyLoss()
    emotion_optimizer = optim.Adam(emotion_model.parameters(), lr=0.001)

    train_model(emotion_model, emotion_train_loader, emotion_test_loader, emotion_criterion, emotion_optimizer, device, epochs=5)

    # 训练专注度识别模型
    print("\n=== Training Focus Recognition Model ===")
    focus_train_dataset = SingleTaskDataset(csv_train_file, target_column='if_focus')
    focus_test_dataset = SingleTaskDataset(csv_test_file, target_column='if_focus')  # 测试集
    focus_train_loader = DataLoader(focus_train_dataset, batch_size=32, shuffle=True)
    focus_test_loader = DataLoader(focus_test_dataset, batch_size=32, shuffle=False)

    focus_input_dim = 58
    focus_hidden_dim = 128
    focus_output_dim = len(pd.read_csv(csv_train_file)['if_focus'].unique())
    focus_model = SingleTaskMLP(focus_input_dim, focus_hidden_dim, focus_output_dim)
    focus_criterion = nn.CrossEntropyLoss()
    focus_optimizer = optim.Adam(focus_model.parameters(), lr=0.001)

    train_model(focus_model, focus_train_loader, focus_test_loader, focus_criterion, focus_optimizer, device, epochs=5)

    # 保存两个任务的模型到一个文件
    print("\n=== Saving the models as a single .pth file ===")
    model_save_path = 'pth_normalized/single_mlp_model.pth'

    # 合并两个模型的state_dict
    model_state_dict = {
        'emotion_model_state_dict': emotion_model.state_dict(),
        'focus_model_state_dict': focus_model.state_dict()
    }

    # 保存模型
    torch.save(model_state_dict, model_save_path)
    print(f"Models saved as '{model_save_path}'")


if __name__ == '__main__':
    main()
