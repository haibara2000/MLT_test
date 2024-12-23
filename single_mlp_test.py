import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from model.single_mlp import SingleTaskDataset, SingleTaskMLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 测试函数
def evaluate_model(model, dataloader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)  # 数据转到 GPU
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            # 将 GPU 张量转为 NumPy
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return acc, f1


# 主函数
def main():
    # 加载模型和数据到 GPU
    # csv_test_file = 'data/normalized_test.csv'
    csv_test_file = 'data/test.csv'

    # 加载模型
    print("=== Loading the single_mlp_model ===")
    checkpoint = torch.load('pth_origin/single_mlp_model.pth')  # 加载保存的模型文件

    emotion_input_dim = 58
    emotion_hidden_dim = 128
    emotion_output_dim = len(pd.read_csv(csv_test_file)['emotion'].unique())  # 表情类别数
    emotion_model = SingleTaskMLP(emotion_input_dim, emotion_hidden_dim, emotion_output_dim)

    focus_input_dim = 58
    focus_hidden_dim = 128
    focus_output_dim = len(pd.read_csv(csv_test_file)['if_focus'].unique())  # 专注度类别数
    focus_model = SingleTaskMLP(focus_input_dim, focus_hidden_dim, focus_output_dim)

    # 加载保存的模型参数
    emotion_model.load_state_dict(checkpoint['emotion_model_state_dict'])
    focus_model.load_state_dict(checkpoint['focus_model_state_dict'])

    # 将模型转移到 GPU
    emotion_model.to(device)
    focus_model.to(device)

    # 测试表情识别模型
    print("\n=== Testing Emotion Recognition Model ===")
    emotion_test_dataset = SingleTaskDataset(csv_test_file, target_column='emotion')
    emotion_test_loader = DataLoader(emotion_test_dataset, batch_size=32, shuffle=False)
    evaluate_model(emotion_model, emotion_test_loader)

    # 测试专注度识别模型
    print("\n=== Testing Focus Recognition Model ===")
    focus_test_dataset = SingleTaskDataset(csv_test_file, target_column='if_focus')
    focus_test_loader = DataLoader(focus_test_dataset, batch_size=32, shuffle=False)
    evaluate_model(focus_model, focus_test_loader)


if __name__ == '__main__':
    main()
