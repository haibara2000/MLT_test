import torch
import torch.nn as nn

class MMoECNNModel(nn.Module):
    def __init__(self, emotion_output_dim, focus_output_dim, num_experts=4):
        super(MMoECNNModel, self).__init__()
        self.num_experts = num_experts

        # 定义专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 输出 (16, 58, 30)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (16, 29, 15)
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输出 (32, 29, 15)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (32, 14, 7)
                nn.Flatten()  # 输出 (32 * 14 * 7)
            ) for _ in range(num_experts)
        ])

        # 动态获取专家输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 58, 30)
            self.expert_output_dim = self.experts[0](dummy_input).shape[-1]

        # 定义门控网络
        self.gates_emotion = nn.Sequential(
            nn.Linear(self.expert_output_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        self.gates_focus = nn.Sequential(
            nn.Linear(self.expert_output_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        # 定义任务分支
        self.emotion_branch = nn.Sequential(
            nn.Linear(self.expert_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emotion_output_dim)
        )
        self.focus_branch = nn.Sequential(
            nn.Linear(self.expert_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, focus_output_dim)
        )

    def forward(self, x):
        # 获取专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts],
                                     dim=1)  # (batch_size, num_experts, expert_output_dim)

        # 平均聚合用于门控网络
        flat_x = expert_outputs.mean(dim=1)

        # 门控网络生成权重
        emotion_gate_weights = self.gates_emotion(flat_x)
        focus_gate_weights = self.gates_focus(flat_x)

        # 加权求和专家输出
        emotion_output = torch.sum(emotion_gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        focus_output = torch.sum(focus_gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        # 通过任务分支
        emotion_output = self.emotion_branch(emotion_output)
        focus_output = self.focus_branch(focus_output)

        return emotion_output, focus_output

