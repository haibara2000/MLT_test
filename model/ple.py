import torch
import torch.nn as nn
import torch.nn.functional as F

class CGCLayer(nn.Module):
    def __init__(self, num_shared_experts, num_task_experts, num_tasks):
        super(CGCLayer, self).__init__()
        self.num_tasks = num_tasks

        # 定义共享专家网络，使用卷积操作
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 输出 (16, 58, 30)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (16, 29, 15)
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输出 (32, 29, 15)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (32, 14, 7)
            ) for _ in range(num_shared_experts)
        ])

        # 定义任务专家网络，使用卷积操作
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 输出 (16, 58, 30)
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (16, 29, 15)
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输出 (32, 29, 15)
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (32, 14, 7)
                ) for _ in range(num_task_experts)
            ]) for _ in range(num_tasks)
        ])

        # 用于动态计算卷积层输出的大小
        self.dummy_input = torch.randn(1, 1, 58, 30)
        self.dummy_output = self.shared_experts[0](self.dummy_input)  # 使用第一个共享专家计算输出
        self.expert_output_dim = self.dummy_output.view(1, -1).shape[1]  # 计算展平后的维度

        # 门控网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),  # 直接展平原始输入
                nn.Linear(self.dummy_input.numel(), num_shared_experts + num_task_experts, bias=False),  # 去掉 bias
                nn.Softmax(dim=-1)  # Softmax 层，输出为专家的权重
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        # 获取共享专家输出，保持四维形状
        shared_outputs = torch.stack([expert(x) for expert in self.shared_experts],
                                     dim=1)  # (batch_size, num_shared_experts, channels, height, width)

        # 获取每个任务特定专家输出
        task_outputs = [
            torch.stack([expert(x) for expert in task_experts], dim=1)
            # (batch_size, num_task_experts, channels, height, width)
            for task_experts in self.task_experts
        ]

        # 每个任务根据门控网络加权组合共享和特定专家
        outputs = []
        for task_id in range(self.num_tasks):
            # 平均聚合用于门控网络
            expert_outputs = torch.cat([shared_outputs, task_outputs[task_id]],
                                       dim=1)  # (batch_size, total_experts, channels, height, width)
            flat_x = expert_outputs.view(expert_outputs.size(0),
                                         -1)  # 展平为 (batch_size, total_experts * channels * height * width)

            # 门控网络生成权重
            gate_weights = self.gates[task_id](flat_x)  # (batch_size, total_experts)

            # gate_weights 的形状为 (batch_size, total_experts)，需要扩展到 (batch_size, total_experts, 1, 1, 1) 才能与 expert_outputs 对齐
            gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(
                -1)  # (batch_size, total_experts, 1, 1, 1)

            # 加权求和专家输出
            weighted_output = torch.sum(gate_weights * expert_outputs, dim=1)  # (batch_size, channels, height, width)
            outputs.append(weighted_output)

        return outputs


class PLEModel(nn.Module):
    def __init__(self, emotion_output_dim, focus_output_dim, num_shared_experts=1, num_task_experts=1, num_tasks=2):
        super(PLEModel, self).__init__()

        # 第一层 CGC
        self.cgc_layer1 = CGCLayer(num_shared_experts, num_task_experts, num_tasks)

        # 第二层 CGC
        self.cgc_layer2 = CGCLayer(num_shared_experts, num_task_experts, num_tasks)

        # 任务特定输出层
        self.emotion_task_layer = nn.Sequential(
            nn.Linear(self.cgc_layer1.expert_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emotion_output_dim)
        )
        self.focus_task_layer = nn.Sequential(
            nn.Linear(self.cgc_layer1.expert_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, focus_output_dim)
        )

    def forward(self, x):
        # 输入通过第一层 CGC
        emotion_out1, focus_out1 = self.cgc_layer1(x)

        # 输出通过第二层 CGC
        emotion_out2, focus_out2 = self.cgc_layer2(emotion_out1), self.cgc_layer2(focus_out1)

        # 任务特定输出
        emotion_output = self.emotion_task_layer(emotion_out2.view(emotion_out2.size(0), -1))  # 展平
        focus_output = self.focus_task_layer(focus_out2.view(focus_out2.size(0), -1))  # 展平

        return emotion_output, focus_output
