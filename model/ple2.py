import torch
import torch.nn as nn

# 第一层Expert使用CNN架构
class Expert1(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(Expert1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=kernel_size, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 14 * 7, output_channels)  # 将CNN输出展平并进行线性变换

    def forward(self, x):
        x = self.conv1(x)  # (batch_size, 16, 58, 30)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch_size, 16, 29, 15)
        x = self.conv2(x)  # (batch_size, 32, 29, 15)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch_size, 32, 14, 7)
        x = self.flatten(x)  # (batch_size, 32 * 14 * 7)
        x = self.fc(x)  # (batch_size, output_channels)
        return x

# 后续的Expert使用线性层架构
class Expert2(nn.Module):
    def __init__(self, input_feature_size, expert_hidden_size, expert_output_size):
        super(Expert2, self).__init__()
        self.fc1 = nn.Linear(input_feature_size, expert_hidden_size)
        self.fc2 = nn.Linear(expert_hidden_size, expert_output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class CGC(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden, if_last, first_layer=True):
        super(CGC, self).__init__()

        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.if_last = if_last
        self.first_layer = first_layer  # 判断是否是第一层

        # 第一层使用CNN架构，后续层使用线性层
        if self.first_layer:
            self.experts_shared = nn.ModuleList([Expert1(1, self.experts_out) for _ in range(self.num_shared_experts)])   # 32 * 14 * 7
            self.experts_task1 = nn.ModuleList([Expert1(1, self.experts_out) for _ in range(self.num_specific_experts)])
            self.experts_task2 = nn.ModuleList([Expert1(1, self.experts_out) for _ in range(self.num_specific_experts)])
            gate_input_size = 58 * 30  # 第一层的门控网络输入维度来自CNN的输出
        else:
            self.experts_shared = nn.ModuleList([Expert2(self.input_size, self.experts_hidden, self.experts_out) for _ in
                                                range(self.num_shared_experts)])
            self.experts_task1 = nn.ModuleList([Expert2(self.input_size, self.experts_hidden, self.experts_out) for _ in
                                                range(self.num_specific_experts)])
            self.experts_task2 = nn.ModuleList([Expert2(self.input_size, self.experts_hidden, self.experts_out) for _ in
                                                range(self.num_specific_experts)])
            gate_input_size = self.input_size  # 后续层的门控网络输入维度来自线性层的输出

        # 定义门控网络，门控网络的输入维度根据上面动态计算
        self.gate_shared = nn.Sequential(
            nn.Linear(gate_input_size, self.num_specific_experts * 2 + self.num_shared_experts),
            nn.Softmax(dim=1),
            nn.Dropout(0.5)  # 在门控网络中加入 Dropout
        )
        self.gate_task1 = nn.Sequential(
            nn.Linear(gate_input_size, self.num_specific_experts + self.num_shared_experts),
            nn.Softmax(dim=1),
            nn.Dropout(0.5)  # 在门控网络中加入 Dropout
        )
        self.gate_task2 = nn.Sequential(
            nn.Linear(gate_input_size, self.num_specific_experts + self.num_shared_experts),
            nn.Softmax(dim=1),
            nn.Dropout(0.5)  # 在门控网络中加入 Dropout
        )

    def forward(self, x):
        inputs_shared, inputs_task1, inputs_task2 = x

        # # 确保输入是四维的，恢复为(batch_size, 1, 58, 30)
        # inputs_shared = inputs_shared.view(inputs_shared.size(0), 1, 58, 30)  # (batch_size, 1, 58, 30)
        # inputs_task1 = inputs_task1.view(inputs_task1.size(0), 1, 58, 30)  # (batch_size, 1, 58, 30)
        # inputs_task2 = inputs_task2.view(inputs_task2.size(0), 1, 58, 30)  # (batch_size, 1, 58, 30)

        # 计算专家网络的输出
        experts_shared_o = [e(inputs_shared) for e in self.experts_shared]
        experts_shared_o = torch.stack(experts_shared_o)

        experts_task1_o = [e(inputs_task1) for e in self.experts_task1]
        experts_task1_o = torch.stack(experts_task1_o)

        experts_task2_o = [e(inputs_task2) for e in self.experts_task2]
        experts_task2_o = torch.stack(experts_task2_o)

        # 仅对输入进行展平，作为Gate的输入
        inputs_shared_flatten = inputs_shared.view(inputs_shared.size(0), -1)  # 展平为 (batch_size, feature_size)
        inputs_task1_flatten = inputs_task1.view(inputs_task1.size(0), -1)  # 展平为 (batch_size, feature_size)
        inputs_task2_flatten = inputs_task2.view(inputs_task2.size(0), -1)  # 展平为 (batch_size, feature_size)

        # Gate 1
        selected_task1 = self.gate_task1(inputs_task1_flatten)
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)
        gate_task1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected_task1)

        # Gate 2
        selected_task2 = self.gate_task2(inputs_task2_flatten)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate_task2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected_task2)

        # Gate shared
        selected_shared = self.gate_shared(inputs_shared_flatten)
        gate_expert_outputshared = torch.cat((experts_task1_o, experts_task2_o, experts_shared_o), dim=0)
        gate_shared_out = torch.einsum('abc, ba -> bc', gate_expert_outputshared, selected_shared)

        # # 输出门控网络的权重
        # print("Gate Shared Weights:",selected_task1)  # 打印 gate_shared 权重
        # print("Gate Task1 Weights:", selected_task2)  # 打印 gate_task1 权重
        # print("Gate Task2 Weights:", selected_shared)  # 打印 gate_task2 权重

        if self.if_last:
            return [gate_task1_out, gate_task2_out]
        else:
            return [gate_shared_out, gate_task1_out, gate_task2_out]


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class PLEModel(nn.Module):
    def __init__(self, num_CGC_layers, input_size, emotion_output_dim, focus_output_dim, num_specific_experts, num_shared_experts, experts_out, experts_hidden, towers_hidden):
        super(PLEModel, self).__init__()
        self.num_CGC_layers = num_CGC_layers
        self.input_size = input_size
        self.emotion_output_dim = emotion_output_dim
        self.focus_output_dim = focus_output_dim
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden

        self.cgc_layer1 = CGC(self.input_size, self.num_specific_experts, self.num_shared_experts, self.experts_out, self.experts_hidden, if_last=False, first_layer=True)
        self.cgc_layers = nn.ModuleList([CGC(32, num_specific_experts, num_shared_experts, experts_out, experts_hidden, if_last=(i == num_CGC_layers - 1), first_layer=False) for i in range(num_CGC_layers)])

        self.tower1 = Tower(self.experts_out, self.emotion_output_dim, self.towers_hidden)  # 情绪任务分支
        self.tower2 = Tower(self.experts_out, self.focus_output_dim, self.towers_hidden)  # 专注度任务分支

    def forward(self, x):
        cgc_outputs = self.cgc_layer1([x, x, x])

        for cgc_layer in self.cgc_layers:
            cgc_outputs = cgc_layer(cgc_outputs)

        final_output1 = self.tower1(cgc_outputs[0])  # 输出情绪任务预测
        final_output2 = self.tower2(cgc_outputs[1])  # 输出专注度任务预测

        return [final_output1, final_output2]
