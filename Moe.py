import torch
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.empty_cache()


class Expert(nn.Module):
    def __init__(self, d_model):
        super(Expert, self).__init__()
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        return F.relu(self.fc(x))
    
class MoE(nn.Module):
    def __init__(self, d_model, num_experts=4, k=2):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.k = k

    def forward(self, x):
        gate_outputs = self.gate(x)
        top_k_indices = torch.topk(gate_outputs, self.k, dim=-1).indices

        # Convert top_k_indices to a list of lists, then flatten to 1D tensor
        top_k_indices = top_k_indices.view(-1).tolist()
        top_k_indices = torch.tensor(top_k_indices, device=x.device).long()

        expert_outputs = [self.experts[i](x) for i in top_k_indices]
        expert_outputs = torch.stack(expert_outputs)

        # Compute mean of expert outputs along the first dimension
        return torch.mean(expert_outputs, dim=0)

class VisionTransformerMoE(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, d_model, num_experts):
        super(VisionTransformerMoE, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed = nn.Linear(patch_size*patch_size*3, d_model)
        self.position_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.moe = MoE(d_model, num_experts=num_experts)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.num_patches, -1)
        x = self.embed(patches) + self.position_embed
        x = self.moe(x)
        x = x.mean(dim=1)
        return self.classifier(x)


import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Transformaciones para normalizar y aumentar los datos
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# Carga del dataset CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Entrenamiento básico
model = VisionTransformerMoE(image_size=32, patch_size=4, num_classes=10, d_model=64, num_experts=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
for epoch in range(1):  # Entrenar por 10 épocas
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
