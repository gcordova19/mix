import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from old.mixture_of_experts import MoE  
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super(PatchEmbedding, self).__init__()
        num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)  
    def forward(self, x):
        x = self.proj(x)
        x = self.flatten(x)
        return x.transpose(1, 2)  

class VisionMoE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=128, num_heads=8, num_experts=4, expert_dim=256, num_classes=10):
        super(VisionMoE, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans=3, embed_dim=embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.moe_layer = MoE(dim=embed_dim, num_experts=num_experts, hidden_dim=expert_dim)
        self.fc = nn.Linear(expert_dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)  
        x = self.transformer_layer(x)
        x, aux_loss = self.moe_layer(x)
        return self.fc(x.mean(dim=1)), aux_loss

def train_model():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = VisionMoE()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, aux_loss = model(inputs)
            loss = criterion(outputs, labels) + aux_loss
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

if __name__ == "__main__":
    train_model()
