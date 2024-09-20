import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
import os

# Definir el modelo Mixtures of Experts (MoE)
class MixturesOfExperts(nn.Module):
    def __init__(self, num_experts, base_model, num_classes):
        super(MixturesOfExperts, self).__init__()
        self.base_model = base_model
        self.experts = nn.ModuleList([nn.Linear(base_model.num_features, num_classes) for _ in range(num_experts)])
        self.gate = nn.Linear(base_model.num_features, num_experts)
        self.classifier = nn.Linear(num_classes, num_classes)
    
    def forward(self, x):
        features = self.base_model.forward_features(x)
        gate_scores = self.gate(features)  # Dimensión: (batch_size, num_experts)
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)  # Dimensión: (batch_size, num_experts, num_classes)
        
        # Aplicar softmax en las gate_scores para obtener las ponderaciones
        gate_scores = torch.softmax(gate_scores, dim=1)  # Dimensión: (batch_size, num_experts)
        
        # Ajustar dimensiones para la multiplicación de matrices
        gate_scores = gate_scores.unsqueeze(2)  # Dimensión: (batch_size, num_experts, 1)
        expert_outputs = expert_outputs  # Dimensión: (batch_size, num_experts, num_classes)
        
        # Realizar la multiplicación de matrices
        mixed_output = torch.matmul(gate_scores.transpose(1, 2), expert_outputs)  # Dimensión: (batch_size, 1, num_classes)
        mixed_output = mixed_output.squeeze(1)  # Dimensión: (batch_size, num_classes)

        return self.classifier(mixed_output)

# Cargar el modelo ViT preentrenado
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)

# Configurar el modelo Mixtures of Experts (MoE)
num_experts = 4
moe_model = MixturesOfExperts(num_experts=num_experts, base_model=vit_model, num_classes=10)

# Transferir los pesos del modelo ViT al modelo MoE
def load_vit_weights_to_moe(vit_model, moe_model):
    vit_state_dict = vit_model.state_dict()
    moe_state_dict = moe_model.state_dict()
    for key in vit_state_dict:
        if key in moe_state_dict:
            moe_state_dict[key] = vit_state_dict[key]
    moe_model.load_state_dict(moe_state_dict)

load_vit_weights_to_moe(vit_model, moe_model)

# Configurar DataLoader
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Configurar optimizador, scheduler y loss
optimizer = optim.Adam(moe_model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
criterion = nn.CrossEntropyLoss()

# Configurar TensorBoard
writer = SummaryWriter('runs/MoE_vit')

# Función para visualizar imágenes en TensorBoard
def visualize_images_in_tensorboard(data_loader):
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('CIFAR10 Images', img_grid)

# Visualizar arquitectura del modelo en TensorBoard
dummy_input = torch.randn(1, 3, 224, 224)  # Tamaño de entrada para ViT
try:
    writer.add_graph(moe_model, dummy_input)
except Exception as e:
    print("Error while adding graph to TensorBoard:", e)

# Funciones de entrenamiento y evaluación
def train_epoch(model, data_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:
            writer.add_scalar('Training Loss', running_loss / 10, epoch * len(data_loader) + i)
            running_loss = 0.0

def evaluate_model(model, data_loader, criterion, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)
    writer.add_scalar('Test Loss', avg_loss, epoch)
    writer.add_scalar('Test Accuracy', accuracy, epoch)
    return avg_loss, accuracy

# Guardar el mejor modelo
best_accuracy = 0.0  # Para llevar un registro de la mejor precisión
save_path = './best_model_moevit.pth'  # Ruta donde guardar el mejor modelo

# Entrenar el modelo
num_epochs = 20
for epoch in range(num_epochs):
    train_epoch(moe_model, train_loader, optimizer, criterion, epoch)
    test_loss, test_accuracy = evaluate_model(moe_model, test_loader, criterion, epoch)
    scheduler.step()
    
    # Guardar el mejor modelo
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(moe_model.state_dict(), save_path)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Visualizar el conjunto de imágenes de entrenamiento
#visualize_images_in_tensorboard(train_loader)

# Cerrar el writer de TensorBoard
writer.close()

print("Training complete.")
