import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# TensorBoard
writer = SummaryWriter('runs/vision_transformer_moe_random_search')

# Aumento de datos para normalizar y aumentar la variabilidad del dataset
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Carga del dataset CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Hyperparameter search space
param_grid = {
    'lr': [0.002, 0.005, 0.0005, 0.0001],
    'num_experts': [2, 4],
    'k': [2, 3],#, 3, 4],
    'batch_size': [32,64],#, 64, 128],
    'epochs': [10, 20]
}

# Modelo Vision Transformer MoE
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
        top_k = torch.topk(gate_outputs, self.k, dim=-1).indices
        batch_size = x.size(0)
        expert_outputs = torch.zeros(batch_size, x.size(1), x.size(2)).to(x.device)

        for batch_idx in range(batch_size):
            selected_experts = top_k[batch_idx]
            expert_outputs_batch = []
            for i in selected_experts:
                if i.numel() == 1:
                    expert_idx = i.item()
                else:
                    expert_idx = i[0].item()

                expert_output = self.experts[expert_idx](x[batch_idx].unsqueeze(0))
                expert_outputs_batch.append(expert_output)

            expert_outputs_batch = torch.stack(expert_outputs_batch)
            expert_outputs[batch_idx] = torch.mean(expert_outputs_batch, dim=0)

        return expert_outputs

class VisionTransformerMoE(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, d_model, num_experts):
        super(VisionTransformerMoE, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed = nn.Linear(patch_size * patch_size * 3, d_model)
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

# Función para entrenar y evaluar
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, lr=None):
    best_accuracy = 0.0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Pérdida y precisión por mini-batch
            writer.add_scalar(f'Training Loss Batch_lr_{lr}', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar(f'Training Accuracy Batch_lr_{lr}', correct / total, epoch * len(train_loader) + batch_idx)

        train_accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {train_accuracy}%")
        writer.add_scalar(f'Training Loss_lr_{lr}', avg_loss, epoch+1)
        writer.add_scalar(f'Training Accuracy_lr_{lr}', train_accuracy, epoch+1)

        test_accuracy = evaluate_model(model, test_loader, criterion, epoch, lr)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model.state_dict()

        # Guardar distribuciones de pesos
        for name, param in model.named_parameters():
            writer.add_histogram(f'{name}_lr_{lr}', param, epoch+1)

    return best_model, best_accuracy

# Función para evaluación
def evaluate_model(model, test_loader, criterion, epoch, lr):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100. * correct / total
    avg_loss = running_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}, Test Accuracy: {test_accuracy}%")
    writer.add_scalar(f'Test Loss_lr_{lr}', avg_loss, epoch+1)
    writer.add_scalar(f'Test Accuracy_lr_{lr}', test_accuracy, epoch+1)

    return test_accuracy

# Función para visualizar imágenes aumentadas
def visualize_augmented_images(dataloader, writer):
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('Aumented Images', img_grid)

# Función para guardar el mejor modelo
def save_best_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Modelo guardado en {path}")

# Proyección 3D de embeddings usando TensorBoard
def visualize_embeddings(model, dataloader, writer):
    model.eval()
    embeddings = []
    labels = []
    images_list = []
    with torch.no_grad():
        for images, label in dataloader:
            output = model(images)
            embeddings.append(output)
            labels.append(label)
            images_list.append(images)
            if len(embeddings) * len(images) >= 100:  # Limitar el número de puntos de datos y imágenes
                break
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    images_list = torch.cat(images_list)
    
    # Asegúrate de tener 100 o menos imágenes
    num_images = min(100, images_list.size(0))
    writer.add_embedding(embeddings[:num_images], metadata=labels[:num_images], label_img=images_list[:num_images])

# Guardar combinaciones previas
combinaciones_usadas = set()

# Función para generar una combinación aleatoria no repetida
def generar_combinacion(param_grid, combinaciones_usadas):
    # Obtener todas las combinaciones posibles de hiperparámetros
    combinaciones_posibles = list(itertools.product(*param_grid.values()))
    
    # Convertir a un formato fácil de comparar y verificar
    while combinaciones_posibles:
        combinacion_aleatoria = random.choice(combinaciones_posibles)
        combinacion_tupla = tuple(combinacion_aleatoria)
        
        if combinacion_tupla not in combinaciones_usadas:
            combinaciones_usadas.add(combinacion_tupla)
            # Convertir la combinación de tupla a diccionario
            params = {key: combinacion_aleatoria[i] for i, key in enumerate(param_grid)}
            return params
        else:
            combinaciones_posibles.remove(combinacion_aleatoria)
    return None

# Búsqueda aleatoria de hiperparámetros sin repetición
best_model = None
best_accuracy = 0.0
max_iteraciones = 15

for _ in range(max_iteraciones):
    params = generar_combinacion(param_grid, combinaciones_usadas)
    if params is None:
        print("No quedan más combinaciones sin usar.")
        break

    print(f"Probando con configuración: {params}")

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    model = VisionTransformerMoE(image_size=32, patch_size=4, num_classes=10, d_model=128, num_experts=params['num_experts'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Visualizar imágenes aumentadas
    visualize_augmented_images(train_loader, writer)

    model_trained, accuracy = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=params['epochs'], lr=params['lr'])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model_trained
        save_best_model(model, 'best_vit_moe_model.pth')

    # Visualizar embeddings
    #visualize_embeddings(model, test_loader, writer)

print(f"Mejor precisión obtenida: {best_accuracy}%")
