import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# TensorBoard
writer = SummaryWriter('runs/vision_transformer_moe')

# Transformaciones para normalizar y aumentar los datos
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),  # Aumento de datos: Flip horizontal
    transforms.RandomRotation(10),      # Aumento de datos: Rotación aleatoria
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Aumento: Variación de color
    transforms.ToTensor()
])

# Carga del dataset CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Aumento del tamaño del batch

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Visualización de algunos ejemplos de imágenes del dataset en TensorBoard
data_iter = iter(train_loader)
images, labels = next(data_iter)
img_grid = vutils.make_grid(images[:8], normalize=True)
writer.add_image('CIFAR10_images', img_grid)

# Modelo Vision Transformer MoE
class Expert(nn.Module):
    def __init__(self, d_model):
        super(Expert, self).__init__()
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        return F.relu(self.fc(x))

class MoE(nn.Module):
    def __init__(self, d_model, num_experts=4, k=3):  # Aumento de expertos y top-k
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.k = k

    def forward(self, x):
        gate_outputs = self.gate(x)  # Obtener las salidas del gate
        top_k = torch.topk(gate_outputs, self.k, dim=-1).indices  # Seleccionar los top-k expertos

        batch_size = x.size(0)
        expert_outputs = torch.zeros(batch_size, x.size(1), x.size(2)).to(x.device)

        for batch_idx in range(batch_size):  # Iterar sobre el tamaño del batch
            selected_experts = top_k[batch_idx]  # Obtener los expertos seleccionados para este batch

            expert_outputs_batch = []
            for i in selected_experts:  # Iterar sobre los expertos seleccionados
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

# Instancia del modelo
model = VisionTransformerMoE(image_size=32, patch_size=4, num_classes=10, d_model=64, num_experts=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # Aumento de la tasa de aprendizaje
criterion = nn.CrossEntropyLoss()

# Graficar el modelo en TensorBoard
writer.add_graph(model, images)

# Función para entrenar y evaluar
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):  # Entrenar por num_epochs épocas
        running_loss = 0.0
        correct = 0
        total = 0

        # Entrenamiento
        model.train()
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
            writer.add_scalar('Training Loss Batch', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Training Accuracy Batch', correct / total, epoch * len(train_loader) + batch_idx)

        # Promedio de pérdida y precisión por época
        train_accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {train_accuracy}%")

        # Pérdida y precisión por época
        writer.add_scalar('Training Loss Epoch', avg_loss, epoch+1)
        writer.add_scalar('Training Accuracy Epoch', train_accuracy, epoch+1)

        # Distribución de pesos
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch+1)

        # Evaluación en el set de test
        evaluate_model(model, test_loader, criterion, epoch)

# Función para evaluación
def evaluate_model(model, test_loader, criterion, epoch):
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

    # Pérdida y precisión de test por época
    writer.add_scalar('Test Loss', avg_loss, epoch+1)
    writer.add_scalar('Test Accuracy', test_accuracy, epoch+1)

# Proyección 3D de imágenes usando embeddings
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


# Entrenamiento y evaluación
train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=2)

# Visualización de embeddings en 3D
visualize_embeddings(model, train_loader, writer)

# Cerrar TensorBoard al final
writer.close()