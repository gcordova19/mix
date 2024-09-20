import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# TensorBoard
writer = SummaryWriter('runs/vision_transformer_moe_random_search')

# Definición de los expertos
class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modelo MoE adaptativo con RL
class AdaptiveMoEWithRL(nn.Module):
    def __init__(self, num_experts, input_size, hidden_size, output_size):
        super(AdaptiveMoEWithRL, self).__init__()
        self.experts = nn.ModuleList([Expert(input_size, hidden_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)
        self.rl_agent = RLAgent(input_size, hidden_size, num_experts)  # Agente RL integrado

    def forward(self, x):
        gate_weights = torch.softmax(self.gate(x), dim=1)  # Gate adaptativo
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Ajustar las dimensiones de la multiplicación de matrices
        output = torch.bmm(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)
        return output

    def update_rl_agent(self, state, reward):
        action_probs = self.rl_agent(state)
        reward = torch.tensor([reward], dtype=torch.float32)
        loss = -torch.sum(action_probs * reward)  # Pérdida basada en recompensa
        self.rl_agent.optimizer.zero_grad()
        loss.backward()
        self.rl_agent.optimizer.step()

# Agente de aprendizaje por refuerzo
class RLAgent(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts):
        super(RLAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)  # Probabilidades de selección de expertos

# Función para computar la recompensa
def compute_reward(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

# Función para evaluación
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            # Aplanar imágenes para entrada
            images = images.view(images.size(0), -1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(dataloader), correct / total

# Función para guardar el mejor modelo
def save_best_model(model, path):
    torch.save(model.state_dict(), path)

# Carga del dataset CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Inicialización del modelo, función de pérdida y optimizadores
num_experts = 10
input_size = 3 * 32 * 32  # CIFAR-10 imágenes 32x32 con 3 canales
hidden_size = 256
output_size = 10  # 10 clases en CIFAR-10

model = AdaptiveMoEWithRL(num_experts, input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 100
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        # Aplanar imágenes para entrada
        images = images.view(images.size(0), -1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Actualizar el agente de RL
        reward = compute_reward(outputs, labels)
        model.update_rl_agent(images, reward)

        running_loss += loss.item()

    # Evaluar el modelo en el conjunto de prueba
    test_loss, test_acc = evaluate(model, testloader, criterion)
    #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader):.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

    # Guardar el mejor modelo si la precisión mejora
    if test_acc > best_acc:
        best_acc = test_acc
        save_best_model(model, 'best_adaptive_moe_model.pth')

print(f"Mejor precisión obtenida: {best_acc:.4f}")
print('Entrenamiento completado.')



'''Files already downloaded and verified
Epoch 1/100, Loss: 1322.7687, Test Accuracy: 0.4587
Epoch 2/100, Loss: 1162.4194, Test Accuracy: 0.4733
Epoch 3/100, Loss: 1074.3943, Test Accuracy: 0.4723
Epoch 4/100, Loss: 996.2244, Test Accuracy: 0.4795
Epoch 5/100, Loss: 905.0177, Test Accuracy: 0.4819
Epoch 6/100, Loss: 833.5566, Test Accuracy: 0.4761
Epoch 7/100, Loss: 761.4061, Test Accuracy: 0.4810
Epoch 8/100, Loss: 685.8057, Test Accuracy: 0.4759
Epoch 9/100, Loss: 629.3803, Test Accuracy: 0.4836
Epoch 10/100, Loss: 573.4215, Test Accuracy: 0.4773
Epoch 11/100, Loss: 539.1144, Test Accuracy: 0.4847
Epoch 12/100, Loss: 487.1771, Test Accuracy: 0.4774
Epoch 13/100, Loss: 450.0067, Test Accuracy: 0.4709
Epoch 14/100, Loss: 443.6499, Test Accuracy: 0.4770
Epoch 15/100, Loss: 409.3512, Test Accuracy: 0.4855
Epoch 16/100, Loss: 360.5540, Test Accuracy: 0.4773
Epoch 17/100, Loss: 347.6736, Test Accuracy: 0.4737
Epoch 18/100, Loss: 336.8716, Test Accuracy: 0.4793
Epoch 19/100, Loss: 310.3660, Test Accuracy: 0.4792
Epoch 20/100, Loss: 303.2366, Test Accuracy: 0.4686
Epoch 21/100, Loss: 296.0574, Test Accuracy: 0.4670
Epoch 22/100, Loss: 282.0149, Test Accuracy: 0.4755
Epoch 23/100, Loss: 255.5698, Test Accuracy: 0.4778
Epoch 24/100, Loss: 254.0205, Test Accuracy: 0.4739
Epoch 25/100, Loss: 235.3929, Test Accuracy: 0.4731
Epoch 26/100, Loss: 244.8474, Test Accuracy: 0.4694
Epoch 27/100, Loss: 232.2793, Test Accuracy: 0.4667
Epoch 28/100, Loss: 232.3181, Test Accuracy: 0.4678
Epoch 29/100, Loss: 215.2491, Test Accuracy: 0.4703
Epoch 30/100, Loss: 185.4659, Test Accuracy: 0.4724
Epoch 31/100, Loss: 211.2569, Test Accuracy: 0.4654
Epoch 32/100, Loss: 208.6401, Test Accuracy: 0.4727
Epoch 33/100, Loss: 193.6511, Test Accuracy: 0.4680
Epoch 34/100, Loss: 166.4832, Test Accuracy: 0.4683
Epoch 35/100, Loss: 195.3860, Test Accuracy: 0.4648
Epoch 36/100, Loss: 184.8643, Test Accuracy: 0.4648
Epoch 37/100, Loss: 164.1329, Test Accuracy: 0.4699
Epoch 38/100, Loss: 168.8345, Test Accuracy: 0.4674
Epoch 39/100, Loss: 186.6978, Test Accuracy: 0.4667
Epoch 40/100, Loss: 145.1108, Test Accuracy: 0.4645
Epoch 41/100, Loss: 153.4638, Test Accuracy: 0.4549
Epoch 42/100, Loss: 157.8742, Test Accuracy: 0.4710
Epoch 43/100, Loss: 137.2310, Test Accuracy: 0.4646
Epoch 44/100, Loss: 182.0388, Test Accuracy: 0.4609
Epoch 45/100, Loss: 120.3041, Test Accuracy: 0.4665
Epoch 46/100, Loss: 137.6738, Test Accuracy: 0.4678
Epoch 47/100, Loss: 132.7457, Test Accuracy: 0.4674
Epoch 48/100, Loss: 146.8789, Test Accuracy: 0.4670
Epoch 49/100, Loss: 131.9624, Test Accuracy: 0.4634
Epoch 50/100, Loss: 168.0005, Test Accuracy: 0.4650
Epoch 51/100, Loss: 126.4415, Test Accuracy: 0.4692
Epoch 52/100, Loss: 97.1059, Test Accuracy: 0.4752
Epoch 53/100, Loss: 130.0673, Test Accuracy: 0.4654
Epoch 54/100, Loss: 142.4498, Test Accuracy: 0.4597
Epoch 55/100, Loss: 138.5440, Test Accuracy: 0.4632
Epoch 56/100, Loss: 126.7738, Test Accuracy: 0.4637
Epoch 57/100, Loss: 114.1024, Test Accuracy: 0.4645
Epoch 58/100, Loss: 112.0877, Test Accuracy: 0.4661
Epoch 59/100, Loss: 107.8239, Test Accuracy: 0.4683
Epoch 60/100, Loss: 109.7029, Test Accuracy: 0.4547
Epoch 61/100, Loss: 122.2463, Test Accuracy: 0.4685
Epoch 62/100, Loss: 109.7085, Test Accuracy: 0.4626
Epoch 63/100, Loss: 128.1930, Test Accuracy: 0.4668
Epoch 64/100, Loss: 125.0470, Test Accuracy: 0.4681
Epoch 65/100, Loss: 94.4657, Test Accuracy: 0.4634
Epoch 66/100, Loss: 102.2812, Test Accuracy: 0.4631
Epoch 67/100, Loss: 115.0893, Test Accuracy: 0.4645
Epoch 68/100, Loss: 91.8046, Test Accuracy: 0.4710
Epoch 69/100, Loss: 91.6414, Test Accuracy: 0.4652
Epoch 70/100, Loss: 81.6987, Test Accuracy: 0.4718
Epoch 71/100, Loss: 119.2912, Test Accuracy: 0.4621
Epoch 72/100, Loss: 121.7782, Test Accuracy: 0.4628
Epoch 73/100, Loss: 113.7735, Test Accuracy: 0.4594
Epoch 74/100, Loss: 93.1696, Test Accuracy: 0.4683
Epoch 75/100, Loss: 113.2364, Test Accuracy: 0.4616
Epoch 76/100, Loss: 95.6001, Test Accuracy: 0.4608
Epoch 77/100, Loss: 90.3440, Test Accuracy: 0.4699
Epoch 78/100, Loss: 79.4530, Test Accuracy: 0.4603
Epoch 79/100, Loss: 110.9242, Test Accuracy: 0.4616
Epoch 80/100, Loss: 117.6558, Test Accuracy: 0.4628
Epoch 81/100, Loss: 97.1440, Test Accuracy: 0.4665
Epoch 82/100, Loss: 84.1126, Test Accuracy: 0.4608
Epoch 83/100, Loss: 74.5261, Test Accuracy: 0.4695
Epoch 84/100, Loss: 103.1843, Test Accuracy: 0.4642
Epoch 85/100, Loss: 96.2729, Test Accuracy: 0.4727
Epoch 86/100, Loss: 91.0164, Test Accuracy: 0.4666
Epoch 87/100, Loss: 79.0221, Test Accuracy: 0.4682
Epoch 88/100, Loss: 80.5605, Test Accuracy: 0.4676
Epoch 89/100, Loss: 86.0994, Test Accuracy: 0.4662
Epoch 90/100, Loss: 110.6448, Test Accuracy: 0.4715
Epoch 91/100, Loss: 105.5419, Test Accuracy: 0.4701
Epoch 92/100, Loss: 77.1957, Test Accuracy: 0.4717
Epoch 93/100, Loss: 72.7461, Test Accuracy: 0.4649
Epoch 94/100, Loss: 100.0919, Test Accuracy: 0.4561
Epoch 95/100, Loss: 86.9367, Test Accuracy: 0.4692
Epoch 96/100, Loss: 71.1758, Test Accuracy: 0.4699
Epoch 97/100, Loss: 86.5060, Test Accuracy: 0.4653
Epoch 98/100, Loss: 94.7088, Test Accuracy: 0.4638
Epoch 99/100, Loss: 67.2135, Test Accuracy: 0.4687
Epoch 100/100, Loss: 108.3541, Test Accuracy: 0.4619
Mejor precisión obtenida: 0.4855 '''