import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
from torchvision import models  # Import models here
from PIL import Image
import string

# Define the character set
all_chars = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Define the dataset
class CaptchaDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label = image_file.split("_")[0]  # Extract label from filename
        label_indices = torch.tensor([char_to_idx[char] for char in label], dtype=torch.long)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label_indices

# Define preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the model
class CaptchaSolver(nn.Module):
    def __init__(self, num_classes, max_seq_length):
        super(CaptchaSolver, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 8 * 8, 512)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=4
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8), num_layers=6
        )
        self.fc_out = nn.Linear(512, num_classes)
        self.max_seq_length = max_seq_length

    def forward(self, images, target_seq=None):
        features = self.feature_extractor(images)
        features = self.flatten(features)
        features = self.fc(features).unsqueeze(0)
        memory = self.encoder(features)
        outputs = []
        decoder_input = torch.zeros(1, 1, 512).to(features.device)
        for _ in range(self.max_seq_length):
            output = self.decoder(decoder_input, memory)
            char_logits = self.fc_out(output[-1])
            outputs.append(char_logits)
            if target_seq is not None:
                next_input = target_seq[:, len(outputs) - 1].unsqueeze(0)
            else:
                next_input = char_logits.argmax(dim=-1).unsqueeze(0)
            decoder_input = torch.cat([decoder_input, next_input], dim=0)
        return torch.stack(outputs, dim=1)

# Training and evaluation
if __name__ == "__main__":
    # Dataset and DataLoader
    dataset = CaptchaDataset(image_dir="captchas", transform=transform)
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model, loss, and optimizer
    num_classes = len(all_chars)
    max_seq_length = 10
    model = CaptchaSolver(num_classes=num_classes, max_seq_length=max_seq_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10):  # Number of epochs
        for images, labels in train_loader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(images, target_seq=labels)
            loss = sum(criterion(output, labels[:, i]) for i, output in enumerate(outputs))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=-1)
            predicted_text = "".join(idx_to_char[idx.item()] for idx in predicted.squeeze())
            true_text = "".join(idx_to_char[idx.item()] for idx in labels.squeeze())
            if predicted_text == true_text:
                correct += 1
            total += 1
    print(f"Accuracy: {correct / total * 100:.2f}%")