import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision import transforms
from torchvision.models import resnet18

from clearml import Task, OutputModel


# ============================================================
# 1. CLEARML INITIALIZATION
# ============================================================

task = Task.init(
    project_name="TorchScript-Demo",
    task_name="ResNet18-CIFAR10-Training",
    task_type=Task.TaskTypes.training,
    auto_connect_frameworks={
        "pytorch": False,
    },
    output_uri=True
)

logger = task.get_logger()


# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

params = {
    "epochs": 10,
    "batch_size": 128,
    "learning_rate": 0.001,
    "num_classes": 10,
    "architecture": "ResNet18",
    "optimizer": "Adam",
    "image_size": 64,
    "random_seed": 42,
}

# Register parameters in ClearML.
# They can be modified through the Web UI when cloning a task.

params = task.connect(params)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

torch.manual_seed(params["random_seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(params["random_seed"])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("CLEARML TRAINING")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"PyTorch version: {torch.__version__}")


# ============================================================
# 4. DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (params["image_size"], params["image_size"])
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])


train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

validation_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=params["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=params["batch_size"],
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 5. MODEL
# ============================================================

model = resnet18(weights=None)

# Adapt ResNet18 to CIFAR-10.

model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
)

model.maxpool = nn.Identity()

model.fc = nn.Linear(
    model.fc.in_features,
    params["num_classes"],
)

model = model.to(device)


# ============================================================
# 6. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=params["learning_rate"],
)


# ============================================================
# 7. CLEARML MODEL REGISTRATION
# ============================================================

output_model = OutputModel(
    task=task,
    name="ResNet18-CIFAR10-TorchScript",
    framework="PyTorch",
    tags=[
        "torchscript",
        "cifar10",
        "resnet18",
    ],
)


# ============================================================
# 8. TRAINING LOOP
# ============================================================

for epoch in range(params["epochs"]):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100.0 * correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            validation_total += labels.size(0)

            validation_correct += (
                predictions == labels
            ).sum().item()

    validation_loss /= len(validation_loader)

    validation_accuracy = (
        100.0 * validation_correct / validation_total
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{params['epochs']}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Loss: {validation_loss:.4f} "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # CLEARML METRICS
    # --------------------------------------------------------

    logger.report_scalar(
        title="Loss",
        series="Train",
        value=train_loss,
        iteration=epoch,
    )

    logger.report_scalar(
        title="Loss",
        series="Validation",
        value=validation_loss,
        iteration=epoch,
    )

    logger.report_scalar(
        title="Accuracy",
        series="Train",
        value=train_accuracy,
        iteration=epoch,
    )

    logger.report_scalar(
        title="Accuracy",
        series="Validation",
        value=validation_accuracy,
        iteration=epoch,
    )


# ============================================================
# 9. EXPORT TORCHSCRIPT
# ============================================================

model.eval()

model_directory = Path("models")

model_directory.mkdir(
    parents=True,
    exist_ok=True,
)

torchscript_path = (
    model_directory / "resnet18_cifar10.pt"
)

example_input = torch.randn(
    1,
    3,
    params["image_size"],
    params["image_size"],
    device=device,
)

# Trace the trained model.

traced_model = torch.jit.trace(
    model,
    example_input,
)

# Save the TorchScript model.

torch.jit.save(
    traced_model,
    str(torchscript_path),
)

print("=" * 60)
print("TORCHSCRIPT EXPORT COMPLETED")
print("=" * 60)

print(f"Model: {torchscript_path}")


# ============================================================
# 10. VERIFY TORCHSCRIPT
# ============================================================

loaded_model = torch.jit.load(
    str(torchscript_path),
    map_location=device,
)

loaded_model.eval()

with torch.no_grad():

    test_input = torch.randn(
        1,
        3,
        params["image_size"],
        params["image_size"],
        device=device,
    )

    prediction = loaded_model(test_input)

assert prediction.shape == (
    1,
    params["num_classes"],
)

print("TorchScript verification successful!")

print(f"Output shape: {prediction.shape}")


# ============================================================
# 11. UPLOAD MODEL TO CLEARML
# ============================================================

output_model.update_weights(
    weights_filename=str(torchscript_path),
)

print("=" * 60)
print("MODEL UPLOAD COMPLETED")
print("=" * 60)

print("Model registered in ClearML Model Registry.")


# ============================================================
# 12. FINALIZE TASK
# ============================================================

task.close()

print("Training task completed successfully.")