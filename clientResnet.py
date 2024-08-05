import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from io import BytesIO
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
classes = [str(i) for i in range(10)]

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

class ResNetClient(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetClient, self).__init__()
        resnet = resnet18()
        self.firstLayers = nn.Sequential(*list(resnet.children())[:6])  # conv1, bn1, relu, maxpool, layer1, layer2        
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(100352, num_classes) # 100352 is the number of features after layer2


    def forward(self, x):
        x = self.firstLayers(x)
        ee = self.flatten(x)
        ee = self.classifier(ee)
        return ee, x


def offload(tensor):
    buffer = BytesIO()
    torch.save(tensor.cpu(), buffer)
    buffer.seek(0)

    try:
        response = requests.post("http://10.110.2.1:8000/process", data=buffer.getvalue(), headers={"Content-Type": "application/octet-stream"})
        response.raise_for_status()

        if response.status_code == 503:
            print("Server is busy processing another request.")
            return None

        result_buffer = BytesIO(response.content)
        return torch.load(result_buffer, map_location=device)
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None



def process_image(image_path, threshold=0.9):
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform_test(image).unsqueeze(0).to(device)


    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Load the model
    model = ResNetClient()
    model.load_state_dict(torch.load("resnet.pth"), strict=False)
    model.to(device)
    model.eval()

    isOffload = False

    # Process the image
    with torch.inference_mode():
        output, ee_layer = model(image_tensor)
        softmax_output = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(softmax_output, dim=1)

        if confidence.item() > threshold:
            result = predicted_class.item()
        else:
            offloaded_result = offload(ee_layer)
            if offloaded_result is not None:
                result = offloaded_result.item()
                isOffload = True
            else:
                result = predicted_class.item()

    # Map the result to a class name (you may want to update this based on your specific classes)
    class_name = classes[result]

    offloadText = " (offloaded)" if isOffload else " (processed locally)"

    return f"Predicted class: {class_name}, Confidence: {confidence.item() * 100:.2f}%  {offloadText}"

if __name__ == "__main__":
    # This block is for testing the script independently
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = process_image(image_path)
        print(result)
    else:
        print("Please provide an image path as an argument.")