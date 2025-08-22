import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from .models import MLP_MNIST, SimpleCNN_CIFAR10, DeepCNN_CIFAR10, ResNet_CIFAR10
from .utils import get_device


def load_and_preprocess_image(image_path: str, task: str):
    """画像を読み込んで前処理する"""
    # 画像を読み込み
    image = Image.open(image_path).convert('RGB')
    
    if task == "mnist-mlp":
        # MNIST用の前処理
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # グレースケールに変換
            transforms.Resize((28, 28)),  # 28x28にリサイズ
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNISTと同じ正規化
        ])
    elif task in ["cifar10-cnn", "cifar10-deepcnn", "cifar10-resnet"]:
        # CIFAR10用の前処理
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 32x32にリサイズ
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR10と同じ正規化
        ])
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # 前処理を適用
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # バッチ次元を追加


def predict_image(model, image_tensor, task: str):
    """画像の予測を行う"""
    device = get_device()
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0]


def get_class_names(task: str):
    """クラス名を取得"""
    if task == "mnist-mlp":
        return [str(i) for i in range(10)]  # 0-9の数字
    elif task in ["cifar10-cnn", "cifar10-deepcnn", "cifar10-resnet"]:
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        raise ValueError(f"Unknown task: {task}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="画像推論")
    parser.add_argument("--image_path", required=True, help="推論したい画像のパス")
    parser.add_argument("--model_path", required=True, help="学習済みモデルのパス")
    parser.add_argument("--task", choices=["mnist-mlp", "cifar10-cnn", "cifar10-deepcnn", "cifar10-resnet"], required=True, help="タスク名")
    args = parser.parse_args()
    
    # モデルを読み込み
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # モデルを構築
    if args.task == "mnist-mlp":
        model = MLP_MNIST(num_classes=10)
    elif args.task == "cifar10-cnn":
        model = SimpleCNN_CIFAR10(num_classes=10)
    elif args.task == "cifar10-deepcnn":
        model = DeepCNN_CIFAR10(num_classes=10)
    elif args.task == "cifar10-resnet":
        model = ResNet_CIFAR10(num_classes=10)
    
    model.load_state_dict(checkpoint['model'])
    
    # 画像を読み込んで前処理
    image_tensor = load_and_preprocess_image(args.image_path, args.task)
    
    # 予測実行
    predicted_class, confidence, probabilities = predict_image(model, image_tensor, args.task)
    
    # クラス名を取得
    class_names = get_class_names(args.task)
    
    # 結果を表示
    print(f"予測結果: {class_names[predicted_class]} (クラス {predicted_class})")
    print(f"信頼度: {confidence:.4f}")
    print("\n全クラスの確率:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {class_name}: {prob:.4f}")


if __name__ == "__main__":
    main() 