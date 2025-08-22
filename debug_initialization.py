import torch
import torch.nn as nn
from src.models import DeepCNN_CIFAR10
from src.datasets import get_dataloaders
from src.utils import get_device, set_seed

def analyze_initialization():
    """モデルの初期化状態を分析"""
    set_seed(42)  # 同じ初期化を再現
    device = get_device()
    
    # モデルを構築
    model = DeepCNN_CIFAR10(num_classes=10).to(device)
    
    print("=== DeepCNN Initialization Analysis ===")
    
    # 各層の重みの統計を確認
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\n{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {param.mean().item():.6f}")
            print(f"  Std: {param.std().item():.6f}")
            print(f"  Min: {param.min().item():.6f}")
            print(f"  Max: {param.max().item():.6f}")
    
    # データローダーを取得
    _, test_loader, _ = get_dataloaders("cifar10", batch_size=128, num_workers=2)
    
    model.eval()
    
    # 初期化直後の予測を確認
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 1:  # 最初のバッチのみ
                break
                
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            print(f"\n=== Initial Prediction Analysis ===")
            print(f"Output shape: {outputs.shape}")
            
            # 生のlogits
            print(f"\nRaw logits (first 5 samples):")
            print(outputs[:5])
            
            # softmax出力
            probs = torch.softmax(outputs, dim=1)
            print(f"\nSoftmax probabilities (first 5 samples):")
            print(probs[:5])
            
            # 予測クラス
            preds = outputs.argmax(dim=1)
            print(f"\nPredicted classes (first 20): {preds[:20].cpu().numpy()}")
            
            # 予測クラスの分布
            unique, counts = torch.unique(preds, return_counts=True)
            print(f"\nPrediction distribution:")
            for cls, count in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                print(f"  Class {cls}: {count} times ({count/len(preds)*100:.1f}%)")
            
            # 全サンプルの平均確率分布
            mean_probs = probs.mean(dim=0)
            print(f"\nAverage probability distribution across all classes:")
            for i, prob in enumerate(mean_probs.cpu().numpy()):
                print(f"  Class {i}: {prob:.4f}")
            
            break

if __name__ == "__main__":
    analyze_initialization()
