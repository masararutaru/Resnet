import torch
import torch.nn.functional as F
from src.models import DeepCNN_CIFAR10
from src.datasets import get_dataloaders
from src.utils import get_device

def analyze_deepcnn_predictions():
    """DeepCNNの予測分布を分析"""
    device = get_device()
    
    # データローダーを取得
    _, test_loader, _ = get_dataloaders("cifar10", batch_size=128, num_workers=2)
    
    # モデルを構築
    model = DeepCNN_CIFAR10(num_classes=10).to(device)
    
    # 保存されたモデルを読み込み
    checkpoint = torch.load("ckpts/cifar10-deepcnn_best.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['acc']*100:.2f}%")
    
    model.eval()
    
    # 最初の100サンプルで予測を確認
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 1:  # 最初のバッチのみ
                break
                
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            # 生のlogits
            print("Raw logits (first 5 samples):")
            print(outputs[:5])
            
            # softmax出力
            probs = F.softmax(outputs, dim=1)
            print("\nSoftmax probabilities (first 5 samples):")
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
    analyze_deepcnn_predictions()
