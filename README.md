# PyTorch Basics (MNIST/CIFAR-10) with ResNet Comparison

手で学習ループを書き、MNIST(MLP)/CIFAR-10(CNN)を学習する最小構成。ResNetの効果を実証するための比較実験を含む。

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windowsは .venv\Scripts\activate
pip install -r requirements.txt
```

> GPU版PyTorchは環境に合わせて公式手順でインストールしてください。

## Run

```bash
# MNIST (MLP)
python -m src.train --cfg configs/default.yaml

# CIFAR-10 (Simple CNN)
python -m src.train --cfg configs/default.yaml

# CIFAR-10 (Deep CNN - ResNetなし)
# configs/default.yaml の task を `cifar10-deepcnn` に変更してから実行

# CIFAR-10 (ResNet - スキップ接続あり)
# configs/default.yaml の task を `cifar10-resnet` に変更してから実行
```

## Structure

* `src/datasets.py`: torchvisionでMNIST/CIFAR-10を取得しDataLoader化
* `src/models.py`: 
  - MLP_MNIST: MNIST用の全結合ネットワーク
  - SimpleCNN_CIFAR10: シンプル3層CNN（比較用）
  - DeepCNN_CIFAR10: 深いCNN（ResNetなし、勾配消失問題を再現）
  - ResNet_CIFAR10: ResNet（スキップ接続で勾配消失問題を解決）
* `src/train.py`: 学習・評価ループ、StepLR対応
* `configs/default.yaml`: 学習設定

## ResNet効果の実証

このプロジェクトでは、ResNet論文の効果を実証するための比較実験が可能です：

1. **SimpleCNN_CIFAR10**: 浅いネットワーク（3層）
2. **DeepCNN_CIFAR10**: 深いネットワーク（10層、ResNetなし）
3. **ResNet_CIFAR10**: 深いネットワーク（10層、スキップ接続あり）

期待される結果：
- DeepCNN: 深い層で学習が停滞（勾配消失問題）
- ResNet: スキップ接続により安定した学習

## Notes

* データは `./data/` に自動ダウンロード
* チェックポイントは `./ckpts/` に保存
