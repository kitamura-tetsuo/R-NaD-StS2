# R-NaD-StS2

## 概要
**R-NaD-StS2** は、**Slay the Spire 2** に Regularized Nash Dynamics (R-NaD) と呼ばれる強化学習アルゴリズムを統合するための Godot Mod です。StS2 のゲームプロセス内で完結する言語間の堅牢な連携ブリッジにより、Pythonベースの強力な強化学習エコシステムをゲーム上で直接動作させることができます。

## アーキテクチャ
このプロジェクトは、ゲーム内部で動作する3つのモジュールと、外部のコントロールパネルの計4つのコンポーネントで構成されています。

1. **`communication-mod` (C#)**: Slay the Spire 2 の Mod エントリポイントです。Harmony パッチを使用してゲームの各状態（State）をインターセプトし、シリアライズして AI ブリッジへ渡します。また AI から返されるアクションをゲーム本体に適用します。
2. **`GDExtension` (Rust + PyO3)**: C# Mod と Python の間を取り持つ Godot ノード (`AiBridge`) です。ゲームプロセス内に Python インタプリタを埋め込み、ゲームステートの非同期処理を行います。
3. **`R-NaD` (Python)**: 推論と学習のコアモジュール (`rnad_bridge.py`) です。ゲームから送られる `state_json` を受け取り、アクションを返却します。同時に裏側でローカルの HTTP サーバー（ポート 8081）を起動し、外部からのコマンドを待ち受けます。
4. **`Streamlit UI` (Python)**: 独立した別プロセスから動作するウェブベースのコントロールパネル (`ui.py`) です。ゲームが実行されている状態のまま、ワンクリックで学習のオン・オフを動的に切り替えるために使用します。

## 動作要件・前提条件
- **Godot 4.x** (Slay the Spire 2 が使用するバージョンに準拠します)
- **Rust ツールチェーン** (GDExtension のコンパイル用)
- **.NET SDK** (C# Mod のコンパイル用)
- **Python 3.10 以上** (システムのPythonと互換性があること)

## ビルド・実行手順

# 仮想環境の再作成
rm -rf ./R-NaD/venv
python3 -m venv ./R-NaD/venv

# 構成のインストール
./R-NaD/venv/bin/pip install -r ./R-NaD/requirements_sts2_gpu.txt


### 1. Rust GDExtension のビルド
```bash
cd GDExtension
cargo build
```

### 2. C# Mod のビルド
```bash
cd communication-mod
dotnet build
```

### 3. Streamlit UI 起動の準備
Streamlit UI を稼働させるため、Python環境と依存パッケージをインストールします：
```bash
cd R-NaD
python3 -m venv venv
source venv/bin/activate
pip install streamlit requests jax jaxlib dm-haiku optax mlflow numpy
streamlit run ui.py
```
*ブラウザから `http://localhost:8501` へアクセスし、UIを開いておきます。*

### 4. R-NaD の学習実行 (Optional)
ゲームを起動する前に、あるいはゲームと並行して R-NaD の学習プロセスを個別に実行・検証することができます：
```bash
# JAXベースの学習ループを起動
export PYTHONPATH=$PYTHONPATH:$(pwd)/R-NaD
python3 R-NaD/train_sts2.py --max_steps 1000
```
*学習済みのチェックポイントは `checkpoints/` ディレクトリに保存されます。*

### 5. Slay the Spire 2 の起動
StS2をModローダー経由で起動します。`communication-mod` が `AiBridge` ノードを展開し、Pythonのデーモンがバックグラウンドで起動して通信を開始します。先ほど起動したStreamlit UIから、「Learning ACTIVE/INACTIVE」を切り替えることで推論および学習をコントロールします。


## To visualize trajectories:
/home/ubuntu/src/R-NaD-StS2/R-NaD/venv/bin/streamlit run visualize_trajectories.py --server.port 8501


## ライセンス
MIT License (詳細は LICENSE ファイルをご確認ください)。
