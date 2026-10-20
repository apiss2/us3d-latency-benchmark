# US3D latency benchmark

超音波断面と位置情報の時刻ずれが、再構成の位置・形状に与える影響を比較する実験用ハーネスです。

利用者が **入力データ生成アダプター** と **3D再構成アダプター** を実装すると、同じ条件の構成・評価・集計を実行できます。アダプターなしで実画像の実験を自動完了するものではありません。再現対象は実験条件と評価方法であり、任意の再構成手法が同じ数値を返すとは限りません。

## 利用者が実装するもの

1. **入力アダプター**: [`scene/phantom.blend`](scene/phantom.blend) の走査アニメーションから実際の切断面画像と対応するpose・時刻列を取得し、[入力契約](docs/input-adapter.md)のNPY配列とmanifestを書き出します。
2. **再構成アダプター**: [コマンド契約](docs/reconstruction-adapter.md)に従って画像と条件付きposeを受け取り、世界座標に対応した3D強度配列を返します。アルゴリズムや実装言語は任意です。

入力アダプターを先に作り、原点・単位・方向・時刻を少数ケースで確認してください。3D再構成の設定と欠損処理方針を固定して比較します。

## セットアップ

Python 3.10以上、Blender 4.5 LTSを想定しています。Blenderはシーンを読む入力アダプターで使用します。評価コード自体はBlenderのPython APIに依存しません。

```sh
python -m venv .venv
# 作成した仮想環境を有効化
python -m pip install -e .
python -m unittest discover -s tests -v
```

テストは手書きの小さなプロトコルfixtureを使います。実験データの生成や実際の再構成の正しさを代わりに検証するものではありません。

## 実験条件

| 項目 | 既定値 |
|---|---|
| ファントム | 円柱、半径4 mm、長さ80 mm、中心深さ20 mm |
| 経路 | 直線・緩いカーブ・蛇行。姿勢固定、横方向の平行移動、一方向走査 |
| 速度 | 経路の弧長速度5 / 10 / 15 mm/s |
| 取得 | 30 FPS、400×400、0.1 mm/pixel、原点 `(W/2,0)=(200,0)` |
| 遅延 | 0〜6 frame、0〜200 ms、超音波側が遅い向き |
| 位置欠損 | 各走査で合計5 pose、seed 1001〜1008 |
| 再構成・評価格子 | 0.2 mm等方、閾値128、全条件で固定の世界ROI |

設定は [`benchmark.json`](benchmark.json)、評価式と限界は[評価仕様](docs/evaluation.md)を参照してください。シーンの説明は [`scene/README.md`](scene/README.md) にあります。

## 実行手順

入力アダプターで `data/dataset.json` と配列を用意します。manifestは[手書きの仕様サンプル](examples/dataset-manifest.json)を参考にしてください。

```sh
python -m us3d_latency validate-input data/dataset.json
python -m us3d_latency prepare data/dataset.json --output runs/experiment-a
python -m us3d_latency run runs/experiment-a -- python my_reconstruction_adapter.py
python -m us3d_latency evaluate runs/experiment-a
python -m us3d_latency summarize runs/experiment-a --output summary/experiment-a
```

`my_reconstruction_adapter.py` は利用者が用意するファイルです。`run` はそのコマンドの末尾に `--request <JSON> --output <NPZ>` を付けて各条件を順番に実行します。アダプター側でもCPU/GPU・スレッド数を明示して資源使用量を調整してください。

全9走査を入力した既定設定では、欠損なし63条件と欠損8 seedの504条件、合計567条件を構成します。1走査だけのmanifestでも接続確認できますが、集計には全体が未完了である旨を記録します。`prepare` と `summarize` には新しい出力先を指定してください。

`run` は入力hashと出力契約・case IDを検査し、既存の有効な出力を再利用します。設定を変える場合は別runを作ってください。異常終了後の `.adapter.lock` は、対応プロセスが終了したことを確認してから利用者が削除してください。

## 出力と解釈

- `all_conditions.csv`: 生IoU、重心平行移動補正IoU、XY中心の平均偏り・RMSE・平均を除いた変動・最大偏差、検出率、Z重心移動、対照との差。
- `missing_seed_summary.csv`: 同遅延の欠損なし対照に対するIoU差の平均とmin/max。帯域を信頼区間とは解釈しません。
- `iou_delay.png` / `corrected_iou_delay.png` / `center_rmse.png` / `centerlines.png`: 経路・速度・遅延ごとの比較。
- 各caseの `request.json`、`condition.npz`、`reconstruction.npz`、`metrics.json`、`centerline.csv`。

生IoUを主指標にし、平行移動補正IoUは移動と変形を区別する補助指標として扱います。中心RMSEは**世界XYのみ**で、Z方向へのずれは重心とIoU・検出率で別に評価します。

位置欠損時に未来poseを使って補間する場合、その結果はオフライン補間の評価です。実時間処理や、長い連続欠損・実機ノイズへの性能を意味しません。シーンは遅延を分離する幾何学的ファントムで、音響シミュレーションではありません。

0.1 mm格子との感度比較を行う場合は、設定の `evaluation.voxel_mm` を変更した**別run**を作り、再構成アダプターも同じ要求格子へ合わせてください。入力画像・世界ROI・閾値など他の定義は固定します。2段階の格子比較だけで厳密な収束を証明したとはみなしません。

ライセンスは未指定です。
