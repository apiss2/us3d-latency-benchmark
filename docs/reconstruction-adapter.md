# Reconstruction command adapter

利用者が実装したコマンドを、次の呼び出し規約に接続してください。Python以外の実装でも構いません。

```text
<your command> --request <request.json> --output <reconstruction.pending.npz>
```

`run` はシェル文字列を評価せず、引数配列としてコマンドを起動します。成功時はexit code 0で、指定した出力ファイルを作成します。失敗時は非0で終了します。

## Request

[`examples/reconstruction-request.json`](../examples/reconstruction-request.json) は手書きの仕様例です。実行時の `request.json` には以下が含まれます。

- `case_id`、経路・速度、遅延frame/ms、seed、画像枚数、欠損数・率。
- 元の画像・pose・時刻配列への `inputs` と、それぞれのSHA256。
- 画像shape、spacing、原点、単位mm、要求する出力格子spacing。
- `condition_file`: 構成済みのposeと時刻対応を持つNPZ。実行時の参照パスは絶対パスです。

`condition.npz` の配列:

| key | 内容 |
|---|---|
| `poses [N,4,4]` | 画像に対応づけたpose。欠損位置は全0 |
| `pose_valid [N]` | poseの有効フラグ。欠損の判定はこちらを使用 |
| `image_frame_index [N]` | 元画像index。常に0..N-1 |
| `tracking_frame_index [N]` | 対応する元の追跡index。`image_frame_index+k` |
| `image_times_s [N]` | 画像取得時刻を保持 |
| `pose_times_s [N]` | 割り当てたposeの取得時刻 |

画像 `I[i]` とpose `T[i+k]` を組にするため、超音波側が遅い方向です。同値な受信時刻表現は `I[n-k]` と `T[n]` です。時刻対応を再補正して遅延を相殺しないでください。

欠損マスクは追跡frame番号で定義し、同じ走査・seedの全遅延に共有します。全条件で前後poseが存在する共通内部領域 `[max_delay+1, N-2]` から5個を無作為抽出します。NumPy PCG64と、非復元抽出後の昇順整列を使用します。

## Missing-pose policy

再構成アダプターで方針を明示してください。例えば、前後の有効poseを使うオフライン補間、因果的な保持、該当画像の処理を保留する方法などです。同一runで方針を固定します。元の画像・時刻・frame番号の対応は維持してください。

前後pose補間を採用するなら、未来のposeを使うことを記録します。固定姿勢では並進を時刻で線形補間できますが、その実装と再構成への組み込みは利用者が担当します。

## Output

`np.load(path, allow_pickle=False)` で読めるNPZに次のkeyを返します。

- `volume`: **uint8 `[Z,Y,X]`**、強度0..255。判定閾値は全条件共通で128。確率や他の強度尺度を使う場合は、全条件で同じ規則でこの尺度へ変換します。
- `index_to_world_mm`: float `[4,4]`。`world_mm = matrix @ [x_index,y_index,z_index,1]`。indexはvoxel中心を表します。3×3部分の各列は、回転とその軸のspacingを含むmm単位の基底ベクトルです。右手系・直交格子としてください。
- `metadata_json`: JSON文字列を格納した0次元Unicode配列。必須keyは `case_id`、`method`、`missing_pose_policy` で、いずれも空でない文字列です。手法の版、融合・穴埋め、厚み、平滑化、資源設定などは追加keyで記録できます。

手書きの出力仕様は [`examples/reconstruction-output.json`](../examples/reconstruction-output.json) を参照してください。これはNPZの代わりに返すJSONではありません。

`run` は成功した一時出力を検証してから `reconstruction.npz` に確定します。case IDと元入力hashが一致する既存出力は再利用します。再構成アルゴリズムや欠損方針を変更する際は新しいrunを作ってください。
