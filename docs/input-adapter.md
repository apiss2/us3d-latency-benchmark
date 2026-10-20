# Input acquisition adapter

利用者のアダプターでBlenderシーンを評価し、各走査を次の4配列に変換してください。シーンを読み出すAPIや、断面画像の取得・ラスタ化方法の実装は利用者が選びます。

| 配列 | dtype / shape | 意味 |
|---|---|---|
| `images.npy` | `uint8 [N,H,W]` | 全画像。既定は400×400、輪郭内255・外0 |
| `image_times_s.npy` | float `[N]` | 取得時刻。30 FPS、時刻原点はpose列と同じ |
| `poses.npy` | float `[M,4,4]` | ローカル画像物理座標から右手系世界座標への剛体変換。並進mm |
| `pose_times_s.npy` | float `[M]` | 全有効poseの取得時刻。M ≥ N+6 |

NPYはpickleを使わない通常の数値配列です。ファイル名は自由で、manifestから相対パスで指定します。親ディレクトリへの脱出や絶対パスは入力manifestでは受け付けません。

manifestの書式は [`examples/dataset-manifest.json`](../examples/dataset-manifest.json) を参照してください。実行用manifestは書き出したNPYファイルと同じデータディレクトリに置きます。全9走査の `path` と `speed_mm_s` の組は一意にしてください。

## 座標と画素中心

画像画素index `(ix,iy)` のローカル座標は、mm単位で

`[(ix-origin_x)*spacing_x, (iy-origin_y)*spacing_y, 0, 1]`

です。既定はorigin=`(200,0)`、spacing=`(0.1,0.1)`。画素index自体が中心なので、追加の半画素オフセットは付けません。`world = pose @ local` の列ベクトル規約です。世界Xが横、Yが深さ、Zが円柱軸。固定姿勢の単位回転を用います。

入力画像の定義と、評価時のROI cell中心は異なります。[評価仕様](evaluation.md)の格子規約を確認してください。

## 時刻・余白・欠損

画像・poseとも共通時刻原点から `index/30` 秒で取得します。画像列はN枚すべてを維持し、末尾に最低6枚分の未来poseを持たせます。前後の背景画像を省かず、[scene contract](../scene/README.md)の取得範囲を維持してください。

ここで入力するpose列はすべて有効です。遅延・欠損の条件は `prepare` が構成します。フレーム欠損のために入力画像・時刻配列を詰めたり、画像の順番を付け替えたりしないでください。

入力検証は配列形状・dtype・時刻・剛体行列・固定姿勢・Zの単調増加を検査します。断面画像の取得精度、シーンの取り扱い、画素中心の実装、意図した軌道と速度の忠実性は、利用者が別に検証する必要があります。
