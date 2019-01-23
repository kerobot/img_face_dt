# img_face_dt

Face Detect by OpenCV

OpenCVと既存の分類器を利用して画像ファイルから顔を抽出します。

事前に OpenCV の haarcascade_frontalface_default.xml を用意しておくこと。

## 環境

* Windows 10 x64 1809
* Python 3.6.5 x64
* Power Shell 6 x64
* Visual Studio Code x64
* Git for Windows x64
* OpenCV 3.4.4

## 構築

プロジェクトを clone してディレクトリに移動します。

```powershell
> git clone https://github.com/kerobot/img_face_dt.git img_face_dt
> cd img_face_dt
```

プロジェクトのための仮想環境を作成して有効化します。

```powershell
> python -m venv venv
> .\venv\Scripts\activate.ps1
```

念のため、仮想環境の pip をアップグレードします。

```powershell
> python -m pip install --upgrade pip
```

依存するパッケージをインストールします。

```powershell
> pip install -r requirements.txt
```

環境変数を設定します。

> CASCADE_FILE_PATHを設定

```powershell
> copy .\.env.sample .\.env
> code .\.env
```

## 実行

origin_imageディレクトリに画像ファイルを配置して実行します。

> 画像ファイルごとに認識した顔画像を出力

```powershell
> python .\img_face_dt.py
```
