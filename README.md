# 規定類 検索システム（沖縄支社パイロット版）

損保8社の規定類（約款・ハンドブック・マニュアル）を、会社・種目で絞って
AI検索できる社内ツールです。Google Driveの最新PDFに常に追従し、
回答には必ず「ファイル名＋ページ」の根拠を付けます。

## 特徴
- **会社・種目フィルタで他社データを物理的に除外**（回答混同のリスクをゼロに）
- **根拠（ファイル名・ページ）を明示**
- **ハルシネーション抑制**：規定に無いことは「記載がありません」と回答
- **管理はDriveに上書きするだけ**：全員が最新データで検索

## フォルダ構成
```
規定類検索/
├── app.py                     … Streamlitメインアプリ（UI・認証・検索）
├── rag/
│   ├── config.py              … 会社名・種目・モデル等の設定（主にここを編集）
│   ├── sources.py             … Drive同期／ローカル走査＋変更検知
│   ├── indexer.py             … PDF→ページ単位ベクトル化・保存・読込
│   └── query_engine.py        … フィルタ検索＋Gemini生成＋出典抽出
├── data/                      … PDFキャッシュ・インデックス（自動生成／Git対象外）
├── .streamlit/
│   ├── config.toml            … テーマ設定
│   └── secrets.toml.example   … APIキー等の設定テンプレ（→ secrets.toml にコピー）
├── requirements.txt
├── README.md                  … このファイル
└── docs/管理者マニュアル.md    … 非エンジニア向けの導入・運用手順（★まずこれ）
```

## クイックスタート（ローカルで試す）
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 設定ファイルを用意
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
#  → secrets.toml を開いてAPIキー・パスワード等を記入

# まず動作確認だけしたい場合は rag/config.py の SOURCE_MODE を "local" にして
# data/pdfs/<会社>/<種目>/xxx.pdf にPDFを置く（Drive設定不要）

streamlit run app.py
```

## 導入・デプロイ手順
非エンジニアの管理者向けに **[docs/管理者マニュアル.md](docs/管理者マニュアル.md)** に
すべて図解なしの手順で書いています。まずはそちらを参照してください。

## 運用フロー（管理者）
1. Google Driveの `<ルート>/<会社>/<種目>/` に最新PDFをアップロード（上書き）
2. アプリの「🔄 データ更新」を1回押す
3. 以降、全メンバーが最新データで検索できる

## 既知の制約
- **スキャン（画像）PDFは非対応**：文字が埋め込まれたPDFが必要（OCRは将来対応）
- **Streamlit Community Cloudはディスクが揮発性**：再起動時にインデックス再構築（1〜2分）が走る。堅牢性重視ならCloud Run等の常時稼働ホスト推奨
- **認証は簡易共通パスワード**：50人パイロット向け。1000人展開時はSSO等へ移行推奨
