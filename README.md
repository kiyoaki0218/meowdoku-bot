# MeowDoku LINE Bot (Vercel Serverless)

スマートフォンから送信されたパズルゲーム（みゃおドク等）のスクリーンショットをAI・アルゴリズムで解析し、解答位置を自動で返信するLINE Bot（Vercel対応版）です。

## 構成
- `api/index.py`: LINE Webhookサーバーおよび画像解析・解法ロジック
- `requirements.txt`: 依存ライブラリ一覧
- `vercel.json`: Vercelルーティング設定

## Vercel環境変数の設定
Vercelのプロジェクト設定 (Settings > Environment Variables) で以下を設定してください。
- `LINE_CHANNEL_SECRET`: LINE Developersで取得したチャネルシークレット
- `LINE_CHANNEL_ACCESS_TOKEN`: LINE Developersで取得したチャネルアクセストークン
