import os
import io
import uuid
import logging
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, abort, send_file
from sklearn.cluster import KMeans

# LINE SDK v3 Imports
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    ImageMessage
)
from linebot.v3.webhooks import MessageEvent, ImageMessageContent

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# LINE設定 (Vercel環境変数から取得)
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ==========================================
# 1. パズル解法ロジック (バックトラッキング)
# ==========================================
def solve_meowdoku(grid):
    N = len(grid)
    colors = set(cell for row in grid for cell in row)
    if len(colors) != N:
        return None

    solution = [[0] * N for _ in range(N)]
    used_cols = [False] * N
    used_colors = [False] * N
    cats_pos = []

    def is_safe(r, c, color_id):
        if used_cols[c] or used_colors[color_id]:
            return False
        for pr, pc in cats_pos:
            if abs(pr - r) <= 1 and abs(pc - c) <= 1:
                return False
        return True

    def backtrack(r):
        if r == N:
            return True
        for c in range(N):
            color_id = grid[r][c]
            if is_safe(r, c, color_id):
                solution[r][c] = 1
                used_cols[c] = True
                used_colors[color_id] = True
                cats_pos.append((r, c))

                if backtrack(r + 1):
                    return True

                solution[r][c] = 0
                used_cols[c] = False
                used_colors[color_id] = False
                cats_pos.pop()
        return False

    if backtrack(0):
        return solution
    return None


# Imgur への無料画像アップロード関数
def upload_to_imgur(image_bytes):
    try:
        url = "https://api.imgur.com/3/image"
        headers = {"Authorization": "Client-ID 5442646d79e5d9c"}
        payload = {'image': base64.b64encode(image_bytes).decode('utf-8')}
        res = requests.post(url, headers=headers, data=payload, timeout=10)
        data = res.json()
        if data.get('success'):
            return data['data']['link']
    except Exception as e:
        logging.error(f"Imgur upload failed: {e}")
    return None


# ==========================================
# 2. 精密な盤面枠自動検出 (カラフルピクセル分析)
# ==========================================
def find_board_bounds(img_np):
    """
    パズル盤面のカラフルな領域の正確なバウンディングボックス (bx, by, bw, bh) を検出
    """
    h, w, _ = img_np.shape
    
    # 彩度判定 (RGBの最大値と最小値の差が大きいピクセル＝パズルの色マス)
    r = img_np[..., 0].astype(int)
    g = img_np[..., 1].astype(int)
    b = img_np[..., 2].astype(int)
    
    color_diff = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(b - r))
    
    # 明らかな色がついているピクセル (彩度 > 25)
    colored_mask = color_diff > 25

    # 画像の上部・下部のUI領域（アイコン等）を除外するため、Y範囲を20%〜80%に限定
    y_min_limit = int(h * 0.20)
    y_max_limit = int(h * 0.82)
    colored_mask[:y_min_limit, :] = False
    colored_mask[y_max_limit:, :] = False

    y_indices, x_indices = np.where(colored_mask)

    if len(y_indices) > 0 and len(x_indices) > 0:
        bx = int(np.min(x_indices))
        by = int(np.min(y_indices))
        bw = int(np.max(x_indices) - bx)
        bh = int(np.max(y_indices) - by)
        
        # 正方形に近づける微調整
        side = max(bw, bh)
        return bx, by, side, side

    # 万が一検出失敗した場合のバックアップ推定
    bw = int(w * 0.9)
    bh = bw
    bx = int((w - bw) / 2)
    by = int(h * 0.31) # 盤面の平均的な開始位置
    return bx, by, bw, bh


# ==========================================
# 3. 画像解析 (PIL + NumPy)
# ==========================================
def process_puzzle_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    # 1. 精密な盤面枠の検出
    bx, by, bw, bh = find_board_bounds(img_np)

    # 2. 9x9 または 10x10 の判定と解法
    for N in [9, 10]:
        cell_w = bw / N
        cell_h = bh / N
        colors_samples = []

        for r in range(N):
            row_samples = []
            for c in range(N):
                cx = int(bx + (c + 0.5) * cell_w)
                cy = int(by + (r + 0.5) * cell_h)
                patch = img_np[max(0, cy-3):cy+4, max(0, cx-3):cx+4]
                mean_rgb = patch.mean(axis=(0, 1))
                row_samples.append(mean_rgb)
            colors_samples.append(row_samples)

        flat_samples = np.array(colors_samples).reshape(-1, 3)
        kmeans = KMeans(n_clusters=N, random_state=42, n_init=10).fit(flat_samples)
        labels = kmeans.labels_.reshape(N, N)

        solution = solve_meowdoku(labels)
        if solution is not None:
            # 解が見つかったら描画
            draw = ImageDraw.Draw(img)
            cat_coords = []
            for r in range(N):
                for c in range(N):
                    if solution[r][c] == 1:
                        cat_coords.append((r + 1, c + 1))
                        cx = bx + (c + 0.5) * cell_w
                        cy = by + (r + 0.5) * cell_h
                        rad = min(cell_w, cell_h) * 0.38
                        
                        # 太い赤色の二重丸を中心ピッタリに描画
                        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline="red", width=7)
                        draw.ellipse([cx - rad*0.35, cy - rad*0.35, cx + rad*0.35, cy + rad*0.35], fill="red")

            # 画像をバイト列に変換
            out_buffer = io.BytesIO()
            img.save(out_buffer, format="JPEG", quality=85)
            out_bytes = out_buffer.getvalue()

            # Imgur にアップロードして公開URLを取得
            public_image_url = upload_to_imgur(out_bytes)

            return solution, cat_coords, public_image_url

    raise ValueError("解答パターンが見つかりませんでした。")


# ==========================================
# 4. Webhook エンドポイント
# ==========================================
@app.route("/", methods=['GET'])
def index():
    return "MeowDoku LINE Bot is Running on Vercel!"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


# ==========================================
# 5. LINE メッセージ処理
# ==========================================
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    message_id = event.message.id

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_blob_api = MessagingApiBlob(api_client)

        try:
            image_bytes = line_bot_blob_api.get_message_content(message_id=message_id)
            solution, cat_coords, image_url = process_puzzle_image(image_bytes)

            messages = []

            # 1. テキスト形式のネコの位置情報
            coord_text = "🐱 ネコの位置（上から何行目 - 左から何列目）：\n"
            coord_text += "\n".join([f"・{r}行目 - {c}列目" for r, c in cat_coords])
            messages.append(TextMessage(text=coord_text))

            # 2. 画像メッセージ (公開URL)
            if image_url:
                messages.append(ImageMessage(
                    original_content_url=image_url,
                    preview_image_url=image_url
                ))

            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=messages
                )
            )
        except Exception as e:
            logging.error(f"エラー: {str(e)}")
            error_msg = TextMessage(text="画像の読み取りまたは解答の検索に失敗しました 😿\nもう一度明るくハッキリ写ったスクリーンショットを送信してください。")
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[error_msg]
                )
            )
