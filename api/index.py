import os
import io
import uuid
import logging
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, abort, send_file, render_template_string
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
        res = requests.post(url, headers=headers, data=payload, timeout=5)
        data = res.json()
        if data.get('success'):
            return data['data']['link']
    except Exception as e:
        logging.error(f"Imgur upload failed: {e}")
    return None


# ==========================================
# 2. オンデマンド画像生成 (LINE画像配信用)
# ==========================================
@app.route("/render_solution")
def render_solution():
    """
    外部画像アップロードが失敗した場合でも、確実にLINEへ画像を返すためのオンデマンド描画ルート
    """
    try:
        cats_param = request.args.get("cats", "")
        N = int(request.args.get("N", 9))
        
        coords = [tuple(map(int, p.split("_"))) for p in cats_param.split(",") if p]
        
        # 600x600 のシンプルな解答ボード画像を動的作成
        img_size = 600
        img = Image.new("RGB", (img_size, img_size), "#F8F4F0")
        draw = ImageDraw.Draw(img)
        
        cell_size = img_size / N
        
        # 枠線の描画
        for i in range(N + 1):
            draw.line([(i * cell_size, 0), (i * cell_size, img_size)], fill="#D0C8B8", width=2)
            draw.line([(0, i * cell_size), (img_size, i * cell_size)], fill="#D0C8B8", width=2)
            
        # ネコ（赤丸）の位置を描画
        for r, c in coords:
            # 1-indexed to 0-indexed
            r_idx = r - 1
            c_idx = c - 1
            cx = (c_idx + 0.5) * cell_size
            cy = (r_idx + 0.5) * cell_size
            rad = cell_size * 0.35
            
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline="red", width=6)
            draw.ellipse([cx - rad*0.35, cy - rad*0.35, cx + rad*0.35, cy + rad*0.35], fill="red")
            
        out_buf = io.BytesIO()
        img.save(out_buf, "JPEG", quality=90)
        out_buf.seek(0)
        return send_file(out_buf, mimetype="image/jpeg")
    except Exception as e:
        logging.error(f"Render solution failed: {e}")
        return "Error", 500


# ==========================================
# 3. 画像解析 (PIL + NumPy)
# ==========================================
def process_puzzle_image(image_bytes, host_url):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    bw = int(w * 0.885)
    bh = bw
    bx = int((w - bw) / 2)
    y_offsets = [int(h * 0.313), int(h * 0.310), int(h * 0.316), int(h * 0.305), int(h * 0.320)]

    for by in y_offsets:
        for N in [9, 10]:
            cell_w = bw / N
            cell_h = bh / N
            colors_samples = []

            for r in range(N):
                row_samples = []
                for c in range(N):
                    cx = int(bx + (c + 0.5) * cell_w)
                    cy = int(by + (r + 0.5) * cell_h)
                    patch = img_np[max(0, cy-3):min(h, cy+4), max(0, cx-3):min(w, cx+4)]
                    mean_rgb = patch.mean(axis=(0, 1))
                    row_samples.append(mean_rgb)
                colors_samples.append(row_samples)

            flat_samples = np.array(colors_samples).reshape(-1, 3)
            kmeans = KMeans(n_clusters=N, random_state=42, n_init=10).fit(flat_samples)
            labels = kmeans.labels_.reshape(N, N)

            solution = solve_meowdoku(labels)
            if solution is not None:
                draw = ImageDraw.Draw(img)
                cat_coords = []
                coords_param_list = []
                for r in range(N):
                    for c in range(N):
                        if solution[r][c] == 1:
                            cat_coords.append((r + 1, c + 1))
                            coords_param_list.append(f"{r+1}_{c+1}")
                            cx = bx + (c + 0.5) * cell_w
                            cy = by + (r + 0.5) * cell_h
                            rad = min(cell_w, cell_h) * 0.38
                            
                            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline="red", width=7)
                            draw.ellipse([cx - rad*0.35, cy - rad*0.35, cx + rad*0.35, cy + rad*0.35], fill="red")

                # 画像をバイト列に変換
                out_buffer = io.BytesIO()
                img.save(out_buffer, format="JPEG", quality=85)
                out_bytes = out_buffer.getvalue()

                # 優先1: Imgur 公開アップロード
                public_image_url = upload_to_imgur(out_bytes)

                # 優先2: 万が一Imgurがタイムアウト・エラーの場合の自前動的画像URL
                if not public_image_url:
                    cats_str = ",".join(coords_param_list)
                    public_image_url = f"{host_url}/render_solution?cats={cats_str}&N={N}"

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
    # リクエスト元のホスト名を取得 (例: https://meowdoku-bot.vercel.app)
    host_url = request.host_url.rstrip('/')

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_blob_api = MessagingApiBlob(api_client)

        try:
            image_bytes = line_bot_blob_api.get_message_content(message_id=message_id)
            solution, cat_coords, image_url = process_puzzle_image(image_bytes, host_url)

            messages = []

            # 1. テキスト形式のネコの位置情報
            coord_text = "🐱 ネコの位置（上から何行目 - 左から何列目）：\n"
            coord_text += "\n".join([f"・{r}行目 - {c}列目" for r, c in cat_coords])
            messages.append(TextMessage(text=coord_text))

            # 2. 画像メッセージ (確実に画像URLをセット)
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
