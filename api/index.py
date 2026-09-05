import os
import io
import uuid
import time
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

TMP_DIR = "/tmp"

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


# 無料画像ホスティングへの多重アタック
def upload_image_to_cloud(image_bytes):
    unique_name = f"sol_{uuid.uuid4().hex[:10]}.jpg"
    
    # 1. FreeImageHost API
    try:
        url = "https://freeimage.host/api/1/upload"
        data = {
            'key': '6d207e641c6e4d4282047490a09e16f3',
            'action': 'upload',
            'source': base64.b64encode(image_bytes).decode('utf-8'),
            'format': 'json'
        }
        res = requests.post(url, data=data, timeout=5)
        j = res.json()
        if 'image' in j and 'url' in j['image']:
            return j['image']['url']
    except Exception as e:
        logging.error(f"FreeImageHost failed: {e}")

    # 2. Catbox.moe
    try:
        url = "https://catbox.moe/user/api.php"
        files = {'fileToUpload': (unique_name, image_bytes, 'image/jpeg')}
        data = {'reqtype': 'fileupload'}
        res = requests.post(url, files=files, data=data, timeout=5)
        if res.status_code == 200 and res.text.startswith("http"):
            return res.text.strip()
    except Exception as e:
        logging.error(f"Catbox upload failed: {e}")

    # 3. Imgur
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


@app.route('/solution/<filename>')
def serve_solution(filename):
    file_path = os.path.join(TMP_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    return "Not Found", 404


# ==========================================
# 2. カラフルセル・ダイレクト外挿（ズレゼロ検出）
# ==========================================
def find_board_color_bounding(img_np):
    """
    上部ルールカードや背景白に惑わされず、カラフルなパズルマスの外周を直接ピクセル検出
    """
    h, w, _ = img_np.shape
    
    # 検索範囲：画面の25%〜80%
    y_start = int(h * 0.25)
    y_end = int(h * 0.80)
    sub_img = img_np[y_start:y_end, :, :]
    
    r = sub_img[..., 0].astype(int)
    g = sub_img[..., 1].astype(int)
    b = sub_img[..., 2].astype(int)
    
    # 彩度判定（RGBの最大と最小の差が30以上の有彩色ピクセル）
    color_diff = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(b - r))
    is_colored = color_diff > 30

    y_indices, x_indices = np.where(is_colored)

    if len(y_indices) > 50 and len(x_indices) > 50:
        min_x = np.min(x_indices)
        max_x = np.max(x_indices)
        min_y = y_start + np.min(y_indices)
        max_y = y_start + np.max(y_indices)

        bw = max_x - min_x
        bh = max_y - min_y

        return min_x, min_y, bw, bh

    # バックアップ用
    bw = int(w * 0.885)
    bx = int((w - bw) / 2)
    by = int(h * 0.312)
    return bx, by, bw, bw


# ==========================================
# 3. 画像解析 (元画像オーバーレイ)
# ==========================================
def process_puzzle_image(image_bytes, host_url):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    bx, by, bw, bh = find_board_color_bounding(img_np)

    for N in [9, 10]:
        cell_w = bw / N
        cell_h = bh / N
        colors_samples = []

        for r in range(N):
            row_samples = []
            for c in range(N):
                cx = int(bx + (c + 0.5) * cell_w)
                cy = int(by + (r + 0.5) * cell_h)
                patch = img_np[max(0, cy-4):min(h, cy+5), max(0, cx-4):min(w, cx+5)]
                mean_rgb = patch.mean(axis=(0, 1))
                row_samples.append(mean_rgb)
            colors_samples.append(row_samples)

        flat_samples = np.array(colors_samples).reshape(-1, 3)
        kmeans = KMeans(n_clusters=N, random_state=42, n_init=15).fit(flat_samples)
        labels = kmeans.labels_.reshape(N, N)

        solution = solve_meowdoku(labels)
        if solution is not None:
            draw = ImageDraw.Draw(img)
            cat_coords = []
            
            for r in range(N):
                for c in range(N):
                    if solution[r][c] == 1:
                        cat_coords.append((r + 1, c + 1))
                        # マス目の中心座標を正確に計算
                        cx = bx + (c + 0.5) * cell_w
                        cy = by + (r + 0.5) * cell_h
                        rad = min(cell_w, cell_h) * 0.38
                        
                        # 元画像上の該当マスの中央にピッタリ赤丸を描画
                        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline="#E60012", width=7)
                        draw.ellipse([cx - rad*0.35, cy - rad*0.35, cx + rad*0.35, cy + rad*0.35], fill="#E60012")

            out_buffer = io.BytesIO()
            img.save(out_buffer, format="JPEG", quality=90)
            out_bytes = out_buffer.getvalue()

            public_image_url = upload_image_to_cloud(out_bytes)

            filename = f"sol_{uuid.uuid4().hex}.jpg"
            local_save_path = os.path.join(TMP_DIR, filename)
            with open(local_save_path, "wb") as f:
                f.write(out_bytes)

            if not public_image_url:
                public_image_url = f"{host_url}/solution/{filename}"

            cache_key = f"v={int(time.time())}_{uuid.uuid4().hex[:6]}"
            if "?" in public_image_url:
                public_image_url += f"&{cache_key}"
            else:
                public_image_url += f"?{cache_key}"

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
    host_url = request.host_url.rstrip('/')

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_blob_api = MessagingApiBlob(api_client)

        try:
            image_bytes = line_bot_blob_api.get_message_content(message_id=message_id)
            solution, cat_coords, image_url = process_puzzle_image(image_bytes, host_url)

            messages = []

            coord_text = "🐱 ネコの位置（上から何行目 - 左から何列目）：\n"
            coord_text += "\n".join([f"・{r}行目 - {c}列目" for r, c in cat_coords])
            messages.append(TextMessage(text=coord_text))

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
