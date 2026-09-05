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


# 無料画像ホスティングへのアップロード
def upload_image_to_cloud(image_bytes):
    unique_name = f"solution_{uuid.uuid4().hex}.jpg"
    
    # 1. Catbox.moe
    try:
        url = "https://catbox.moe/user/api.php"
        files = {'fileToUpload': (unique_name, image_bytes, 'image/jpeg')}
        data = {'reqtype': 'fileupload'}
        res = requests.post(url, files=files, data=data, timeout=6)
        if res.status_code == 200 and res.text.startswith("http"):
            return res.text.strip()
    except Exception as e:
        logging.error(f"Catbox upload failed: {e}")

    # 2. Imgur
    try:
        url = "https://api.imgur.com/3/image"
        headers = {"Authorization": "Client-ID 5442646d79e5d9c"}
        payload = {'image': base64.b64encode(image_bytes).decode('utf-8'), 'name': unique_name}
        res = requests.post(url, headers=headers, data=payload, timeout=6)
        data = res.json()
        if data.get('success'):
            return data['data']['link']
    except Exception as e:
        logging.error(f"Imgur upload failed: {e}")

    return None


# ==========================================
# 2. 精密な白枠ベースの盤面位置検出
# ==========================================
def find_board_precise(img_np):
    """
    白背景枠（外枠の角丸長方形）をピクセル検索して1ピクセルの狂いもなく検出
    """
    h, w, _ = img_np.shape
    
    # 画像のY方向 20% 〜 85% のエリアを対象にする（ダイナミックアイランドや下部ボタンを除外）
    y_start = int(h * 0.20)
    y_end = int(h * 0.85)
    
    # 白背景領域 (RGBが全て240以上)
    is_white = (img_np[y_start:y_end, :, 0] > 240) & \
               (img_np[y_start:y_end, :, 1] > 240) & \
               (img_np[y_start:y_end, :, 2] > 240)
               
    # 各行・各列の白ピクセル数の分布から枠の端を検索
    row_counts = is_white.sum(axis=1)
    col_counts = is_white.sum(axis=0)
    
    # 幅の大部分が白背景である行
    valid_rows = np.where(row_counts > w * 0.7)[0]
    valid_cols = np.where(col_counts > (y_end - y_start) * 0.4)[0]
    
    if len(valid_rows) > 0 and len(valid_cols) > 0:
        by1 = y_start + valid_rows[0]
        by2 = y_start + valid_rows[-1]
        bx1 = valid_cols[0]
        bx2 = valid_cols[-1]
        
        bw = bx2 - bx1
        bh = by2 - by1
        
        # 内側のグリッド位置（白枠の内側余白を約3%除外）
        margin_x = int(bw * 0.025)
        margin_y = int(bh * 0.025)
        
        grid_x = bx1 + margin_x
        grid_y = by1 + margin_y
        grid_w = bw - (margin_x * 2)
        grid_h = bh - (margin_y * 2)
        
        return grid_x, grid_y, grid_w, grid_h

    # 万が一失敗した場合の代替位置
    bw = int(w * 0.885)
    bx = int((w - bw) / 2)
    by = int(h * 0.312)
    return bx, by, bw, bw


# ==========================================
# 3. 画像解析 (元のスクリーンショットへの直接描画)
# ==========================================
def process_puzzle_image(image_bytes):
    # 元のスクリーンショットをそのまま読み込む
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    # 精密な盤面枠の検出
    bx, by, bw, bh = find_board_precise(img_np)

    # 9x9 または 10x10 の判定
    for N in [9, 10]:
        cell_w = bw / N
        cell_h = bh / N
        colors_samples = []

        for r in range(N):
            row_samples = []
            for c in range(N):
                cx = int(bx + (c + 0.5) * cell_w)
                cy = int(by + (r + 0.5) * cell_h)
                
                # サンプル領域 (セルの中心付近 9x9 ピクセル)
                patch = img_np[max(0, cy-4):min(h, cy+5), max(0, cx-4):min(w, cx+5)]
                mean_rgb = patch.mean(axis=(0, 1))
                row_samples.append(mean_rgb)
            colors_samples.append(row_samples)

        flat_samples = np.array(colors_samples).reshape(-1, 3)
        kmeans = KMeans(n_clusters=N, random_state=42, n_init=15).fit(flat_samples)
        labels = kmeans.labels_.reshape(N, N)

        solution = solve_meowdoku(labels)
        if solution is not None:
            # ★送られてきた「元のスクリーンショット画像」の上に直接赤丸を描画★
            draw = ImageDraw.Draw(img)
            cat_coords = []
            
            for r in range(N):
                for c in range(N):
                    if solution[r][c] == 1:
                        cat_coords.append((r + 1, c + 1))
                        cx = bx + (c + 0.5) * cell_w
                        cy = by + (r + 0.5) * cell_h
                        rad = min(cell_w, cell_h) * 0.38
                        
                        # 元画像上の正解位置に二重の赤丸を描画
                        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline="#E60012", width=7)
                        draw.ellipse([cx - rad*0.35, cy - rad*0.35, cx + rad*0.35, cy + rad*0.35], fill="#E60012")

            # 加工後の「元画像＋赤丸」をJPEG化
            out_buffer = io.BytesIO()
            img.save(out_buffer, format="JPEG", quality=90)
            out_bytes = out_buffer.getvalue()

            # クラウドへ保存
            public_image_url = upload_image_to_cloud(out_bytes)
            cache_key = f"v={int(time.time())}_{uuid.uuid4().hex[:6]}"

            if public_image_url:
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

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_blob_api = MessagingApiBlob(api_client)

        try:
            image_bytes = line_bot_blob_api.get_message_content(message_id=message_id)
            solution, cat_coords, image_url = process_puzzle_image(image_bytes)

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
