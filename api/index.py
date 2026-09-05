import os
import io
import uuid
import time
import logging
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


# ==========================================
# 2. PC版LINE 表示対応・静的画像配信ルート
# ==========================================
@app.route('/solution/<filename>')
def serve_solution(filename):
    file_path = os.path.join(TMP_DIR, filename)
    if os.path.exists(file_path):
        res = send_file(file_path, mimetype='image/jpeg')
        res.headers["Cache-Control"] = "public, max-age=86400"
        res.headers["Access-Control-Allow-Origin"] = "*"
        res.headers["Accept-Ranges"] = "bytes"
        return res
    return "Not Found", 404


# ==========================================
# 3. ブレゼロ！堅牢な白カード枠ベースの盤面検出
# ==========================================
def find_board_card_robust(img_np):
    """
    外周マスの色（淡い色など）に影響されず、盤面の白カード外枠を幾何学的にブレゼロで抽出
    """
    h, w, _ = img_np.shape
    
    # 白背景領域 (RGB > 242)
    is_white = (img_np[:, :, 0] > 242) & (img_np[:, :, 1] > 242) & (img_np[:, :, 2] > 242)

    # 画面の縦 28% 〜 78% の範囲で、盤面の太い白カード枠を探す
    y_min_search = int(h * 0.28)
    y_max_search = int(h * 0.78)

    # 画面中央の縦線における白ピクセルの分布から白カード領域の上端・下端を検索
    center_x = int(w / 2)
    center_column_white = is_white[y_min_search:y_max_search, center_x]
    
    white_indices = np.where(center_column_white)[0]

    if len(white_indices) > 50:
        card_top_y = y_min_search + white_indices[0]
        card_bottom_y = y_min_search + white_indices[-1]
        card_height = card_bottom_y - card_top_y

        # 白カードの幅は高度とほぼ同じ（正方形）
        card_width = card_height
        card_left_x = int((w - card_width) / 2)

        # 白カードの内側の実際のパズルグリッド位置（内側余白パディング約1.8%を除外）
        margin = card_width * 0.018
        grid_x = card_left_x + margin
        grid_y = card_top_y + margin
        grid_size = card_width - (margin * 2)

        return grid_x, grid_y, grid_size, grid_size

    # フォールバック幾何推定
    bw = int(w * 0.885)
    bx = int((w - bw) / 2)
    by = int(h * 0.312)
    return bx, by, bw, bw


# ==========================================
# 4. 元画像高精度解析 & オーバーレイ
# ==========================================
def process_puzzle_image(image_bytes, host_url):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    # ブレゼロ幾何検出
    bx, by, bw, bh = find_board_card_robust(img_np)

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
                        cx = bx + (c + 0.5) * cell_w
                        cy = by + (r + 0.5) * cell_h
                        rad = min(cell_w, cell_h) * 0.38
                        
                        # 元画像の中心にピッタリ赤丸を描画
                        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline="#E60012", width=7)
                        draw.ellipse([cx - rad*0.35, cy - rad*0.35, cx + rad*0.35, cy + rad*0.35], fill="#E60012")

            filename = f"sol_{uuid.uuid4().hex[:12]}.jpg"
            local_save_path = os.path.join(TMP_DIR, filename)
            img.save(local_save_path, "JPEG", quality=90)

            base_image_url = f"{host_url}/solution/{filename}"
            cache_key = f"v={int(time.time())}"
            public_image_url = f"{base_image_url}?{cache_key}"

            return solution, cat_coords, public_image_url

    raise ValueError("解答パターンが見つかりませんでした。")


# ==========================================
# 5. Webhook エンドポイント
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
# 6. LINE メッセージ処理
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
