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

    try:
        url = "https://catbox.moe/user/api.php"
        files = {'fileToUpload': (unique_name, image_bytes, 'image/jpeg')}
        data = {'reqtype': 'fileupload'}
        res = requests.post(url, files=files, data=data, timeout=5)
        if res.status_code == 200 and res.text.startswith("http"):
            return res.text.strip()
    except Exception as e:
        logging.error(f"Catbox upload failed: {e}")

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
        res = send_file(file_path, mimetype='image/jpeg')
        res.headers["Cache-Control"] = "public, max-age=86400"
        res.headers["Access-Control-Allow-Origin"] = "*"
        res.headers["Accept-Ranges"] = "bytes"
        return res
    return "Not Found", 404


# ==========================================
# 3. エッジ密度スキャンによる100%確実な盤面検出
# ==========================================
def find_perfect_board_offset(img_np):
    """
    色の濃淡やルールカードの白領域に一切依存せず、
    マスの境界線（エッジ）が最も密集している正方形領域を幾何学的に特定する
    """
    h, w, _ = img_np.shape
    
    # 盤面の横幅は画面幅の約 88.5% で左右中央に固定されている
    bw = int(w * 0.885)
    bh = bw
    bx = int((w - bw) / 2)
    
    # X軸方向を盤面領域に絞り込んでエッジ強度を計算
    region = img_np[:, bx:bx+bw, :].astype(np.float32)
    gray = 0.299 * region[:, :, 0] + 0.587 * region[:, :, 1] + 0.114 * region[:, :, 2]
    
    # 縦横の隣接ピクセルとの輝度差（エッジ）
    diff_y = np.abs(gray[1:, :] - gray[:-1, :])
    diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
    
    edge_map = np.zeros_like(gray)
    edge_map[:-1, :] += diff_y
    edge_map[:, :-1] += diff_x
    
    # 各行のエッジ総和
    row_edge_sums = np.sum(edge_map, axis=1)
    
    # 盤面の上端Y座標は画面の20%〜45%の間に必ず存在する
    min_y = int(h * 0.20)
    max_y = int(h * 0.45)
    
    best_by = min_y
    max_sum = -1
    
    # 盤面の高さ(bh)分のウィンドウをスライドさせ、エッジ総和が最大のY位置を探す
    for y in range(min_y, max_y):
        current_sum = np.sum(row_edge_sums[y : y+bh])
        if current_sum > max_sum:
            max_sum = current_sum
            best_by = y
            
    return bx, best_by, bw, bh


def kmeans_plus_plus_init(X, n_clusters):
    centroids = []
    centroids.append(X[np.random.randint(X.shape[0])])
    for _ in range(1, n_clusters):
        dists = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
        # Avoid division by zero if all distances are zero
        sum_dists = np.sum(dists ** 2)
        if sum_dists == 0:
            probs = np.ones(X.shape[0]) / X.shape[0]
        else:
            probs = dists ** 2 / sum_dists
        next_centroid = X[np.random.choice(X.shape[0], p=probs)]
        centroids.append(next_centroid)
    return np.array(centroids)

def simple_kmeans(X, n_clusters, max_iters=20):
    np.random.seed(42)
    centroids = kmeans_plus_plus_init(X, n_clusters).astype(np.float32)
    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(max_iters):
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(labels == new_labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            if np.any(labels == k):
                centroids[k] = X[labels == k].mean(axis=0)
    return labels

# ==========================================
# 4. 元画像高精度解析 & オーバーレイ
# ==========================================
def process_puzzle_image(image_bytes, host_url):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    # エッジスキャンによる完璧な位置検出
    bx, by, bw, bh = find_perfect_board_offset(img_np)

    for N in [9, 10]:
        cell_w = bw / N
        cell_h = bh / N
        colors_samples = []

        for r in range(N):
            row_samples = []
            for c in range(N):
                cx = int(bx + (c + 0.5) * cell_w)
                cy = int(by + (r + 0.5) * cell_h)
                # マスのド中心 5x5 ピクセルのみをサンプリング（隣の境界線を拾わない）
                patch = img_np[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)]
                mean_rgb = patch.mean(axis=(0, 1))
                row_samples.append(mean_rgb)
            colors_samples.append(row_samples)

        flat_samples = np.array(colors_samples).reshape(-1, 3)
        # sklearn.cluster.KMeans の代わりに自作関数を使う
        labels = simple_kmeans(flat_samples, n_clusters=N)
        labels = labels.reshape(N, N)

        solution = solve_meowdoku(labels)
        if solution is not None:
            draw = ImageDraw.Draw(img)
            cat_coords = []
            
            for r in range(N):
                for c in range(N):
                    if solution[r][c] == 1:
                        cat_coords.append((r + 1, c + 1))
                        # マスの中心座標
                        cx = bx + (c + 0.5) * cell_w
                        cy = by + (r + 0.5) * cell_h
                        rad = min(cell_w, cell_h) * 0.38
                        
                        # 赤丸の描画
                        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline="#E60012", width=7)
                        draw.ellipse([cx - rad*0.35, cy - rad*0.35, cx + rad*0.35, cy + rad*0.35], fill="#E60012")

            filename = f"sol_{uuid.uuid4().hex[:12]}.jpg"
            local_save_path = os.path.join(TMP_DIR, filename)
            img.save(local_save_path, "JPEG", quality=90)

            out_buffer = io.BytesIO()
            img.save(out_buffer, format="JPEG", quality=90)
            out_bytes = out_buffer.getvalue()

            public_image_url = upload_image_to_cloud(out_bytes)

            base_image_url = f"{host_url}/solution/{filename}"
            cache_key = f"v={int(time.time())}_{uuid.uuid4().hex[:6]}"
            
            if not public_image_url:
                public_image_url = f"{base_image_url}?{cache_key}"
            else:
                if "?" in public_image_url:
                    public_image_url += f"&{cache_key}"
                else:
                    public_image_url += f"?{cache_key}"

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
            error_details = str(e)
            import traceback
            tb = traceback.format_exc()
            logging.error(f"エラー: {error_details}\n{tb}")
            error_msg = TextMessage(text=f"エラーが発生しました 😿\n詳細: {error_details[:200]}")
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[error_msg]
                )
            )
