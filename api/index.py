import os
import io
import uuid
import logging
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, abort, send_file
from sklearn.cluster import KMeans
import cv2

# LINE SDK v3
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    ImageMessage,
)
from linebot.v3.webhooks import (
    MessageEvent,
    ImageMessageContent,
    TextMessageContent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

TMP_DIR = "/tmp"

# ==================================================
# ユーザー向け案内文
# ==================================================
HELP_TEXT = (
    "🐱 MeowDoku 攻略Bot の使い方 🐱\n\n"
    "【送り方】\n"
    "① パズルのスクリーンショットを用意します。\n"
    "② 画像編集アプリで、各色のマス1つずつに「×」印を手書きして\n"
    "   「どの色がどの列・行にあるか」が分かるようにしてください。\n"
    "③ その画像をこのチャットに送信すると、解答を返します。\n\n"
    "【ポイント】\n"
    "・×印は各色のマスに1つずつ（合計N個）書いてください。\n"
    "・×は黒や濃い色でハッキリと書くと認識精度が上がります。\n"
    "・盤面全体が画面に収まるよう撮影してください。"
)


# ==================================================
# 1. パズル解法ロジック（バックトラッキング）
# ==================================================
def solve_meowdoku(grid):
    """
    grid: N×N の 2D リスト。各要素は色ID（0始まり整数）。
    戻り値: 解答グリッド（猫を置く位置が 1）、解なしなら None。
    制約:
      - 各行に猫は1匹のみ
      - 各列に猫は1匹のみ
      - 各色領域に猫は1匹のみ
      - 斜め含む隣接マスに猫は置けない
    """
    N = len(grid)

    solution = [[0] * N for _ in range(N)]
    used_cols = [False] * N
    used_colors = [False] * N
    cats_pos = []

    def is_safe(r, c, color_id):
        if used_cols[c]:
            return False
        if color_id < N and used_colors[color_id]:
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
                if color_id < N:
                    used_colors[color_id] = True
                cats_pos.append((r, c))

                if backtrack(r + 1):
                    return True

                solution[r][c] = 0
                used_cols[c] = False
                if color_id < N:
                    used_colors[color_id] = False
                cats_pos.pop()
        return False

    if backtrack(0):
        return solution
    return None


# ==================================================
# 2. ×印の検出
# ==================================================
def detect_cross_marks(img_np):
    """
    画像から×印の中心座標リストを返す。
    手法:
      1. グレースケール化 → Canny エッジ検出
      2. 膨張処理でエッジを太くする
      3. HoughLinesP で直線セグメントを検出
      4. 斜め線（45° ± 20°）のみ抽出
      5. 近い座標どうしをクラスタリングして交差点（×の中心）を確定
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # コントラスト強調
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ガウスブラーでノイズ除去
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny エッジ検出
    edges = cv2.Canny(blurred, 50, 150)

    # 膨張でエッジを繋ぐ
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 確率的ハフ変換で線分検出
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=15,
        maxLineGap=8,
    )

    if lines is None:
        return []

    # 斜め線（±45° 付近）だけ残す
    diag_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        # 30°〜60° または 120°〜150° の線を斜め線とみなす
        if (30 <= angle <= 60) or (120 <= angle <= 150):
            diag_lines.append(line[0])

    if not diag_lines:
        return []

    # 各線分の中点を計算
    midpoints = np.array(
        [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in diag_lines],
        dtype=np.float32,
    )

    # 中点を DBSCAN 的にクラスタリング（近い点をまとめる）
    crosses = _cluster_points(midpoints, radius=30)
    return crosses


def _cluster_points(points, radius=30):
    """近い点をまとめて重心を返す（シンプルなグリーディクラスタリング）。"""
    if len(points) == 0:
        return []
    used = [False] * len(points)
    clusters = []
    for i, pt in enumerate(points):
        if used[i]:
            continue
        group = [pt]
        used[i] = True
        for j in range(i + 1, len(points)):
            if not used[j]:
                dist = np.linalg.norm(points[j] - pt)
                if dist < radius:
                    group.append(points[j])
                    used[j] = True
        clusters.append(np.mean(group, axis=0))
    return clusters


# ==================================================
# 3. ×印の座標からグリッド境界を復元
# ==================================================
def infer_grid_from_crosses(crosses, img_w, img_h):
    """
    ×印の中心座標群からグリッドのセル境界（行・列）を推定する。

    戻り値:
      col_edges: 列の左端 x 座標リスト（長さ N+1）
      row_edges: 行の上端 y 座標リスト（長さ N+1）
      N: グリッドサイズ
    None を返す場合は推定失敗。
    """
    if len(crosses) < 4:
        return None

    xs = np.array([c[0] for c in crosses])
    ys = np.array([c[1] for c in crosses])

    # ×印の数から N を推測（meowdoku は通常 8〜12）
    # 試せるサイズ
    candidate_Ns = list(range(6, 13))

    best = None
    best_score = float("inf")

    for N in candidate_Ns:
        if len(crosses) < N:
            continue

        # KMeans で x 座標を N グループに分類
        try:
            km_x = KMeans(n_clusters=N, random_state=0, n_init=10).fit(xs.reshape(-1, 1))
            km_y = KMeans(n_clusters=N, random_state=0, n_init=10).fit(ys.reshape(-1, 1))
        except Exception:
            continue

        centers_x = np.sort(km_x.cluster_centers_.flatten())
        centers_y = np.sort(km_y.cluster_centers_.flatten())

        # セル幅の均等性スコア（小さいほど良い）
        diffs_x = np.diff(centers_x)
        diffs_y = np.diff(centers_y)
        if len(diffs_x) == 0 or len(diffs_y) == 0:
            continue
        score = np.std(diffs_x) / (np.mean(diffs_x) + 1e-9) + np.std(diffs_y) / (
            np.mean(diffs_y) + 1e-9
        )

        if score < best_score:
            best_score = score
            best = (N, centers_x, centers_y)

    if best is None:
        return None

    N, centers_x, centers_y = best
    cell_w = float(np.mean(np.diff(centers_x))) if N > 1 else img_w / N
    cell_h = float(np.mean(np.diff(centers_y))) if N > 1 else img_h / N

    # セル中心 → セル境界（左端・上端）に変換
    col_edges = [float(cx - cell_w / 2) for cx in centers_x] + [
        float(centers_x[-1] + cell_w / 2)
    ]
    row_edges = [float(cy - cell_h / 2) for cy in centers_y] + [
        float(centers_y[-1] + cell_h / 2)
    ]

    return col_edges, row_edges, N


# ==================================================
# 4. 色サンプリング → カラーラベル生成
# ==================================================
def sample_colors(img_np, col_edges, row_edges, N):
    """
    グリッドの各セル中央付近の色をサンプリングし、
    KMeans で N 色にクラスタリングしてラベルグリッドを返す。
    """
    samples = []
    positions = []

    for r in range(N):
        for c in range(N):
            cx = int((col_edges[c] + col_edges[c + 1]) / 2)
            cy = int((row_edges[r] + row_edges[r + 1]) / 2)
            cx = np.clip(cx, 0, img_np.shape[1] - 1)
            cy = np.clip(cy, 0, img_np.shape[0] - 1)

            # 7×7 パッチの平均色
            patch = img_np[
                max(0, cy - 3) : cy + 4, max(0, cx - 3) : cx + 4
            ]
            mean_rgb = patch.mean(axis=(0, 1))
            samples.append(mean_rgb)
            positions.append((r, c))

    flat = np.array(samples)
    km = KMeans(n_clusters=N, random_state=42, n_init=10).fit(flat)
    labels = km.labels_

    grid = [[0] * N for _ in range(N)]
    for idx, (r, c) in enumerate(positions):
        grid[r][c] = int(labels[idx])

    return grid


# ==================================================
# 5. メイン：画像処理パイプライン
# ==================================================
def process_puzzle_image(image_bytes):
    """
    送られた画像を処理して解答画像を /tmp に保存し、ファイル名を返す。
    新方式:
      1. ×印を検出してグリッドを実測
      2. セル色をサンプリングしてカラーラベルを生成
      3. バックトラッキングで解答を求める
      4. 実測グリッド座標に丸を描画して返す
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # --- ×印を検出 ---
    crosses = detect_cross_marks(img_np)
    logger.info(f"検出した×印の数: {len(crosses)}")

    if len(crosses) < 4:
        raise ValueError(
            f"×印が{len(crosses)}個しか検出できませんでした（最低4個必要）。"
            "各色のマスに×印を書いてから送り直してください。"
        )

    # --- グリッド境界を推定 ---
    result = infer_grid_from_crosses(crosses, w, h)
    if result is None:
        raise ValueError(
            "グリッド境界の推定に失敗しました。"
            "×印が盤面全体に均等に分布するよう書いてください。"
        )

    col_edges, row_edges, N = result
    logger.info(f"推定グリッドサイズ: {N}×{N}")

    # --- 色サンプリング ---
    grid = sample_colors(img_np, col_edges, row_edges, N)

    # --- 解答計算 ---
    solution = solve_meowdoku(grid)
    if solution is None:
        raise ValueError(
            f"解答が見つかりませんでした（{N}×{N}グリッド）。"
            "×印の位置や色の認識が正しいか確認してください。"
        )

    # --- 解答を描画 ---
    draw = ImageDraw.Draw(img)
    for r in range(N):
        for c in range(N):
            if solution[r][c] == 1:
                cx = (col_edges[c] + col_edges[c + 1]) / 2
                cy = (row_edges[r] + row_edges[r + 1]) / 2
                cell_w = col_edges[c + 1] - col_edges[c]
                cell_h = row_edges[r + 1] - row_edges[r]
                rad = min(cell_w, cell_h) * 0.35

                # 外側の赤丸（輪郭）
                draw.ellipse(
                    [cx - rad, cy - rad, cx + rad, cy + rad],
                    outline="red",
                    width=5,
                )
                # 内側の赤い点
                dot_r = rad * 0.35
                draw.ellipse(
                    [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                    fill="red",
                )

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(TMP_DIR, filename)
    img.save(save_path, "JPEG", quality=90)
    return filename


# ==================================================
# 6. Flask ルーティング
# ==================================================
@app.route("/", methods=["GET"])
def index():
    return "MeowDoku LINE Bot is Running!"


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@app.route("/solution/<filename>")
def serve_solution(filename):
    # パストラバーサル防止：ファイル名のみ許可
    safe_name = os.path.basename(filename)
    file_path = os.path.join(TMP_DIR, safe_name)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/jpeg")
    return "Not Found", 404


# ==================================================
# 7. LINE イベントハンドラ
# ==================================================
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    """テキストメッセージを受け取ったら使い方を案内する。"""
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=HELP_TEXT)],
            )
        )


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    """画像を受け取ってパズルを解いて返す。"""
    message_id = event.message.id
    host_url = request.host_url.rstrip("/")

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_blob_api = MessagingApiBlob(api_client)

        try:
            image_bytes = line_bot_blob_api.get_message_content(
                message_id=message_id
            )
            solution_filename = process_puzzle_image(image_bytes)

            image_url = f"{host_url}/solution/{solution_filename}"
            reply = ImageMessage(
                original_content_url=image_url,
                preview_image_url=image_url,
            )
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[reply],
                )
            )
        except ValueError as e:
            # ユーザーに分かりやすいエラーを返す
            logger.warning(f"処理エラー: {e}")
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=f"😿 {e}\n\n{HELP_TEXT}")],
                )
            )
        except Exception as e:
            logger.error(f"予期せぬエラー: {e}", exc_info=True)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text="画像の処理中にエラーが発生しました 😿\nもう一度試してください。"
                        )
                    ],
                )
            )
