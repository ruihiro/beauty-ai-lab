import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import sqlite3
import pandas as pd

# --- 1. 初期設定 & データベース準備 ---
st.set_page_config(page_title="Biteki AI Beauty Lab", layout="centered")

def init_db():
    """データベースの初期化（初回のみ実行されます）"""
    conn = sqlite3.connect('skin_diary.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS diary 
                 (date TEXT, target TEXT, sym_score INTEGER, r_score REAL, t_ratio REAL)''')
    conn.commit()
    conn.close()

init_db()

def save_to_db(target, sym, red, trouble):
    """診断結果をデータベースに保存"""
    conn = sqlite3.connect('skin_diary.db')
    c = conn.cursor()
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    c.execute("INSERT INTO diary VALUES (?, ?, ?, ?, ?)", (date_str, target, sym, red, trouble))
    conn.commit()
    conn.close()

# --- 2. AIエンジン設定 (MediaPipe) ---
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    mp_face_mesh = None

# --- 3. 解析ロジック群 ---
def get_face_mask(img_cv):
    h, w, _ = img_cv.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if mp_face_mesh:
        try:
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    hull_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                    points = np.array([ [int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in hull_indices ])
                    cv2.fillPoly(mask, [points], 255)
                    exclude_parts = [[33, 133, 159, 145, 153], [362, 263, 386, 374, 380], [61, 291, 0, 17], [70, 107, 55], [336, 285, 300]]
                    for feature in exclude_parts:
                        feat_pts = np.array([ [int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in feature ])
                        cv2.fillPoly(mask, [feat_pts], 0)
        except Exception: pass
    return mask

def analyze_skin_details(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    skin_mask = get_face_mask(img_cv)
    
    if cv2.countNonZero(skin_mask) == 0:
        cv2.circle(skin_mask, (w//2, h//2), int(min(w,h)*0.35), 255, -1)

    skin_area = cv2.countNonZero(skin_mask)

    # 赤み
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_and(cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255])) + cv2.inRange(hsv, np.array([170, 40, 40]), np.array([180, 255, 255])), skin_mask)
    red_score = round((cv2.countNonZero(red_mask) / skin_area) * 100, 1) if skin_area > 0 else 0

    # トラブル
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    trouble_raw = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
    trouble_mask = cv2.bitwise_and(trouble_raw, skin_mask)
    trouble_ratio = (cv2.countNonZero(trouble_mask) / skin_area) * 100 if skin_area > 0 else 0

    display_map = cv2.addWeighted(img_cv, 0.4, np.zeros_like(img_cv), 0.6, 0)
    display_map[red_mask > 0] = [0, 0, 255]
    display_map[trouble_mask > 0] = [255, 255, 0]
    
    return red_score, trouble_ratio, cv2.cvtColor(display_map, cv2.COLOR_BGR2RGB)

def get_seasonal_advice():
    month = datetime.now().month
    if month in [2, 3]: return "🌸 春のゆらぎ肌注意報", "寒暖差（三寒四温）でバリア機能が低下しやすい時期です。朝はぬるま湯洗顔、夜は摩擦レスな保湿を。"
    elif month in [4, 5, 6]: return "☀️ 紫外線対策強化月間", "UV量が急増中。日焼け止めは2時間おきの塗り直しを意識して。"
    elif month in [11, 12, 1]: return "❄️ 冬の乾燥警報", "湿度が下がり肌水分が奪われています。オイルやクリームでしっかり蓋を。"
    else: return "✨ 季節の美肌ケア", "今の肌状態に合わせて、水分と油分のバランスを整えましょう。"

# --- 4. デザイン設定 ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@700&family=Noto+Sans+JP:wght@300;400&display=swap');
    .stApp { background-color: #fdfaf9; font-family: 'Noto Sans JP', sans-serif; color: #4a4a4a; }
    .summary-card { background: linear-gradient(135deg, #fceeee 0%, #f7dada 100%); padding: 30px; border-radius: 25px; border: 3px solid #fff; box-shadow: 0 10px 30px rgba(216, 167, 167, 0.3); text-align: center; color: #5a4a4a; margin-bottom: 30px; }
    .card-title { font-family: 'Noto Serif JP', serif; font-size: 1.4rem; color: #8e6d6d; }
    .card-score { font-size: 3.5rem; font-weight: bold; color: #8e6d6d; text-shadow: 2px 2px 0px white; }
    .diag-card { background-color: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.04); text-align: center; margin-bottom: 20px;}
    .stButton>button { border-radius: 50px; background-color: #d8a7a7; color: white; height: 55px; border: none; font-weight: bold; width: 100%; font-size: 1rem; }
    .stButton>button:hover { background-color: #c08e8e; transform: translateY(-2px); }
    .item-card { background: white; padding: 15px; border-radius: 12px; border: 1px solid #eee; margin-bottom: 10px; text-align: left; display:flex; align-items:center; }
    .icon-box { font-size: 2rem; margin-right: 15px; width: 40px; text-align: center; }
    .seasonal-box { background-color: white; padding: 20px; border-radius: 15px; border-left: 6px solid #d8a7a7; box-shadow: 0 5px 15px rgba(0,0,0,0.03); margin: 20px 0; text-align: left; }
    </style>
""", unsafe_allow_html=True)

# --- 5. セッション管理 ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'ans' not in st.session_state: st.session_state.ans = {}
if 'img' not in st.session_state: st.session_state.img = None
if 'result' not in st.session_state: st.session_state.result = {}

def next_step(): st.session_state.step += 1
def reset():
    st.session_state.step = 1
    st.session_state.img = None
    st.session_state.result = {}

# --- 6. サイドバー（メニューナビゲーション） ---
st.sidebar.title("メニュー")
menu = st.sidebar.radio("機能を選択してください", ["✨ AI肌診断", "📅 肌日記（履歴）", "🔄 Before/After 比較"])

# ==========================================
# 画面1: メインのAI診断フロー
# ==========================================
if menu == "✨ AI肌診断":
    st.title("美的 AI Beauty Lab")
    
    if st.session_state.step <= 3:
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        if st.session_state.step == 1:
            st.subheader("Q1. 現在の肌悩みは？")
            st.session_state.ans['target'] = st.radio("", ["赤み・敏感", "シミ・くすみ", "毛穴・黒ずみ", "シワ・たるみ"], key="q1")
            if st.button("次へ"): next_step(); st.rerun()
            
        elif st.session_state.step == 2:
            st.subheader("Q2. 理想の肌質は？")
            st.session_state.ans['ideal'] = st.radio("", ["透明感のある肌", "ハリ・弾力肌", "トラブルのない安定肌", "毛穴レス肌"], key="q2")
            if st.button("次へ"): next_step(); st.rerun()
            
        elif st.session_state.step == 3:
            st.subheader("Photo Scan")
            st.write("明るい場所で撮影した、正面の写真をアップロードしてください。")
            file = st.file_uploader("", type=['jpg','png','jpeg'], label_visibility="collapsed")
            if file:
                img = Image.open(file)
                st.image(img, use_container_width=True)
                if st.button("AI精密診断を開始"):
                    st.session_state.img = img
                    next_step()
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.step == 4:
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        st.subheader("AI Analysis Running...")
        with st.spinner("肌の奥までスキャン中..."):
            time.sleep(1.5)
            r_score, t_ratio, d_map = analyze_skin_details(st.session_state.img)
            
            gray = st.session_state.img.convert('L').resize((200,200))
            arr = np.array(gray)
            diff = np.mean(cv2.absdiff(arr, cv2.flip(arr, 1)))
            sym_score = int(max(60, 100 - (diff * 1.2))) 

            st.session_state.result = {"r_score": r_score, "t_ratio": t_ratio, "d_map": d_map, "sym_score": sym_score}
            
            # ★ データベースに自動保存 ★
            target_str = st.session_state.ans.get('target', '未設定')
            save_to_db(target_str, sym_score, r_score, t_ratio)
            
            next_step()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.step == 5:
        res = st.session_state.result
        target = st.session_state.ans.get('target', '赤み・敏感')
        
        st.markdown(f"""
            <div class="summary-card">
                <div class="card-title">My Beauty Score</div>
                <div class="card-score">{res['sym_score']}</div>
                <div style="display:flex; justify-content: space-around; padding: 10px 0;">
                    <div><small>REDNESS</small><br><strong>{res['r_score']}%</strong></div>
                    <div><small>TROUBLE</small><br><strong>{int(res['t_ratio'])}%</strong></div>
                </div>
                <div style="margin-top:15px; font-size:0.8rem; color:#c08e8e;">#{datetime.now().strftime('%Y%m%d')} #美的AI診断</div>
            </div>
        """, unsafe_allow_html=True)

        s_title, s_text = get_seasonal_advice()
        st.markdown(f'<div class="seasonal-box"><strong style="color:#d8a7a7;">{s_title}</strong><br><span style="color:#666;">{s_text}</span></div>', unsafe_allow_html=True)

        with st.expander("🔍 AIトラブル解析マップを見る", expanded=True):
            st.image(res['d_map'], caption="赤：炎症リスク / 水色：シミ・くすみ", use_container_width=True)

        st.divider()
        st.subheader("💄 Personal Prescription")
        
        if target == "赤み・敏感" or res['r_score'] > 12: rec = {"c_n": "鎮静バリア美容液", "c_d": "抗炎症成分で赤みを鎮静。", "s_n": "ビタミンB群", "s_d": "肌荒れを防ぎ粘膜を保護。"}
        elif target == "シミ・くすみ" or res['t_ratio'] > 12: rec = {"c_n": "高濃度ビタミンC", "c_d": "メラニン生成を抑制し透明感へ。", "s_n": "L-システイン", "s_d": "ターンオーバーを促進。"}
        elif target == "シワ・たるみ": rec = {"c_n": "レチノールクリーム", "c_d": "肌の奥からハリを生成。", "s_n": "コラーゲン＆鉄", "s_d": "弾力の土台を作ります。"}
        else: rec = {"c_n": "角質ケア美容液", "c_d": "毛穴詰まりを解消。", "s_n": "ビタミンA", "s_d": "皮脂バランスを整えます。"}

        st.markdown(f"""
            <div class="item-card"><div class="icon-box">🧴</div><div><small style="color:#d8a7a7;font-weight:bold;">COSMETIC</small><br><strong>{rec['c_n']}</strong><br><span style="font-size:0.85rem;color:#666;">{rec['c_d']}</span></div></div>
            <div class="item-card"><div class="icon-box">💊</div><div><small style="color:#d8a7a7;font-weight:bold;">SUPPLEMENT</small><br><strong>{rec['s_n']}</strong><br><span style="font-size:0.85rem;color:#666;">{rec['s_d']}</span></div></div>
        """, unsafe_allow_html=True)

        st.write("")
        if st.button("もう一度診断する"): reset(); st.rerun()

# ==========================================
# 画面2: 肌日記（履歴）
# ==========================================
elif menu == "📅 肌日記（履歴）":
    st.title("📅 Skin Diary")
    st.write("過去のAI診断結果の履歴を確認できます。")
    
    conn = sqlite3.connect('skin_diary.db')
    df = pd.read_sql_query("SELECT * FROM diary ORDER BY date DESC", conn)
    conn.close()
    
    if not df.empty:
        # トレンドグラフの表示
        st.subheader("📈 スコアの推移")
        chart_data = df[['date', 'sym_score', 'r_score']].set_index('date')
        st.line_chart(chart_data)
        
        # データテーブルの表示
        st.subheader("📋 履歴一覧")
        # 列名を見やすく変更
        df_display = df.rename(columns={
            'date': '診断日時', 'target': 'メインの悩み', 
            'sym_score': '美肌スコア', 'r_score': '赤み(%)', 't_ratio': 'トラブル(%)'
        })
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("まだ診断データがありません。「✨ AI肌診断」から最初の診断を行ってみましょう！")

# ==========================================
# 画面3: Before/After 比較
# ==========================================
elif menu == "🔄 Before/After 比較":
    st.title("🔄 Before / After")
    st.write("過去の診断結果と最新の結果を比較して、スキンケアの効果を確認しましょう。")
    
    conn = sqlite3.connect('skin_diary.db')
    df = pd.read_sql_query("SELECT * FROM diary ORDER BY date DESC", conn)
    conn.close()
    
    if len(df) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            date1 = st.selectbox("比較元（Before）", df['date'], index=1)
            res1 = df[df['date'] == date1].iloc[0]
            st.markdown(f"<div class='diag-card'><h4>{date1}</h4><h1 style='color:#8e6d6d;'>{res1['sym_score']} pts</h1><p>赤み: {res1['r_score']}%<br>トラブル: {int(res1['t_ratio'])}%</p></div>", unsafe_allow_html=True)
            
        with col2:
            date2 = st.selectbox("比較先（After）", df['date'], index=0)
            res2 = df[df['date'] == date2].iloc[0]
            
            # 差分の計算
            diff_score = int(res2['sym_score'] - res1['sym_score'])
            color = "green" if diff_score > 0 else "red" if diff_score < 0 else "gray"
            sign = "+" if diff_score > 0 else ""
            
            st.markdown(f"<div class='diag-card'><h4>{date2}</h4><h1 style='color:#8e6d6d;'>{res2['sym_score']} pts</h1><p style='color:{color}; font-weight:bold;'>{sign}{diff_score} pts</p></div>", unsafe_allow_html=True)
        
        # 改善アドバイスの自動生成
        st.divider()
        st.subheader("💡 変化の分析")
        diff_red = round(res1['r_score'] - res2['r_score'], 1) # 減っている方が良い
        if diff_red > 0:
            st.success(f"素晴らしいです！赤みが {diff_red}% 改善しています。今のスキンケアが肌に合っている証拠です。")
        elif diff_red < 0:
            st.warning(f"赤みが {abs(diff_red)}% 増加しています。摩擦や乾燥に注意し、保湿を心がけてください。")
        else:
            st.info("赤みレベルは維持されています。安定した状態です。")
            
    else:
        st.info("比較を行うには、最低でも2回以上のAI診断を行う必要があります。")