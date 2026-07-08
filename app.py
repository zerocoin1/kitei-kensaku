"""規定類 検索チャット（沖縄支社パイロット版）

起動:  streamlit run app.py
"""
import streamlit as st

from rag import config
from rag import sources as src
from rag import indexer
from rag import query_engine as qe

st.set_page_config(page_title="規定類 検索チャット", page_icon="📖", layout="centered")


# ---------- 認証（簡易共通パスワード）----------
def check_password() -> bool:
    if st.session_state.get("authed"):
        return True
    st.title("📖 規定類 検索チャット")
    st.caption("沖縄支社メンバー専用")
    pw = st.text_input("共通パスワード", type="password")
    if st.button("ログイン"):
        if pw and pw == st.secrets.get("app_password", ""):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
    return False


# ---------- シークレット取得 ----------
def get_api_key() -> str:
    return st.secrets["GOOGLE_API_KEY"]


def get_sa_info():
    if "gcp_service_account" in st.secrets:
        return dict(st.secrets["gcp_service_account"])
    return None


# ---------- インデックス（プロセス内でキャッシュ）----------
@st.cache_resource(show_spinner="インデックスを読み込み中…")
def get_index(cache_bust: str):
    return indexer.load_index()


def rebuild(api_key: str) -> None:
    sa = get_sa_info()
    root = st.secrets.get("drive_root_folder_id")
    try:
        with st.spinner("同期＋インデックス再構築中…（PDF量により数分かかる場合があります）"):
            records = src.collect_records(sa, root)
            if not records:
                st.warning(
                    "対象PDFが見つかりませんでした。フォルダ構造 "
                    "「<ルート>/<会社>/<種目>/xxx.pdf」を確認してください。"
                )
                return
            result = indexer.build_index(records, api_key)
        get_index.clear()
        st.session_state["cache_bust"] = src.load_manifest().get("signature", "")
        msg = (
            f"完了：全{len(records)}件中 **{result['updated']}件を更新**"
            f"（{result['reused']}件は変更なしで再利用）／ 索引 {result['pages']} ページ"
        )
        if result["removed"]:
            msg += f"／ 削除 {result['removed']} 件"
        st.success(msg)
    except Exception as e:
        st.error(f"データ更新でエラーが発生しました：{e}")


def product_options(store, company: str, product_type: str) -> list[str]:
    """索引に実在する商品だけを「すべて」に続けて並べる。"""
    if not store:
        return [config.PRODUCT_ANY]
    found = sorted({
        m["product"] for m in store["meta"]
        if m["company"] == company and m["product_type"] == product_type
    })
    return [config.PRODUCT_ANY] + found


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander("根拠（出典）を表示"):
        for s in sources:
            st.markdown(
                f"- **{s.file_name}** — 資料 P{s.doc_page}（PDF {s.pdf_page}ページ目）"
            )


# ---------- メイン ----------
def main() -> None:
    if not check_password():
        st.stop()

    api_key = get_api_key()
    index = get_index(st.session_state.get("cache_bust", "init"))
    st.session_state.setdefault("messages", [])  # [{role, content, sources}]

    # ===== サイドバー =====
    with st.sidebar:
        st.header("検索条件")
        company = st.selectbox("保険会社", config.COMPANIES)
        product_type = st.selectbox("保険種目", config.PRODUCT_TYPES)
        product = st.selectbox(
            "商品（任意）", product_options(index, company, product_type),
            help="「すべて」のままなら種目内を横断検索します。特定商品に絞ると精度が上がります。",
        )
        st.caption("※ 検索条件は次の質問から反映されます。会社を変えたら会話クリア推奨。")

        if st.button("🗑 会話をクリア"):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        st.header("管理者用")
        if st.button("🔄 データ更新（最新PDFを同期）"):
            rebuild(api_key)
        n = len(src.load_manifest().get("records", []))
        st.caption(f"取り込み済みPDF: {n} 件")

    # ===== メイン =====
    st.title("📖 規定類 検索チャット")

    if index is None:
        st.info("まだデータが取り込まれていません。左サイドバーの「🔄 データ更新」を押してください。")
        st.stop()

    label = f"{company} ／ {product_type}"
    if product and product != config.PRODUCT_ANY:
        label += f" ／ {product}"
    st.caption(f"検索対象：{label}　｜　続けて質問すると文脈を踏まえて深掘りできます。")

    # これまでの会話を描画
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("sources"))

    # 入力
    question = st.chat_input("質問を入力（例：弁護士費用特約の対象は誰ですか？）")
    if question and question.strip():
        question = question.strip()
        st.session_state["messages"].append(
            {"role": "user", "content": question, "sources": None}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("規定を確認中…"):
                # 今回の質問は含めず、それ以前の履歴を渡す
                history = [
                    {"role": r["role"], "content": r["content"]}
                    for r in st.session_state["messages"][:-1]
                ]
                ans = qe.ask(
                    index, api_key, company, product_type, question,
                    product, history,
                )
            st.markdown(ans.text)
            render_sources(ans.sources)

        st.session_state["messages"].append(
            {"role": "assistant", "content": ans.text, "sources": ans.sources}
        )

    st.caption(
        "⚠️ 本回答はAIによる検索補助です。重要な判断は必ず規定の原本でご確認ください。"
    )


if __name__ == "__main__":
    main()
