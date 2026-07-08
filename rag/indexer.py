"""PDF → ページ単位のベクトル索引の構築・保存・読込（軽量構成）。

LlamaIndex等の重いフレームワークは使わず、
  pypdf（PDF読取） + google-genai（埋め込み） + numpy（類似度計算）
だけで完結させています。依存が少なく壊れにくいのが狙いです。

索引は data/index/store.pkl に保存:
    {"vecs": np.ndarray(float32, [N, 次元]), "meta": [ {...}, ... ]}
meta 1件 = PDFの1ページ。
"""
from __future__ import annotations

import time
import pickle

import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types

from . import config
from .sources import DocRecord, save_manifest


def _client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


# 埋め込みのレート制御：無料/有料枠の「毎分トークン上限(TPM)」に当たらないよう、
# 送信ペースを文字数ベースで抑える（char≈tokenの安全側近似）。
EMBED_BATCH = 30                 # 1リクエストあたりのテキスト数
EMBED_CHARS_PER_MIN = 600_000    # 1分あたりに送る文字数の上限（実TPM上限より低めに設定）


def embed_texts(texts: list[str], api_key: str, task: str) -> np.ndarray:
    """テキスト群を埋め込みベクトルに変換（バッチ＋レート制御＋リトライ）。

    task は "RETRIEVAL_DOCUMENT"（索引側）か "RETRIEVAL_QUERY"（質問側）。
    戻り値は L2正規化済みのベクトル（検索時は内積＝コサイン類似度になる）。
    """
    client = _client(api_key)
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        chunk = texts[start:start + EMBED_BATCH]
        for attempt in range(8):
            try:
                resp = client.models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(
                        task_type=task,
                        output_dimensionality=config.EMBED_DIM,
                    ),
                )
                out.extend([e.values for e in resp.embeddings])
                break
            except Exception as e:
                if attempt == 7:
                    raise
                msg = str(e)
                # 429（毎分上限超過）は約1分で回復するので長めに待つ
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper():
                    time.sleep(30)
                else:  # 503などの一時的な混雑
                    time.sleep(2 * (attempt + 1))
        # 次のバッチまで、送った文字数に応じて待機し毎分上限の超過を防ぐ
        chars = sum(len(t) for t in chunk)
        time.sleep(chars / EMBED_CHARS_PER_MIN * 60)
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _pdf_to_pages(record: DocRecord) -> list[dict]:
    """1つのPDFをページ単位のメタ情報リストに変換する。"""
    reader = PdfReader(record.local_path)
    pages: list[dict] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) < config.MIN_PAGE_CHARS:
            continue  # 白紙・画像のみページは索引に入れない
        pdf_page = i + 1                              # PDF内の物理ページ（1始まり）
        doc_page = pdf_page + record.page_offset      # 資料上の通しページ（推定）
        pages.append({
            "source_key": _source_key(record),   # どのファイル由来か（差分更新の識別子）
            "file_name": record.file_name,
            "company": record.company,
            "product_type": record.product_type,
            "product": record.product,
            "pdf_page": pdf_page,
            "doc_page": doc_page,
            "text": text,
        })
    return pages


def _source_key(record: DocRecord) -> str:
    """ファイルを一意に識別するキー（会社/種目/ファイル名）。"""
    return f"{record.company}/{record.product_type}/{record.file_name}"


def build_index(records: list[DocRecord], api_key: str) -> dict:
    """索引を構築・保存する（差分更新）。

    前回の索引があれば、変わっていないファイルの埋め込みは再利用し、
    新規・変更されたファイルだけを読み直して再ベクトル化する。
    戻り値: {"pages", "updated", "reused", "removed"}
    """
    cur_files = {_source_key(r): r.signature for r in records}

    prev = load_index()
    prev_usable = (
        prev is not None
        and "files" in prev
        and prev["vecs"].shape[0] == len(prev["meta"])
        and (len(prev["meta"]) == 0 or "source_key" in prev["meta"][0])
    )
    prev_files = prev["files"] if prev_usable else {}

    # --- 変更なし（再利用）か、更新（再ベクトル化）かを判定 ---
    reused_meta: list[dict] = []
    reused_vecs: list[np.ndarray] = []
    reused_keys: set[str] = set()

    if prev_usable:
        # 前回のページを source_key ごとにまとめておく
        rows_by_key: dict[str, list[int]] = {}
        for i, m in enumerate(prev["meta"]):
            rows_by_key.setdefault(m["source_key"], []).append(i)
        for key, sig in cur_files.items():
            if prev_files.get(key) == sig and key in rows_by_key:
                idxs = rows_by_key[key]
                reused_keys.add(key)
                reused_meta.extend(prev["meta"][i] for i in idxs)
                reused_vecs.append(prev["vecs"][idxs])

    # --- 更新対象ファイルだけ読み直して埋め込む ---
    updated_records = [r for r in records if _source_key(r) not in reused_keys]
    new_pages: list[dict] = []
    for rec in updated_records:
        new_pages.extend(_pdf_to_pages(rec))

    if new_pages:
        new_vecs = embed_texts(
            [p["text"] for p in new_pages], api_key, "RETRIEVAL_DOCUMENT"
        )
    else:
        new_vecs = np.zeros((0, config.EMBED_DIM), dtype=np.float32)

    all_meta = reused_meta + new_pages
    if not all_meta:
        raise RuntimeError(
            "PDFからテキストを抽出できませんでした。"
            "スキャン（画像）PDFの可能性があります（文字が選択できるPDFが必要）。"
        )
    parts = reused_vecs + ([new_vecs] if new_vecs.shape[0] else [])
    combined = np.vstack(parts)

    removed = [k for k in prev_files if k not in cur_files]

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STORE_PATH, "wb") as f:
        pickle.dump({"vecs": combined, "meta": all_meta, "files": cur_files}, f)

    save_manifest(records)
    return {
        "pages": len(all_meta),
        "updated": len(updated_records),
        "reused": len(reused_keys),
        "removed": len(removed),
    }


def load_index() -> dict | None:
    """保存済みの索引を読み込む。無ければ None。"""
    if not config.STORE_PATH.exists():
        return None
    with open(config.STORE_PATH, "rb") as f:
        return pickle.load(f)
