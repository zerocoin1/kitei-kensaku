"""会話対応のフィルタ付き検索 + Gemini生成 + 出典抽出（軽量構成）。

チャットとして深掘りできるよう、次の2段構えにしている:
  1) 会話履歴＋今回の質問 → 「それ単体で検索できる質問文」に自動変換（condense）
  2) 変換後の質問で規定を検索し、履歴も踏まえて回答を生成

選択された会社・種目（＋任意で商品）のページだけを検索対象にし、
記載が無ければ推測せず定型文を返すようプロンプトで強く縛る。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from google import genai
from google.genai import types

from . import config
from .indexer import embed_texts

NO_ANSWER = "提供された規定類には記載がありません。"

# 会話履歴として何往復ぶんをLLMに渡すか（長くなりすぎ防止）
HISTORY_TURNS = 6
# 履歴中の1発言の最大文字数（長い回答を要約せず切り詰める）
HISTORY_MSG_LIMIT = 600


# 追加質問を、会話文脈を補った「独立した検索用の質問文」に書き換える
CONDENSE_TEMPLATE = """次は損害保険に関する社内相談の会話です。
最後の質問を、それだけで規定を検索できる独立した日本語の質問文に書き換えてください。
指示語（それ・この場合 など）や省略を、会話の文脈から補ってください。
余計な説明は付けず、質問文だけを1文で出力してください。

# これまでの会話
{history}

# 最後の質問
{question}

# 書き換えた検索用の質問文
"""


QA_TEMPLATE = """あなたは損害保険会社の規定類（約款・取扱規定集・ハンドブック）に精通した社内アシスタントです。
担当者からの相談に、規定に基づいて分かりやすく答えます。

# 厳守事項（正確性・最優先）
1. 回答は下記「参考情報」に書かれている内容のみを根拠にすること。
   参考情報に無い事実（数値・条件・対象者・可否の断定など）を、
   推測・一般論・外部知識で補ってはならない。作り話は絶対に禁止。
2. 質問の前提が規定に無い概念（例：年齢・性別など、規定が対象や可否を
   定義する軸と異なるもの）を含む場合は、まず「その条件自体は規定では
   判断を左右しない」旨を一言添えたうえで、参考情報が実際に定めている
   関連事項（続柄・関係性・手続き要件など）を案内すること。
   安易に可否を断定せず、要件を提示する。
3. 参考情報に関連する記載が一切見当たらない場合にのみ、余計な説明を加えず
   次の一文だけを出力すること：「__NO_ANSWER__」
4. 数値・条件・金額・期間などは、参考情報の表現を改変せず正確に引用すること。

# 回答スタイル
- まず結論を簡潔に述べ、続いて理由・根拠、必要に応じて「今後の対応」を示す。
- これまでの会話の流れを踏まえ、追加の質問には文脈を汲んで答える。
- 日本語で、丁寧かつ実務的に。
- 回答の最後に、使った根拠のページを (根拠: P◯◯) の形で示すこと。

# これまでの会話
{history}

# 参考情報（今回の質問に関連する規定の抜粋）
---------------------
{context}
---------------------

# 今回の質問
{question}

# 回答
""".replace("__NO_ANSWER__", NO_ANSWER)


@dataclass
class Source:
    file_name: str
    doc_page: int      # 資料上の通しページ（出典表示に使う）
    pdf_page: int      # PDF内の物理ページ（原本を開く時の位置）


@dataclass
class Answer:
    text: str
    sources: list[Source]


def _generate(prompt: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=config.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return (resp.text or "").strip()
        except Exception:
            if attempt == 5:
                raise
            time.sleep(3 * (attempt + 1))
    return ""


def _format_history(history: list[dict] | None) -> str:
    """[{role, content}, ...] を LLM向けのテキストに整形（直近数往復のみ）。"""
    if not history:
        return "（まだ会話はありません）"
    recent = history[-HISTORY_TURNS * 2:]
    lines = []
    for m in recent:
        who = "担当者" if m["role"] == "user" else "アシスタント"
        content = m["content"][:HISTORY_MSG_LIMIT]
        lines.append(f"{who}: {content}")
    return "\n".join(lines)


def _condense(question: str, history: list[dict] | None, api_key: str) -> str:
    """追加質問を、文脈を補った独立検索クエリに書き換える。履歴が無ければそのまま。"""
    if not history:
        return question
    prompt = CONDENSE_TEMPLATE.format(
        history=_format_history(history), question=question
    )
    rewritten = _generate(prompt, api_key)
    return rewritten or question


def ask(
    store: dict,
    api_key: str,
    company: str,
    product_type: str,
    question: str,
    product: str | None = None,
    history: list[dict] | None = None,
) -> Answer:
    """store（indexer.load_index の戻り値）に対して、会話文脈を踏まえて質問する。

    history は [{"role": "user"/"assistant", "content": str}, ...]（今回の質問は含めない）。
    """
    meta = store["meta"]
    vecs = store["vecs"]

    # --- 会社・種目（＋任意で商品）で「物理的に」絞り込む ---
    keep = [
        i for i, m in enumerate(meta)
        if m["company"] == company
        and m["product_type"] == product_type
        and (product in (None, config.PRODUCT_ANY) or m["product"] == product)
    ]
    if not keep:
        return Answer(text=NO_ANSWER, sources=[])

    # --- 追加質問なら文脈を補って検索クエリを作る ---
    search_query = _condense(question, history, api_key)

    # --- 検索クエリを埋め込み、絞り込んだページとの類似度で上位を取る ---
    q_vec = embed_texts([search_query], api_key, "RETRIEVAL_QUERY")[0]
    sub = vecs[keep]                      # 絞り込み後のベクトル（正規化済み）
    sims = sub @ q_vec                    # 内積＝コサイン類似度
    order = np.argsort(-sims)[: config.TOP_K]
    top = [keep[j] for j in order]

    # --- 参考情報を組み立て（資料ページ番号でラベリング）---
    context = "\n\n".join(
        f"[P{meta[i]['doc_page']}]\n{meta[i]['text'][: config.PAGE_CHAR_LIMIT]}"
        for i in top
    )
    prompt = QA_TEMPLATE.format(
        history=_format_history(history), context=context, question=question
    )
    text = _generate(prompt, api_key)

    if not text or NO_ANSWER in text:
        return Answer(text=NO_ANSWER if not text else text, sources=[])

    # --- 出典（ファイル名＋資料ページ）を重複排除して返す ---
    seen = set()
    sources: list[Source] = []
    for i in top:
        m = meta[i]
        key = (m["file_name"], m["doc_page"])
        if key not in seen:
            seen.add(key)
            sources.append(Source(
                file_name=m["file_name"],
                doc_page=m["doc_page"],
                pdf_page=m["pdf_page"],
            ))
    return Answer(text=text, sources=sources)
