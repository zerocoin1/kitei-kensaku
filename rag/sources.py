"""PDFの収集（Google Drive同期 / ローカル走査）と変更検知。

期待するフォルダ構造（Drive・ローカル共通）:
    <ルート>/<保険会社>/<保険種目>/xxxx.pdf
    例: /保険規定データ/損保ジャパン/自動車保険/約款2026.pdf

商品（BAP・THEクルマ 等）と「資料上のページ番号オフセット」は
ファイル名から自動判定します（下記 detect_product / detect_page_offset）。
"""
from __future__ import annotations

import io
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from . import config


@dataclass
class DocRecord:
    """1つのPDFを表す。"""
    file_name: str
    company: str
    product_type: str
    product: str          # 種目内の商品（BAP / THEクルマ・SGP / ドライバー保険 / 約款 / その他）
    page_offset: int      # PDFの1枚目 = 資料上の (1 + page_offset) ページ
    local_path: str
    signature: str        # 変更検知用（md5 や 更新時刻）


# ---------- ファイル名／中身からの自動判定 ----------
def _first_pages_text(local_path: str, n_pages: int = 2) -> str:
    """PDF先頭ページのテキスト（商品ライン判定用）。読めなければ空文字。"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(local_path)
        return "".join((p.extract_text() or "") for p in reader.pages[:n_pages])
    except Exception:
        return ""


def detect_product(file_name: str, local_path: str | None = None) -> str:
    """商品ラインを判定する。まずファイル名、無ければPDF先頭ページの本文で判定。"""
    for name, keywords in config.PRODUCT_RULES:
        if any(kw in file_name for kw in keywords):
            return name
    if local_path:
        text = _first_pages_text(local_path)
        for name, keywords in config.PRODUCT_CONTENT_RULES:
            if any(kw in text for kw in keywords):
                return name
    return config.PRODUCT_OTHER


def detect_page_offset(file_name: str) -> int:
    """分割PDFの「資料上の開始ページ」をファイル名の (範囲) から推定する。

    例:
      「…（表紙-第4章P81）」       → 表紙始まり     → offset 0
      「…（第4章P82-裏表紙）」     → P82始まり      → offset 81
      「…（…第2章P185-第2章P312）」 → P185始まり     → offset 184
    括弧やページ表記が無ければ 0（単独ファイル）。
    """
    groups = re.findall(r"[（(]([^（()）]*)[）)]", file_name)
    if not groups:
        return 0
    rng = groups[-1]                                  # 最後の括弧＝ページ範囲
    start = re.split(r"[-−ー―~〜]", rng)[0]            # ダッシュ前＝開始側
    if "表紙" in start:
        return 0
    m = re.search(r"[PpＰｐ](\d+)", start)
    if m:
        return int(m.group(1)) - 1
    return 0


# ---------- 共通 ----------
def _combined_signature(records: list[DocRecord]) -> str:
    """全ファイルの状態をまとめた指紋。1つでも変われば値が変わる。"""
    parts = sorted(
        f"{r.company}/{r.product_type}/{r.file_name}:{r.signature}" for r in records
    )
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(records: list[DocRecord]) -> None:
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "signature": _combined_signature(records),
        "records": [asdict(r) for r in records],
    }
    config.MANIFEST_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def needs_rebuild(records: list[DocRecord]) -> bool:
    """前回インデックス構築時から中身が変わっていれば True。"""
    manifest = load_manifest()
    return manifest.get("signature") != _combined_signature(records)


def _make_record(file_name, company, product_type, local_path, signature) -> DocRecord:
    return DocRecord(
        file_name=file_name,
        company=company,
        product_type=product_type,
        product=detect_product(file_name, str(local_path)),
        page_offset=detect_page_offset(file_name),
        local_path=str(local_path),
        signature=signature,
    )


# ---------- ローカルモード ----------
def _collect_local() -> list[DocRecord]:
    records: list[DocRecord] = []
    root = config.PDF_DIR
    if not root.exists():
        return records
    for pdf in root.rglob("*.pdf"):
        parts = pdf.relative_to(root).parts
        if len(parts) < 3:
            # 期待構造: pdfs/<会社>/<種目>/xxx.pdf に合わないものはスキップ
            continue
        company, product_type = parts[0], parts[1]
        stat = pdf.stat()
        records.append(
            _make_record(
                pdf.name, company, product_type, pdf,
                f"{stat.st_mtime_ns}:{stat.st_size}",
            )
        )
    return records


# ---------- Google Driveモード ----------
def _drive_service(sa_info: dict):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _walk_drive(service, folder_id: str, path: list[str], out: list[dict]) -> None:
    """フォルダを再帰的に辿り、PDFを out に集める。path は現在の階層名リスト。"""
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType, md5Checksum, modifiedTime)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for f in resp.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                _walk_drive(service, f["id"], path + [f["name"]], out)
            elif f["mimeType"] == "application/pdf":
                out.append({**f, "path": path})
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def _download(service, file_id: str, dest: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with io.FileIO(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _collect_drive(sa_info: dict, root_folder_id: str) -> list[DocRecord]:
    service = _drive_service(sa_info)
    found: list[dict] = []
    _walk_drive(service, root_folder_id, [], found)

    # 前回の指紋を引いて、変わったファイルだけダウンロードする
    manifest = load_manifest()
    prev = {r["local_path"]: r["signature"] for r in manifest.get("records", [])}

    records: list[DocRecord] = []
    for f in found:
        path = f["path"]
        if len(path) < 2:
            # 期待構造: <ルート>/<会社>/<種目>/xxx.pdf に合わないものはスキップ
            continue
        company, product_type = path[0], path[1]
        local_path = config.PDF_DIR / company / product_type / f["name"]
        signature = f.get("md5Checksum") or f.get("modifiedTime", "")

        if (
            str(local_path) not in prev
            or prev[str(local_path)] != signature
            or not local_path.exists()
        ):
            _download(service, f["id"], local_path)

        records.append(
            _make_record(f["name"], company, product_type, local_path, signature)
        )
    return records


# ---------- 入口 ----------
def collect_records(
    sa_info: Optional[dict] = None, root_folder_id: Optional[str] = None
) -> list[DocRecord]:
    if config.SOURCE_MODE == "local":
        return _collect_local()
    if not sa_info or not root_folder_id:
        raise ValueError(
            "Driveモードには gcp_service_account と drive_root_folder_id の設定が必要です。"
        )
    return _collect_drive(sa_info, root_folder_id)


def drive_signature(sa_info: dict, root_folder_id: str) -> str:
    """DriveのPDF一覧から合成署名を計算する（ダウンロードせず、md5だけで軽量に）。

    保存済み索引が現在のPDFと一致しているか（先祖返りしていないか）の判定に使う。
    build_index が索引に埋め込む _combined_signature と同じ形式・同じ値になる。
    """
    service = _drive_service(sa_info)
    found: list[dict] = []
    _walk_drive(service, root_folder_id, [], found)
    parts = []
    for f in found:
        path = f["path"]
        if len(path) < 2:
            continue
        company, product_type = path[0], path[1]
        sig = f.get("md5Checksum") or f.get("modifiedTime", "")
        parts.append(f"{company}/{product_type}/{f['name']}:{sig}")
    return hashlib.md5("\n".join(sorted(parts)).encode("utf-8")).hexdigest()
