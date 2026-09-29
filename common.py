import hashlib
import re
import time

import numpy as np
from google.genai import errors, types

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def file_hash(text):
    """Mã băm gồm nội dung + cấu hình; đổi một trong hai thì phải tạo lại chỉ mục."""
    head = f"{EMBED_MODEL}|{EMBED_DIM}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|"
    return hashlib.sha256((head + text).encode("utf-8")).hexdigest()


def embed_batch(client, texts, task_type, max_retries=8):
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=EMBED_DIM
                ),
            )
            return [e.values for e in result.embeddings]
        except errors.ClientError as e:
            if e.code != 429:
                raise
            if "PerDay" in str(e):  # hết hạn mức theo NGÀY: chờ thử lại cũng vô ích
                raise RuntimeError(
                    "Đã hết hạn mức embedding miễn phí TRONG NGÀY của Google (1000 lượt/ngày). "
                    "Hạn mức tự làm mới vào nửa đêm giờ Thái Bình Dương (khoảng 14h giờ Việt Nam). "
                    "Hãy chạy lại sau thời điểm đó."
                ) from e
            if attempt == max_retries - 1:
                raise
            m = re.search(r"retry in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 2 if m else 30
            time.sleep(min(wait, 65))
    raise RuntimeError("Không thể tạo embedding sau nhiều lần thử.")


def normalize(vectors):
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)