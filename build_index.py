"""Tạo chỉ mục embedding cho từng file trong data/ (chỉ xử lý file mới hoặc đã sửa)."""
import glob
import hashlib
import os
import sys
import time

import numpy as np
from google import genai

from common import embed_batch, file_hash, normalize, split_text

INDEX_DIR = "index"
BATCH = 50
MAX_PER_MIN = 80  # thấp hơn hạn mức miễn phí 100/phút


def index_path(name):
    return os.path.join(INDEX_DIR, hashlib.md5(name.encode("utf-8")).hexdigest()[:12] + ".npz")


def main():
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Thiếu biến môi trường GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    os.makedirs(INDEX_DIR, exist_ok=True)

    wanted = set()
    for path in sorted(glob.glob("data/*.txt")):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        h = file_hash(text)
        out = index_path(name)
        wanted.add(out)

        if os.path.exists(out):
            with np.load(out) as saved:
                if str(saved["hash"]) == h:
                    print(f"Bỏ qua (không đổi): {name}")
                    continue

        chunks = split_text(text)
        if not chunks:
            print(f"Bỏ qua (file rỗng): {name}")
            wanted.discard(out)
            continue

        print(f"Đang xử lý: {name} ({len(chunks)} đoạn)")
        vectors = []
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            vectors.extend(embed_batch(client, batch, "RETRIEVAL_DOCUMENT"))
            time.sleep(60 * len(batch) / MAX_PER_MIN)  # giữ dưới hạn mức
        np.savez_compressed(out, source=np.array(name), chunks=np.array(chunks),
                            vectors=normalize(vectors), hash=np.array(h))

    # Xóa chỉ mục của file đã bị xóa khỏi data/
    for old in glob.glob(os.path.join(INDEX_DIR, "*.npz")):
        if old not in wanted:
            os.remove(old)
            print(f"Đã xóa chỉ mục cũ: {old}")


if __name__ == "__main__":
    main()