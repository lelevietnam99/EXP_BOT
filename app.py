import glob
import os

import numpy as np
import streamlit as st
from google import genai
from google.genai import types

from common import embed_batch, file_hash, normalize

CHAT_MODEL = st.secrets.get("CHAT_MODEL", "gemini-2.5-flash")
TOP_K = 4
INDEX_DIR = "index"

st.set_page_config(page_title="Chatbot Bài Giảng", page_icon="🙏", layout="centered")
st.title("🙏 HỎI - ĐÁP GIÁO LÝ")
st.write("Hãy đặt câu hỏi, tôi sẽ trả lời dựa trên các bài giảng đã được tải lên.")

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Chưa cấu hình GOOGLE_API_KEY trong Settings → Secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])


@st.cache_resource(show_spinner="Đang tải chỉ mục bài giảng...")
def load_index():
    """Đọc các chỉ mục đã tạo sẵn (không gọi API). Trả về (đoạn, vector, file_chưa_có_chỉ_mục)."""
    chunks, vectors, indexed = [], [], {}
    for path in sorted(glob.glob(os.path.join(INDEX_DIR, "*.npz"))):
        with np.load(path) as saved:
            chunks.extend(saved["chunks"].tolist())
            vectors.append(saved["vectors"])
            indexed[str(saved["source"])] = str(saved["hash"])

    pending = []
    for path in sorted(glob.glob("data/*.txt")):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            if indexed.get(name) != file_hash(f.read()):
                pending.append(name)

    if not chunks:
        return None, None, pending
    return chunks, np.vstack(vectors), pending


chunks, chunk_vectors, pending = load_index()

if pending:
    st.sidebar.info(
        "Các file sau đang được xử lý (khoảng vài phút), chưa dùng để trả lời được:\n\n"
        + "\n".join(f"- {n}" for n in pending)
    )

if chunks is None:
    st.warning("Chưa có chỉ mục bài giảng. Hãy kiểm tra tab Actions trên GitHub xem "
               "'Build index' đã chạy xong chưa, rồi tải lại trang.")
    st.stop()

PROMPT = """Bạn là một trợ lý ảo hỗ trợ Phật tử, được tạo ra để trả lời câu hỏi dựa trên các bài giảng của Quý Thầy.
Hãy trả lời bằng giọng điệu từ bi, hòa ái, tôn trọng và dễ hiểu.
Chỉ sử dụng thông tin trong phần "Ngữ cảnh" dưới đây để trả lời.
Nếu câu hỏi nằm ngoài ngữ cảnh bài giảng, hãy nhẹ nhàng nói rằng: "Dạ, trong phạm vi bài giảng hiện tại, Thầy chưa đề cập chi tiết đến vấn đề này. Mong bạn hoan hỷ đặt câu hỏi khác có liên quan ạ."
Tuyệt đối không tự bịa ra kiến thức ngoài.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Câu trả lời:"""


def answer_question(question):
    q_vec = normalize(embed_batch(client, [question], "RETRIEVAL_QUERY"))[0]
    top = np.argsort(chunk_vectors @ q_vec)[::-1][:TOP_K]
    context = "\n\n".join(chunks[i] for i in top)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=PROMPT.format(context=context, question=question),
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return response.text or "Dạ, hiện chưa có câu trả lời. Mong bạn thử lại ạ."


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Nhập câu hỏi của bạn (ví dụ: Thầy dạy thế nào về lòng từ bi?)")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm ý trong bài giảng..."):
            try:
                answer = answer_question(question)
            except Exception as e:
                answer = f"Xin lỗi, đã có lỗi khi gọi Gemini API: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})