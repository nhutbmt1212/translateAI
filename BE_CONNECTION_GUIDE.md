# 🚀 Backend Connection Guide for Frontend

This guide explains how to connect your Frontend (React, Vue, etc.) to the local AI Grammar Backend.

---

> [!IMPORTANT]
> **Base URL**: `http://localhost:8000`  
> **Protocol**: HTTP  
> **CORS**: Currently set to `*` (**Allow All**) for easy development.

---

## 🛰️ API Endpoints

### 1. 💬 Chat, Translate & Correct (`/chat`)
This endpoint provides the **direct English version** of your input + linguistic analysis.
- **Tự động lưu**: Lịch sử hội thoại được tự động lưu vào `chat_history.json`.
- **Ngữ cảnh**: AI sẽ nhớ các câu trước đó để trả lời tiếp nối.

#### 📂 Request Body (JSON)
```json
{
  "message": "Your text here...",
  "is_new": false
}
```
- `message`: Câu nhập vào.
- `is_new`: 
  - `true`: Xóa lịch sử cũ và bắt đầu câu mới (Reset context).
  - `false`: Tiếp tục hội thoại dựa trên lịch sử đã lưu.

#### 🌊 Response
- **Type**: `text/plain` (Streaming)

---

### 2. 🔄 Reset Conversation (`/reset`)
Xóa sạch lịch sử hội thoại trong bộ nhớ và file `chat_history.json`.

---

### 3. 🛑 Stop Generation (`/cancel`)
Dừng AI ngay lập tức khi nó đang tạo phản hồi (Dùng cho nút "Stop").

| Method | Endpoint |
| :--- | :--- |
| `POST` | `/cancel` |

#### ✅ Response (JSON)
```json
{ "status": "generation stopped" }
```

---

## 💻 Frontend Implementation (React Example)

> [!TIP]
> Sử dụng **AbortController** để quản lý việc gọi API và kết hợp với nút Stop.

```tsx
const [isGenerating, setIsGenerating] = useState(false);

const handleChat = async (text: string) => {
  setIsGenerating(true);
  try {
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, is_new: false }),
    });

    if (!response.body) return;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      console.log(decoder.decode(value));
    }
  } finally {
    setIsGenerating(false);
  }
};

const stopAI = async () => {
  await fetch('http://localhost:8000/cancel', { method: 'POST' });
};
```

---

## 🛠️ Setup & Troubleshooting

> [!WARNING]
> Đảm bảo server đang chạy: `python grammar_backend.py`

---

## 🏎️ CẤU HÌNH CHO HIỆU NĂNG NHANH NHẤT

### 1. Backend: Tăng Batch Size (Đã tối ưu)
Backend hiện đã được đặt `batch_size = 15`.

### 2. Tối ưu GPU (4-bit quantization)
Hệ thống hiện đang sử dụng **4-bit NF4 quantization**. Đây là mức cấu hình giúp mô hình Google Gemma 2 (9B) chạy mượt mà trên card 12GB mà vẫn giữ được độ thông minh của "Giáo sư".

---

