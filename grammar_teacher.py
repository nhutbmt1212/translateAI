import os
import re

class GrammarTeacher:
    def __init__(self):
        # Path to upgraded Google Gemma 2 9B model
        self.model_path = os.path.join(os.path.dirname(__file__), "gemma-2-9b-it")
        
        print("Đang khởi tạo hệ thống AI (Mô hình Google Gemma 2 9B - Giáo sư AI)...")
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Grammar Teacher using device: {self.device}")
        
        # Lazy loading
        self.tokenizer = None
        self.model = None
        self.history = []
        self.stop_signal = False
        self.history_file = os.path.join(os.path.dirname(__file__), "chat_history.json")
        
        # Load existing history if available
        self._load_history()

    def _load_model(self):
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            print("Đang nạp Giáo sư AI (Gemma 2 9B) vào bộ nhớ (Sử dụng 4-bit quantization để tối ưu VRAM)...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Configure 4-bit quantization for RTX 3060 (12GB)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch.bfloat16 if self.device == "cuda" else self.torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            load_kwargs = {
                "device_map": "auto",
                "quantization_config": quantization_config if self.device == "cuda" else None,
                "torch_dtype": "auto"
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs
            )
            
            # Reset history with system prompt
            self.reset_chat()

    def reset_chat(self):
        self.system_instructions = """Bạn là Chuyên gia Ngữ pháp và Từ vựng Tiếng Anh. Hãy xử lý đầu vào theo quy trình logic sau:
QUY TẮC CỐT LÕI (CRITICAL RULES):
1. NGÔN NGỮ ĐẦU RA: Tuyệt đối chỉ sử dụng Tiếng Việt để giải thích. Không dùng tiếng Anh hay bất kỳ ngôn ngữ nào khác trừ khi đó là câu dịch mẫu hoặc từ chuyên môn cần học.
2. CẤM TUYỆT ĐỐI TIẾNG TRUNG: Không bao giờ được sử dụng tiếng Trung Quốc (hoặc bất kỳ ký tự tượng hình nào khác) trong bất kỳ trường hợp nào.

1. XÁC ĐỊNH LOẠI ĐẦU VÀO (INPUT TYPE):

   TRƯỜNG HỢP A: TỪ ĐƠN (Single Word - Bất kể ngôn ngữ)
   - Dịch nghĩa sang ngôn ngữ còn lại.
   - Giải thích "Usage" (Cách dùng/Mục đích sử dụng của từ) bằng tiếng Việt.
   - Đặt 1 câu ví dụ.

   TRƯỜNG HỢP B: CÂU TIẾNG VIỆT (Vietnamese Sentence)
   - KHÔNG kiểm tra ngữ pháp tiếng Việt.
   - Dịch ngay sang câu Tiếng Anh chuẩn (Natural English).
   - Giải thích cấu trúc bằng tiếng Việt nếu cần.

   TRƯỜNG HỢP C: CÂU TIẾNG ANH (English Sentence) -> KÍCH HOẠT CHẾ ĐỘ CHECK GRAMMAR
   - Bước 1: Kiểm tra ngữ pháp.
   - Bước 2:
     + NẾU SAI (Incorrect):
       * Viết lại câu đúng (Corrected version).
       * Giải thích lỗi sai ngắn gọn BẰNG TIẾNG VIỆT.
     + NẾU ĐÚNG (Correct):
       * Thông báo: "Câu chuẩn ngữ pháp."
       * Dịch câu đó sang Tiếng Việt.

LƯU Ý: Tuyệt đối không giải thích bằng tiếng Anh. Toàn bộ phần phân tích phải là Tiếng Việt thuần túy."""
        self.history = []
        self._save_history()

    def stop_generation(self):
        """Signal the generator to stop as soon as possible."""
        self.stop_signal = True
        print("\n[HỆ THỐNG] Đã gửi tín hiệu dừng tới AI...")

    def _save_history(self):
        import json
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Lỗi khi lưu lịch sử: {e}")

    def _load_history(self):
        import json
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"Đã tải {len(self.history)} tin nhắn từ lịch sử.")
            except Exception as e:
                print(f"Lỗi khi tải lịch sử: {e}")
                self.history = []

    def ask_stream(self, user_input, is_new_sentence=False):
        self._load_model()
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        if is_new_sentence or not self.history:
            self.reset_chat()
            full_prompt = f"{self.system_instructions}\n\nUser Input: {user_input}\n\nƯu tiên kết quả tiếng Anh trước, sau đó phân tích ngôn ngữ bằng tiếng Việt."
            self.history.append({"role": "user", "content": full_prompt})
        else:
            self.history.append({"role": "user", "content": user_input})
        
        text_input = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

        # Initialize the streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generation arguments
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            repetition_penalty=1.1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Run generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = ""
        # Regex to match Chinese characters (CJK Unified Ideographs)
        chinese_re = re.compile(r'[\u4e00-\u9fff]+')
        
        self.stop_signal = False
        for new_text in streamer:
            if self.stop_signal:
                print("\n[HỆ THỐNG] AI đã dừng phản hồi theo yêu cầu.")
                break
            # Filter out Chinese characters from the stream
            filtered_text = chinese_re.sub('', new_text)
            full_response += filtered_text
            yield filtered_text
        
        self.history.append({"role": "assistant", "content": full_response})
        self._save_history()

    def ask(self, user_input, is_new_sentence=False):
        print("\nGiáo viên AI: ", end="", flush=True)
        full_response = ""
        for token in self.ask_stream(user_input, is_new_sentence):
            print(token, end="", flush=True)
            full_response += token
        print()
        return full_response

if __name__ == "__main__":
    teacher = GrammarTeacher()
    
    print("\n" + "="*50)
    print(" CHÀO MỪNG BẠN ĐẾN VỚI LỚP HỌC TIẾNG ANH AI ")
    print("="*50)
    print("- Gõ câu tiếng Anh để được sửa lỗi.")
    print("- Sau khi sửa, bạn có thể hỏi thêm về các lỗi đó.")
    print("- Gõ 'new' để bắt đầu với câu mới.")
    print("- Gõ 'exit' để thoát.")
    print("="*50)

    is_new = True
    while True:
        try:
            user_msg = input("\nBạn: ").strip()
            
            if user_msg.lower() == 'exit':
                print("Tạm biệt!")
                break
            
            if user_msg.lower() == 'new':
                teacher.reset_chat()
                is_new = True
                print("Đã làm mới bộ nhớ. Hãy nhập câu mới!")
                continue
            
            if not user_msg:
                continue

            print("\nAI đang suy nghĩ...")
            teacher.ask(user_msg, is_new_sentence=is_new)
            is_new = False # Following messages will be chat context
            
        except KeyboardInterrupt:
            print("\nĐã ngắt.")
            break
        except Exception as e:
            print(f"\nCó lỗi xảy ra: {e}")
