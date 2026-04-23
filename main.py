from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
import os
import base64
import asyncio

# تحميل المتغيرات من ملف .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Wound AI running"}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        print(f"📥 وصل طلب جديد: {file.filename}")

        # قراءة الصورة
        image_bytes = await file.read()

        # التحقق من حجم الصورة (2 ميجابايت كحد أقصى)
        if len(image_bytes) > 2 * 1024 * 1024:
            return JSONResponse({"error": "الصورة كبيرة جدًا، يرجى اختيار صورة أصغر"}, status_code=400)

        # تحويل الصورة إلى Base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # --- حل مشكلة MIME TYPE (تعديل جوهري) ---
        # OpenAI تقبل فقط: image/jpeg, image/png, image/webp, image/gif
        # بعض الجوالات ترسل image/jpg (بدون e) وهذا يسبب الخطأ الذي ظهر عندك
        content_type = file.content_type.lower() if file.content_type else ""
        
        if "png" in content_type:
            final_mime = "image/png"
        elif "webp" in content_type:
            final_mime = "image/webp"
        else:
            final_mime = "image/jpeg" # نعتمد jpeg كافتراضي لأي نوع آخر (مثل jpg)
        # ---------------------------------------

        prompt = """
أنت خبير تشخيص طبي متخصص في تحليل إصابات الجلد (جروح، حروق، كدمات، خدوش) من الصور.

المهام والقيود الصارمة:

1. تحليل المحتوى أولاً (أولوية قصوى):
    - افحص الصورة بدقة وحدد هل تحتوي على جلد بشري أم لا.
    
    - إذا لم تحتوي على جلد بشري:
        اجعل "نوع_الإصابة": "ليست إصابة جلدية"
        ثم أضف وصف دقيق لما تراه (مثال: جبس طبي، ضماد، ملابس، جسم صناعي...)
        مثال صحيح: "ليست إصابة جلدية - الملاحظ وجود جبس طبي"

    - إذا كان الجلد سليم تماماً:
        "نوع_الإصابة": "جلد سليم"

    - إذا كانت الصورة غير واضحة أو مظلمة:
        "نوع_الإصابة": "غير واضح"

--------------------------------------------------

2. التشخيص الطبي للإصابات الجلدية:

- جرح:
    وجود قطع واضح أو نزيف

- كدمة:
    تغير لون الجلد (أزرق / بنفسجي / أخضر ) بدون قطع

- خدش:
    إصابة سطحية خفيفة بدون نزيف عميق

- حروق (تشخيص دقيق ومتدرج - مهم جدا ):
    لا تكتب "حرق" فقط — يجب تحديد الدرجة بناءً على العلامات التالية:

🔥 حرق درجة أولى:
  - احمرار فقط
  - بدون فقاعات
  - الجلد سليم
  - لا يوجد تلف عميق
  → غالباً بسيطة
  
  🔥 حرق درجة ثانية (يجب التفريق بدقة):
  
    🟢 درجة ثانية سطحية:
      - فقاعات قليلة أو صغيرة (حتى فقاعة واحدة)
      - لون أحمر أو وردي
      - الجلد رطب
      - مساحة صغيرة
      - ألم واضح
      → غالباً بسيطة

  🔴 درجة ثانية عميقة:
      - فقاعات كثيرة أو كبيرة
      - لون مائل للأبيض أو باهت
      - الجلد أقل رطوبة أو شبه جاف
      - مساحة متوسطة إلى كبيرة
      - ألم أقل نسبياً
      → قد تكون خطيرة
  
  🔥 حرق درجة ثالثة:
  - لون أبيض أو أسود
  - الجلد متفحم أو جاف جداً
  - تلف عميق
  - قد لا يظهر ألم
  → خطيرة دائماً
  
  ⚠️ قواعد مهمة جداً:
  
  - وجود فقاعة واحدة فقط لا يعني أن الحرق خطير
  - لا تفترض الخطورة فقط بسبب وجود فقاعات
  - إذا لم يمكن التمييز بين سطحية وعميقة:
    صنّفها "حرق درجة ثانية" بدون افتراض الخطورة مباشرة

---

3. تقييم مستوى الخطورة (دقيق ومتوازن):

قبل تحديد الخطورة، قيّم:

- عدد الفقاعات (قليل / كثير)

- حجم الفقاعات

- لون الجلد

- مساحة الإصابة

- موقع الإصابة

- "بسيطة":
  
  - حرق درجة أولى
  - حرق درجة ثانية سطحية (فقاعات قليلة + مساحة صغيرة + لون أحمر)
  - خدوش أو كدمات خفيفة
  - جروح سطحية

- "خطيرة":
  في الحالات التالية فقط:
  
  - حرق درجة ثالثة
  - حرق درجة ثانية مع:
    • فقاعات كثيرة أو كبيرة
    • لون أبيض أو داكن
    • مساحة واسعة
    • موقع حساس (الوجه، اليد، المفاصل)
  - جرح عميق
  - نزيف غزير
  - إصابة كبيرة أو مفتوحة

---

4. درجة الثقة:

- عالية: الصورة واضحة والعلامات مؤكدة
- متوسطة: بعض الغموض
- منخفضة: الصورة غير واضحة

---

5. التوصية:

- إذا "بسيطة" → إسعافات منزلية
- إذا "خطيرة" → التوجه إلى طبيب مختص

---

تنبيهات مهمة:

- لا تحكم على الخطورة فقط بسبب اللون الأحمر
- لا تخلط بين الحروق والجروح
- لا تبالغ في تصنيف الخطورة
- وجود فقاعة واحدة لا يعني حالة خطيرة
- لا تستخدم التخمين إلا في حالة "ليست إصابة جلدية"
- لا تضف أي شرح خارج JSON
--------------------------------------------------

أجب باللغة العربية فقط وبصيغة JSON حصراً:

{
  "نوع_الإصابة": "جرح | كدمة | خدش | حرق درجة أولى | حرق درجة ثانية | حرق درجة ثالثة | ليست إصابة جلدية - (وصف) | جلد سليم | غير واضح",
  "مستوى_الخطورة": "بسيطة | خطيرة | غير محدد",
  "درجة_الثقة": "عالية | متوسطة | منخفضة",
  "التوصية": "إسعافات منزلية | التوجه إلى طبيب مختص",
  "الإجراءات": [
    "إجراء طبي 1",
    "إجراء طبي 2",
    "إجراء طبي 3"
  ]
}
"""
        # طلب التحليل من OpenAI
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{final_mime};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
        )

        print("✅ اكتمل تحليل الذكاء الاصطناعي بنجاح")

        result_text = response.choices[0].message.content.strip()
        return JSONResponse({"analysis": result_text})

    except asyncio.TimeoutError:
        print("⏰ انتهى الوقت")
        return JSONResponse({"error": "السيرفر تأخر في الرد، حاول مرة أخرى"}, status_code=504)

    except Exception as e:
        # طباعة الخطأ في الكونسول لمعرفة التفاصيل لو حدث فشل
        error_msg = str(e)
        print(f"❌ خطأ: {error_msg}")
        return JSONResponse({"error": f"Error: {error_msg}"}, status_code=500)
    


    # --- الجزء الجديد الخاص بالمرحلة الثالثة (الشات بوت) ---
@app.post("/chat")
async def chat_with_ai(data: dict):
    try:
        user_message = data.get("message")
        analysis_context = data.get("context") 
        chat_history = data.get("history", [])

        system_instruction = f"""
        أنت مساعد طبي ذكي في تطبيق 'المسعف'. 
        سياق حالة المريض بناءً على الصورة المحللة: {analysis_context}.
        أجب على استفسارات المريض بدقة وبناءً على حالته المذكورة في السياق. 
        كن مختصراً، ودوداً، ولا تعطِ نصائح خارج النطاق الطبي للإصابة.
        """

        messages = [{"role": "system", "content": system_instruction}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
        )

        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# التشغيل: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
