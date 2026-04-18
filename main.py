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
أنت خبير تشخيص طبي متخصص في تحليل إصابات الجلد (جروح، حروق، كدمات) من الصور.

المهام والقيود الصارمة:
1. فحص المحتوى والتخمين الذكي (أولوية قصوى): 
    - إذا كانت الصورة لا تحتوي على جلد بشري أو إصابة واضحة، اجعل "نوع_الإصابة": "ليست إصابة جلدية". 
    - يجب أن تضيف تخميناً دقيقاً لما تراه إذا كان شيئاً طبياً غير جلدي. (مثال: "ليست إصابة جلدية - الملاحظ وجود جبس طبي").
    - إذا كانت الصورة لجلد سليم تماماً، اجعل "نوع_الإصابة": "جلد سليم".
    - إذا كانت الصورة غير واضحة أو مظلمة، اجعل "نوع_الإصابة": "غير واضح".

2. التشخيص الدقيق (بدون زيادة كلام):
    - جرح، كدمة، خدش: اكتب المسمى كما هو (مثال: جرح) دون إضافة أوصاف بجانبه في خانة نوع_الإصابة.
    - حرق (قاعدة خاصة): يُمنع كتابة كلمة "حرق" بمفردها. يجب تحديد الدرجة فقط. الصيغة المعتمدة: "حرق درجة أولى" أو "حرق درجة ثانية" أو "حرق درجة ثالثة".

3. معايير تقييم الخطورة (قواعدكِ الصارمة للمصداقية):
    - "بسيطة": إذا كانت سطحية، صغيرة المساحة، نزيفها ضئيل جداً أو متوقف، ويمكن التعامل معها بالإسعافات الأولية (مثل الخدش والكدمة وحرق الدرجة الأولى).
    - "خطيرة": في الحالات التالية فقط: (جرح عميق يكشف ما تحت الجلد، نزيف غزير، حرق من الدرجة الثانية أو الثالثة، وجود أجسام غريبة، أو وجود جبس طبي لكسر).
    
تنبيهات هامة للموديل:
- لا تحكم على الإصابة بأنها "خطيرة" لمجرد رؤية اللون الأحمر؛ ركز على عمق الجرح ومساحته (كما حددتِ في قواعدكِ).
- في الإصابات الجلدية (جرح، كدمة، خدش)، لا تكتب "تخمين"؛ اكتب المسمى الطبي مباشرة. التخمين فقط للأجسام غير الجلدية (كالجبس).

أجب باللغة العربية فقط، وبصيغة JSON حصراً:
{
  "نوع_الإصابة": "المسمى الطبي مباشرة (جرح / كدمة / خدش) أو (حرق درجة ...) أو التخمين (ليست إصابة جلدية - جبس طبي)",
  "مستوى_الخطورة": "بسيطة | خطيرة | غير محدد",
  "درجة_الثقة": "عالية | متوسطة",
  "التوصية": "نصيحة مختصرة جداً (تتضمن 'طبيب مختص' للحالات الخطيرة)",
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

# التشغيل: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
