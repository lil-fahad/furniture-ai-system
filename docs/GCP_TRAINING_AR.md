# دليل تدريب النماذج على Google Cloud (Vertex AI)

هذا الدليل يشرح خطوة بخطوة كيفية تدريب نماذج نظام `furniture-ai-system`
(مصنّف الغرف، ومُجزّئ مخططات الطوابق، ومُرتّب المورّدين) على خدمة **Vertex AI**
باستخدام وحدات GPU، مع تخزين البيانات والنتائج في **Google Cloud Storage (GCS)**.

الدليل مكتوب لمن ليس له خبرة سابقة بـ Google Cloud، وكل الأوامر تعمل من
**Cloud Shell** مباشرة دون تثبيت أي شيء على جهازك.

المشروع المستخدم في الأمثلة: **`round-office-505007-q4`** — استبدله بمعرّف
مشروعك إن كان مختلفًا.

---

## 1. المتطلبات

| المتطلب | التفاصيل |
|---|---|
| حساب Google Cloud | مع **تفعيل الفوترة (Billing)** على المشروع — بدونها لن تعمل أي خدمة |
| مشروع GCP | مثل `round-office-505007-q4` مع صلاحية **Owner** أو **Editor** عليه |
| حصة GPU | حصة `NVIDIA_L4` في المنطقة `us-central1` (انظر قسم استكشاف الأخطاء) |
| Cloud Shell | يُفتح من المتصفح — لا حاجة لتثبيت أي أدوات محليًا |
| نسخة من المستودع | استنساخ (clone) لهذا المستودع داخل Cloud Shell |

> **ملاحظة:** كل شيء يعمل بدون أي مفاتيح أو أسرار مخزّنة في الكود. المصادقة
> تتم عبر حسابك في Cloud Shell تلقائيًا.

---

## 2. فتح Cloud Shell وتجهيز المستودع

1. افتح [console.cloud.google.com](https://console.cloud.google.com) واختر
   المشروع `round-office-505007-q4` من القائمة العلوية.
2. اضغط أيقونة **Cloud Shell** (‎`>_`‎) أعلى يمين الصفحة، وانتظر ثواني حتى
   تفتح الطرفية أسفل الشاشة.
3. تأكد أن المشروع مضبوط:

   ```bash
   gcloud config set project round-office-505007-q4
   ```

4. استنسخ المستودع وادخل إليه:

   ```bash
   git clone <رابط-المستودع>.git
   cd furniture-ai-system
   ```

---

## 3. التشغيل بأمر واحد

من جذر المستودع نفّذ:

```bash
scripts/gcp_bootstrap.sh --project round-office-505007-q4 --region us-central1
```

هذا كل شيء. السكربت آمن لإعادة التشغيل (idempotent): إذا توقف في المنتصف لأي
سبب، أعد تنفيذ نفس الأمر وسيكمل من حيث توقف دون تكرار ما تم إنجازه.

### خيارات إضافية

| الخيار | المعنى |
|---|---|
| `--bucket B` | استخدام اسم حاوية (bucket) مختلف عن الافتراضي |
| `--skip-data` | تخطّي رفع البيانات (إذا رُفعت سابقًا) |
| `--skip-image` | تخطّي بناء صورة التدريب (إذا بُنيت سابقًا) |
| `--skip-jobs` | تجهيز كل شيء دون إطلاق مهام التدريب |

مثال لإعادة إطلاق المهام فقط بعد نجاح التجهيز:

```bash
scripts/gcp_bootstrap.sh --project round-office-505007-q4 --skip-data --skip-image
```

---

## 4. ماذا يفعل السكربت بالضبط؟

السكربت يطبع سطرًا يبدأ بـ `==>` قبل كل خطوة. هذه هي الخطوات السبع:

1. **فحص المصادقة والمشروع** — يتأكد أنك سجّلت الدخول (`gcloud auth`) وأن
   لديك وصولًا للمشروع. إن فشل، نفّذ `gcloud auth login`.
2. **تفعيل واجهات API المطلوبة** — يفعّل أربع خدمات:
   - `aiplatform.googleapis.com` (Vertex AI للتدريب)
   - `storage.googleapis.com` (التخزين GCS)
   - `cloudbuild.googleapis.com` (بناء صورة الحاوية)
   - `artifactregistry.googleapis.com` (مستودع صور الحاويات)

   التفعيل عملية لمرة واحدة ومجاني.
3. **إنشاء مستودع Artifact Registry** باسم `furniture-ai` لتخزين صورة
   التدريب. إن كان موجودًا يتخطاه.
4. **إنشاء حاوية GCS** باسم `gs://furniture-ai-training-round-office-505007-q4`
   لتخزين البيانات والنتائج. إن كانت موجودة يتخطاها.
5. **بناء صورة التدريب** عبر Cloud Build من الملف `cloud/Dockerfile.training`
   (صورة PyTorch مع GPU جاهزة من Vertex AI يُضاف إليها كود المستودع) ورفعها
   إلى Artifact Registry. تستغرق عادة 10–20 دقيقة في أول مرة.
6. **رفع البيانات (staging)** — ينفّذ
   `python -m training.data_ingest.stage_all --bucket <BUCKET>` الذي يجهّز
   مجموعات البيانات محليًا ثم يرفعها إلى `gs://<BUCKET>/datasets/` وفق
   البنية:
   - `datasets/rooms/images/<الفئة>/` — صور مصنّف الغرف
   - `datasets/plans/images/` و `datasets/plans/masks/` — أزواج الصور والأقنعة
   - `datasets/catalog/suppliers_master.csv.gz` — بيانات المورّدين
7. **إطلاق مهام التدريب** — ينفّذ
   `python -m cloud.vertex_jobs --task all ...` الذي يرسل ثلاث مهام
   Custom Training Job إلى Vertex AI (غرف، تجزئة، ترتيب مورّدين)، كل واحدة على
   جهاز `g2-standard-4` مع بطاقة **NVIDIA L4** وبسعر Spot المخفَّض.

### ماذا يحدث داخل كل مهمة؟

الحاوية تبدأ بالأمر `python -m training.cloud_entry --task <المهمة>` مع متغيرات
البيئة `GCP_PROJECT` و `GCP_REGION` و `GCS_BUCKET` و `RUN_ID`. المهمة:

1. تنزّل مجموعة البيانات من `gs://<BUCKET>/datasets/` إلى `/tmp/fai_data/`.
2. تشغّل سكربت التدريب الموجود أصلًا في المستودع (نفس كود التدريب المحلي).
3. تكتب نقاط التفتيش (checkpoints) وملف `metrics.json`.
4. ترفع كل المخرجات إلى `gs://<BUCKET>/runs/<RUN_ID>/`.

---

## 5. مراقبة التدريب

### من الطرفية

عرض قائمة المهام وحالتها:

```bash
gcloud ai custom-jobs list \
  --region us-central1 \
  --project round-office-505007-q4
```

متابعة سجلات (logs) مهمة معيّنة مباشرةً (انسخ معرّف المهمة من القائمة أعلاه):

```bash
gcloud ai custom-jobs stream-logs <JOB_ID> \
  --region us-central1 \
  --project round-office-505007-q4
```

### من واجهة الويب

افتح **Vertex AI → Training** في Cloud Console:
<https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=round-office-505007-q4>

سترى كل مهمة مع حالتها: `JOB_STATE_PENDING` ثم `JOB_STATE_RUNNING` ثم
`JOB_STATE_SUCCEEDED` عند النجاح.

---

## 6. سحب النماذج المدرّبة والنتائج

كل مهمة ترفع مخرجاتها إلى `gs://<BUCKET>/runs/<RUN_ID>/` ويشمل ذلك:

- `checkpoints/` — ملفات النموذج المدرّب (`.pt` / `.json`)
- `metrics.json` — مقاييس التقييم النهائية وحالة المهمة (`succeeded`/`failed`)
- `logs.txt` — سجل التشغيل

لتنزيل كل النتائج إلى جهازك (أو Cloud Shell):

```bash
gcloud storage cp -r \
  gs://furniture-ai-training-round-office-505007-q4/runs .
```

أو لتنزيل نتائج مهمة واحدة فقط:

```bash
gcloud storage cp -r \
  gs://furniture-ai-training-round-office-505007-q4/runs/<RUN_ID> .
```

---

## 7. التكلفة التقريبية

الأرقام تقريبية لمنطقة `us-central1` وقد تتغير مع الأسعار الرسمية:

| البند | التكلفة التقريبية |
|---|---|
| مهمة واحدة (g2-standard-4 + L4 **بسعر Spot**) | **0.4–0.8 دولار/ساعة** |
| نفس المهمة بدون Spot (on-demand) | 1.0–1.3 دولار/ساعة |
| التدريبات الثلاثة كاملة (إعدادات افتراضية) | عادة **أقل من 5 دولارات** بالكامل |
| تخزين البيانات والنتائج (بضعة غيغابايت) | سنتات شهريًا |
| بناء الصورة (Cloud Build) | ضمن الحصة المجانية غالبًا |

نصائح لتقليل التكلفة:

- أبقِ خيار Spot مفعّلًا (هو الافتراضي في `cloud/config.yaml`: `spot: true`).
- احذف النتائج القديمة من `gs://<BUCKET>/runs/` عند عدم الحاجة.
- أنهِ المهام اليدوية المتروكة من صفحة Vertex AI إن ألغيت التجربة.

---

## 8. استكشاف الأخطاء الشائعة

### أ) خطأ حصة GPU ‏(Quota exceeded)

رسالة مثل: `Quota 'NVIDIA_L4_GPUS' exceeded` أو `CustomModelTrainingJobs ... quota`.

- **السبب:** الحسابات الجديدة تحصل على حصة GPU صغيرة أو صفرية.
- **الحل:**
  1. افتح **IAM & Admin → Quotas** في Cloud Console.
  2. ابحث عن `aiplatform.googleapis.com/custom_model_training_nvidia_l4_gpus`
     في منطقة `us-central1`.
  3. اطلب زيادة الحصة إلى 1 على الأقل (الموافقة تستغرق من دقائق إلى يومين).
  4. كحل مؤقت، جرّب منطقة أخرى فيها حصة متاحة عبر `--region`.

### ب) أخطاء الصلاحيات ‏(IAM / Permission denied)

رسالة مثل: `Permission 'aiplatform.customJobs.create' denied`.

- تأكد أن حسابك يملك دور **Editor** أو **Owner** على المشروع (أو أدوار
  `Vertex AI User` + `Storage Admin` + `Artifact Registry Administrator`).
- إذا ظهر خطأ بخصوص حساب الخدمة (service account) أثناء بناء الصورة أو تشغيل
  المهمة، فعادةً يُحل بتفعيل الـ APIs أولًا (الخطوة 2 في السكربت) ثم إعادة
  المحاولة بعد دقيقة، لأن Google تنشئ حسابات الخدمة المطلوبة تلقائيًا عند أول
  استخدام.

### ج) الفوترة غير مفعّلة ‏(Billing disabled)

رسالة مثل: `Billing account for project ... is not enabled`.

- افتح **Billing** في Cloud Console واربط حساب فوترة بالمشروع ثم أعد تشغيل
  السكربت. لا يمكن تفعيل الـ APIs أو تشغيل المهام بدون فوترة.

### د) فشل بناء الصورة في Cloud Build

- تأكد أنك تنفّذ السكربت من **جذر المستودع** (المسار `-f cloud/Dockerfile.training`
  نسبي لجذر المستودع).
- راجع سجل البناء: `gcloud builds list --project round-office-505007-q4` ثم
  `gcloud builds log <BUILD_ID>`.

### هـ) المهمة تبدأ ثم تفشل فورًا

- افحص سجلات المهمة بـ `stream-logs` (انظر قسم المراقبة).
- تأكد أن البيانات رُفعت فعلًا:
  `gcloud storage ls gs://furniture-ai-training-round-office-505007-q4/datasets/`
  — إن كانت فارغة أعد تشغيل السكربت **بدون** `--skip-data`.

### و) إلغاء مهمة عالقة

```bash
gcloud ai custom-jobs cancel <JOB_ID> \
  --region us-central1 \
  --project round-office-505007-q4
```

---

## 9. ملخص سريع (للمتمرسين)

```bash
gcloud config set project round-office-505007-q4
git clone <رابط-المستودع>.git && cd furniture-ai-system
scripts/gcp_bootstrap.sh --project round-office-505007-q4 --region us-central1
gcloud ai custom-jobs list --region us-central1          # المراقبة
gcloud storage cp -r gs://furniture-ai-training-round-office-505007-q4/runs .  # النتائج
```

مرجع إنجليزي مختصر: `cloud/README.md`.
