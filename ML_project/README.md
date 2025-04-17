# مقایسه الگوریتم‌های یادگیری ماشین برای تشخیص سرطان سینه

## توضیحات پروژه
این پروژه به مقایسه عملکرد الگوریتم‌های مختلف یادگیری ماشین برای پیش‌بینی سرطان سینه با استفاده از مجموعه داده استاندارد Breast Cancer Wisconsin می‌پردازد. هدف اصلی این پروژه، تعیین مناسب‌ترین الگوریتم برای این دسته‌بندی دوتایی (خوش‌خیم یا بدخیم) می‌باشد.

## الگوریتم‌های استفاده شده
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- Gradient Boosting

## ویژگی‌های پروژه
- پیش‌پردازش داده‌ها با استفاده از MinMaxScaler
- اعتبارسنجی متقاطع (K-fold Cross-Validation)
- ارزیابی مدل‌ها با معیارهای دقت، صحت، و فراخوانی
- تجسم‌سازی مقایسه‌ای نتایج مدل‌ها
- تحلیل ماتریس اغتشاش برای مدل‌های برتر
- بررسی اهمیت ویژگی‌ها در مدل‌های درختی
- مقایسه منحنی‌های ROC
- تحلیل همبستگی بین ویژگی‌ها

## نتایج کلیدی
- SVM با دقت 98.24% بهترین عملکرد را در اعتبارسنجی متقاطع نشان می‌دهد
- Logistic Regression و KNN هر دو با دقت 95.82% در رتبه دوم قرار دارند
- Random Forest با دقت 95.61% بهترین عملکرد را در داده‌های آزمون نشان می‌دهد
- Decision Tree با دقت 90.99% ضعیف‌ترین عملکرد را در میان مدل‌ها دارد
- ماتریس اغتشاش Random Forest نشان می‌دهد تنها 5 مورد از 114 نمونه به اشتباه طبقه‌بندی شده‌اند
- تحلیل اهمیت ویژگی‌ها نشان می‌دهد ویژگی‌های مرتبط با یکنواختی سلول‌ها و اندازه هسته بیشترین تأثیر را در پیش‌بینی دارند

## نحوه اجرا
1. نصب پکیج‌های مورد نیاز:
```bash
pip install -r requirements.txt
```
2. اجرای نوت‌بوک Jupyter:
```bash
jupyter notebook project1.ipynb
```
3. اجرای نوت‌بوک تحلیل‌های اضافی:
```bash
jupyter notebook Additional_Analysis.ipynb
```
4. اجرای اسکریپت مقایسه مدل‌ها:
```bash
python model_comparison.py
```
5. اجرای اسکریپت تحلیل ویژگی‌ها:
```bash
python feature_analysis.py
```

## نتایج اعتبارسنجی متقاطع (K=5)
- Support Vector Machine: دقت 98.24% (±0.88%)
- Logistic Regression: دقت 95.82% (±1.62%)
- K-Nearest Neighbors: دقت 95.82% (±2.13%)
- Gradient Boosting: دقت 95.38% (±1.28%)
- Random Forest: دقت 95.16% (±1.32%)
- Naive Bayes: دقت 93.41% (±2.09%)
- Decision Tree: دقت 90.99% (±2.13%)

## تکنولوژی‌های استفاده شده
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyter
- StandardScaler و MinMaxScaler برای نرمال‌سازی داده‌ها

## ساختار پروژه
```
ML_project/
├── project1.ipynb             # نوت‌بوک اصلی پروژه
├── model_comparison.py        # اسکریپت مقایسه جامع مدل‌ها
├── feature_analysis.py        # اسکریپت تحلیل ویژگی‌ها
├── Additional_Analysis.ipynb  # نوت‌بوک تحلیل‌های اضافی
├── Model_Analysis.md          # گزارش تحلیلی مدل‌ها
├── README.md                  # مستندات پروژه
└── requirements.txt           # وابستگی‌های پروژه
``` 