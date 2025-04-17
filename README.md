# ML-BCD-Performance-Analysis-of-Classification-Algorithms-on-Wisconsin-Breast-Cancer-Dataset
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

## ویژگی‌های پروژه
- پیش‌پردازش داده‌ها با استفاده از MinMaxScaler
- ارزیابی مدل‌ها با معیارهای دقت، صحت، و فراخوانی
- تجسم‌سازی مقایسه‌ای نتایج مدل‌ها
- تحلیل ماتریس اغتشاش برای مدل‌های برتر
- بررسی اهمیت ویژگی‌ها در مدل‌های درختی
- مقایسه منحنی‌های ROC

## نتایج کلیدی
- تمامی مدل‌ها عملکرد بالای 94% را در دقت آزمون نشان می‌دهند
- Random Forest بالاترین دقت را در میان همه مدل‌ها دارد
- مدل‌های Decision Tree نشانه‌هایی از بیش‌برازش را نشان می‌دهند
- تحلیل اهمیت ویژگی‌ها نشان می‌دهد کدام ویژگی‌ها در پیش‌بینی موثرتر هستند

## نحوه اجرا
1. نصب پکیج‌های مورد نیاز:
```bash
pip install -r requirements.txt
```
2. اجرای نوت‌بوک Jupyter:
```bash
jupyter notebook project1.ipynb
```

## تکنولوژی‌های استفاده شده
- Python 3.13.2
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## ساختار پروژه
```
ML_project/
├── project1.ipynb         # نوت‌بوک اصلی پروژه
├── README.md              # مستندات پروژه
└── requirements.txt       # وابستگی‌های پروژه
``` 
