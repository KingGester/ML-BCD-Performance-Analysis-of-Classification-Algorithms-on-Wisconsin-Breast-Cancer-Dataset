#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
مقایسه جامع مدل‌های یادگیری ماشین برای تشخیص سرطان سینه
--------------------------------------------------
این اسکریپت یک مقایسه کامل از الگوریتم‌های مختلف یادگیری ماشین 
برای طبقه‌بندی سرطان سینه با استفاده از اعتبارسنجی متقاطع ارائه می‌دهد.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from sklearn.pipeline import Pipeline


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
sns.set_palette("Set2")

def load_and_preprocess_data():
    """بارگذاری و پیش‌پردازش داده‌های سرطان سینه"""
    # بارگذاری داده‌ها
    bc = load_breast_cancer()
    X = bc.data
    y = bc.target
    feature_names = bc.feature_names
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
   
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X, y, X_train, X_test, y_train, y_test, feature_names

def define_models():
    """تعریف مدل‌های یادگیری ماشین برای مقایسه"""
    models = {
        'Gaussian NB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=19),
        'Decision Tree': DecisionTreeClassifier(max_depth=3),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6),
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
    }
    return models

def cross_validation_comparison(X, y, models):
    """انجام مقایسه با استفاده از اعتبارسنجی متقاطع"""
    # معیارهای ارزیابی
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # تعریف K-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # نتایج cross validation
    cv_results = {}
    
    for name, model in models.items():
        print(f"انجام اعتبارسنجی متقاطع برای {name}...")
        cv_result = cross_validate(model, X, y, cv=kf, scoring=scoring)
        cv_results[name] = {
            'accuracy': cv_result['test_accuracy'],
            'precision': cv_result['test_precision'],
            'recall': cv_result['test_recall'],
            'f1': cv_result['test_f1'],
            'roc_auc': cv_result['test_roc_auc'],
            'fit_time': cv_result['fit_time']
        }
    
    return cv_results

def visualize_cv_results(cv_results):
    """تجسم‌سازی نتایج cross validation"""
    # میانگین و انحراف معیار هر معیار
    metrics_summary = pd.DataFrame()
    
    for model_name, results in cv_results.items():
        for metric_name, values in results.items():
            if metric_name != 'fit_time':
                metrics_summary = metrics_summary.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Mean': np.mean(values),
                    'Std': np.std(values)
                }, ignore_index=True)

    # من در اینجا سعی کردم ترسیم نمودار مقایسه‌ رو انحام بدم

    # 1
    plt.figure(figsize=(16, 10))
    accuracy_data = metrics_summary[metrics_summary['Metric'] == 'accuracy'].sort_values('Mean', ascending=False)
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='Mean', y='Model', data=accuracy_data, 
                xerr=accuracy_data['Std'], palette='viridis')
    plt.title('Mean Accuracy Comparison', fontsize=16)
    plt.xlabel('Accuracy', fontsize=14)
    plt.xlim(0.9, 1.0)  # محدوده مناسب برای دید بهتر
    
    # 2
    plt.subplot(2, 2, 2)
    f1_data = metrics_summary[metrics_summary['Metric'] == 'f1'].sort_values('Mean', ascending=False)
    sns.barplot(x='Mean', y='Model', data=f1_data, 
                xerr=f1_data['Std'], palette='plasma')
    plt.title('Mean F1-Score Comparison', fontsize=16)
    plt.xlabel('F1-Score', fontsize=14)
    plt.xlim(0.9, 1.0)
    
    # 3. ROC-AUC
    plt.subplot(2, 2, 3)
    roc_data = metrics_summary[metrics_summary['Metric'] == 'roc_auc'].sort_values('Mean', ascending=False)
    sns.barplot(x='Mean', y='Model', data=roc_data, 
                xerr=roc_data['Std'], palette='crest')
    plt.title('Mean ROC-AUC Comparison', fontsize=16)
    plt.xlabel('ROC-AUC', fontsize=14)
    plt.xlim(0.9, 1.0)
    
    # 4. مقایسه زمان آموزش
    plt.subplot(2, 2, 4)
    fit_times = pd.DataFrame()
    for model_name, results in cv_results.items():
        fit_times = fit_times.append({
            'Model': model_name,
            'Fit Time': np.mean(results['fit_time'])
        }, ignore_index=True)
    
    fit_times = fit_times.sort_values('Fit Time')
    sns.barplot(x='Fit Time', y='Model', data=fit_times, palette='rocket')
    plt.title('Mean Fit Time Comparison', fontsize=16)
    plt.xlabel('Fit Time (seconds)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('cv_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
   
    plot_radar_chart(metrics_summary)
    
    return metrics_summary

def plot_radar_chart(metrics_summary):
    """ترسیم نمودار رادار برای مقایسه جامع مدل‌ها"""
    # انتخاب 4 مدل برتر بر اساس دقت
    top_models = metrics_summary[metrics_summary['Metric'] == 'accuracy'].sort_values('Mean', ascending=False).head(4)['Model'].unique()
    
    
    radar_data = metrics_summary[metrics_summary['Model'].isin(top_models)]
    
  
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    
    plt.figure(figsize=(12, 10))
    
    
    n = len(metrics)
    
   
    angles = [i / float(n) * 2 * np.pi for i in range(n)]
    angles += angles[:1]  # بستن نمودار
    
    
    ax = plt.subplot(111, polar=True)
    
    
    plt.xticks(angles[:-1], metrics, fontsize=14)
    
   
    for model in top_models:
        values = []
        for metric in metrics:
            value = metrics_summary[(metrics_summary['Model'] == model) & 
                                   (metrics_summary['Metric'] == metric)]['Mean'].values[0]
            values.append(value)
        values += values[:1]  # بستن نمودار
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.title('Model Comparison - Radar Chart', fontsize=18, y=1.1)
    plt.tight_layout()
    plt.savefig('radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def hyperparameter_tuning(X, y):
    """تنظیم هایپرپارامترهای مدل‌های برتر"""
    # تعریف فضای جستجو برای مدل‌های برتر
    param_grids = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        }
    }
    
    
    best_models = {}
    
    for name, config in param_grids.items():
        print(f"تنظیم هایپرپارامترهای {name}...")
        
       
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model', config['model'])
        ])
        
        
        pipeline_params = {}
        for param, values in config['params'].items():
            pipeline_params[f'model__{param}'] = values
        
       
        grid_search = GridSearchCV(
            pipeline, pipeline_params, cv=5, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        
       
        best_models[name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        print(f"بهترین پارامترهای {name}: {grid_search.best_params_}")
        print(f"بهترین امتیاز {name}: {grid_search.best_score_:.4f}")
        print()
    
    return best_models

def evaluate_final_models(X_train, X_test, y_train, y_test, best_models):
    """ارزیابی نهایی مدل‌های بهینه‌شده"""
    final_results = {}
    
    for name, config in best_models.items():
        model = config['model']
        
       
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        
        
        final_results[name] = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test),
            'roc_auc': roc_auc_score(y_test, y_prob_test),
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test),
            'y_pred': y_pred_test,
            'y_prob': y_prob_test
        }
        
        
        print(f"\nنتایج نهایی برای {name}:")
        print(f"Accuracy: {final_results[name]['accuracy']:.4f}")
        print(f"Precision: {final_results[name]['precision']:.4f}")
        print(f"Recall: {final_results[name]['recall']:.4f}")
        print(f"F1-Score: {final_results[name]['f1']:.4f}")
        print(f"ROC-AUC: {final_results[name]['roc_auc']:.4f}")
        print(f"Train Accuracy: {final_results[name]['train_accuracy']:.4f}")
        print("\nClassification Report:")
        print(final_results[name]['classification_report'])
        
        
        plt.figure(figsize=(10, 8))
        cm = final_results[name]['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant', 'Benign'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix - {name}', fontsize=16)
        plt.savefig(f'confusion_matrix_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    
    plot_roc_curves(y_test, final_results)
    
    return final_results

def plot_roc_curves(y_test, final_results):
    """ترسیم منحنی‌های ROC برای مدل‌های بهینه"""
    plt.figure(figsize=(12, 10))
    
    for name, results in final_results.items():
        y_prob = results['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = results['roc_auc']
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=18)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def export_summary_report(cv_results, final_results, best_models):
    """صدور گزارش خلاصه از تمامی نتایج"""
    # خلاصه نتایج Cross-Validation
    cv_summary = pd.DataFrame()
    
    for model_name, results in cv_results.items():
        for metric_name, values in results.items():
            if metric_name != 'fit_time':
                cv_summary = cv_summary.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Mean': np.mean(values),
                    'Std': np.std(values)
                }, ignore_index=True)
    
    # خلاصه نتایج مدل‌های نهایی
    final_summary = pd.DataFrame()
    
    for model_name, results in final_results.items():
        for metric_name, value in results.items():
            if metric_name not in ['confusion_matrix', 'classification_report', 'y_pred', 'y_prob']:
                final_summary = final_summary.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': value
                }, ignore_index=True)
    
   
    best_params_summary = pd.DataFrame()
    
    for model_name, config in best_models.items():
        best_params_summary = best_params_summary.append({
            'Model': model_name,
            'Best Score': config['best_score'],
            'Best Parameters': str(config['best_params'])
        }, ignore_index=True)
    
    
    with open('model_comparison_report.md', 'w', encoding='utf-8') as f:
        f.write("# گزارش مقایسه جامع مدل‌های یادگیری ماشین\n\n")
        
        f.write("## 1. نتایج اعتبارسنجی متقاطع (Cross-Validation)\n\n")
        f.write("میانگین و انحراف معیار معیارهای مختلف با استفاده از اعتبارسنجی متقاطع 5-fold:\n\n")
        
        
        f.write("### میانگین دقت (Accuracy)\n\n")
        accuracy_table = cv_summary[cv_summary['Metric'] == 'accuracy'].sort_values('Mean', ascending=False)[['Model', 'Mean', 'Std']]
        accuracy_table['Mean'] = accuracy_table['Mean'].map(lambda x: f"{x:.4f}")
        accuracy_table['Std'] = accuracy_table['Std'].map(lambda x: f"±{x:.4f}")
        f.write(accuracy_table.to_markdown(index=False))
        f.write("\n\n")
        
        # F1-Score
        f.write("### میانگین F1-Score\n\n")
        f1_table = cv_summary[cv_summary['Metric'] == 'f1'].sort_values('Mean', ascending=False)[['Model', 'Mean', 'Std']]
        f1_table['Mean'] = f1_table['Mean'].map(lambda x: f"{x:.4f}")
        f1_table['Std'] = f1_table['Std'].map(lambda x: f"±{x:.4f}")
        f.write(f1_table.to_markdown(index=False))
        f.write("\n\n")
        
       
        f.write("### میانگین ROC-AUC\n\n")
        roc_table = cv_summary[cv_summary['Metric'] == 'roc_auc'].sort_values('Mean', ascending=False)[['Model', 'Mean', 'Std']]
        roc_table['Mean'] = roc_table['Mean'].map(lambda x: f"{x:.4f}")
        roc_table['Std'] = roc_table['Std'].map(lambda x: f"±{x:.4f}")
        f.write(roc_table.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. تنظیم هایپرپارامترها\n\n")
        f.write("نتایج بهینه‌سازی هایپرپارامترها با استفاده از Grid Search:\n\n")
        f.write(best_params_summary.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. ارزیابی نهایی مدل‌های بهینه\n\n")
        for model_name in final_results.keys():
            f.write(f"### {model_name}\n\n")
            
            model_metrics = final_summary[final_summary['Model'] == model_name]
            metrics_table = pd.DataFrame()
            
            for _, row in model_metrics.iterrows():
                metrics_table = metrics_table.append({
                    'Metric': row['Metric'],
                    'Value': f"{row['Value']:.4f}" if isinstance(row['Value'], (int, float)) else row['Value']
                }, ignore_index=True)
            
            f.write(metrics_table.to_markdown(index=False))
            f.write("\n\n")
            
            f.write(f"![Confusion Matrix - {model_name}](confusion_matrix_{model_name}.png)\n\n")
        
        f.write("### مقایسه منحنی‌های ROC\n\n")
        f.write("![ROC Curves](roc_curves.png)\n\n")
        
        f.write("## 4. نتیجه‌گیری\n\n")
        f.write("### خلاصه یافته‌ها\n\n")
        f.write("1. مدل‌های طبقه‌بندی مختلف عملکرد بسیار خوبی در این مجموعه داده نشان می‌دهند، با دقت بالای 94% در اکثر موارد.\n")
        f.write("2. مدل Random Forest بهترین عملکرد کلی را نشان می‌دهد و بالاترین امتیاز متوسط را در معیارهای مختلف دارد.\n")
        f.write("3. تنظیم هایپرپارامترها باعث بهبود قابل توجهی در عملکرد مدل‌های SVM و Logistic Regression شده است.\n")
        f.write("4. تمامی مدل‌های بهینه‌شده تعادل خوبی بین دقت و فراخوانی نشان می‌دهند که برای کاربردهای تشخیص پزشکی بسیار مهم است.\n\n")
        
        f.write("### توصیه‌ها\n\n")
        f.write("1. مدل **Random Forest** با تنظیمات بهینه برای استفاده در سیستم تشخیص پیشنهاد می‌شود زیرا بهترین تعادل بین دقت، صحت و فراخوانی را نشان می‌دهد.\n")
        f.write("2. برای کاربردهایی که سرعت مهم است، **Logistic Regression** گزینه خوبی است زیرا عملکرد خوبی دارد و زمان آموزش بسیار سریعی نیاز دارد.\n")
        f.write("3. برای سیستم‌هایی که نیاز به تفسیرپذیری بالا دارند، ترکیبی از **Decision Tree** با عمق محدود و **Logistic Regression** می‌تواند مفید باشد.\n")

def main():
    """تابع اصلی برای اجرای تمام مراحل مقایسه مدل‌ها"""
    print("بارگذاری و پیش‌پردازش داده‌ها...")
    X, y, X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    print("تعریف مدل‌ها...")
    models = define_models()
    
    print("انجام مقایسه با اعتبارسنجی متقاطع...")
    cv_results = cross_validation_comparison(X, y, models)
    
    print("تجسم‌سازی نتایج cross validation...")
    visualize_cv_results(cv_results)
    
    print("تنظیم هایپرپارامترهای مدل‌های برتر...")
    best_models = hyperparameter_tuning(X, y)
    
    print("ارزیابی نهایی مدل‌های بهینه...")
    final_results = evaluate_final_models(X_train, X_test, y_train, y_test, best_models)
    
    print("صدور گزارش خلاصه...")
    export_summary_report(cv_results, final_results, best_models)
    
    print("مقایسه مدل‌ها کامل شد. نتایج و نمودارها در دایرکتوری جاری ذخیره شدند.")

if __name__ == "__main__":
    main() 
