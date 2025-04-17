#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
تحلیل ویژگی‌های مجموعه داده سرطان سینه
-----------------------------
این اسکریپت تحلیل جامعی از ویژگی‌های مجموعه داده سرطان سینه ارائه می‌دهد
و اهمیت ویژگی‌ها را با استفاده از چند روش مختلف بررسی می‌کند.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# در این جا میخوام تنظیمات نمودارها رو انجام بدم
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
sns.set_palette("Set2")

def load_data():
    """بارگذاری مجموعه داده سرطان سینه"""
    bc = load_breast_cancer()
    X = bc.data
    y = bc.target
    feature_names = bc.feature_names
    
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
    
    return df, X, y, feature_names

def feature_correlation(df, feature_names):
    """تحلیل همبستگی بین ویژگی‌ها"""
   
    corr_matrix = df[feature_names].corr()
    
    # ترسیم نقشه حرارتی همبستگی برای تحلیل 
    plt.figure(figsize=(18, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=18)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # گزارش ویژگی‌های با همبستگی بالا (> 0.9)
    high_corr = pd.DataFrame()
    for i, row in enumerate(corr_matrix.values):
        for j in range(i+1, len(row)):
            if abs(row[j]) > 0.9:
                high_corr = high_corr.append({
                    'Feature 1': feature_names[i],
                    'Feature 2': feature_names[j],
                    'Correlation': row[j]
                }, ignore_index=True)
    
    return high_corr

def feature_distributions(df, feature_names):
    """ترسیم توزیع ویژگی‌ها بر اساس کلاس"""
    # انتخاب 6 ویژگی مهم برای نمایش
    important_features = ['mean radius', 'mean texture', 'mean perimeter', 
                         'mean area', 'mean compactness', 'mean concave points']
    
    plt.figure(figsize=(18, 10))
    for i, feature in enumerate(important_features):
        plt.subplot(2, 3, i+1)
        sns.histplot(data=df, x=feature, hue='diagnosis', kde=True, 
                    element='step', common_norm=False, alpha=0.7)
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def univariate_feature_importance(X, y, feature_names):
    """تحلیل اهمیت تک متغیره ویژگی‌ها با استفاده از ANOVA F-value و اطلاعات متقابل"""
    # ANOVA F-value
    selector_f = SelectKBest(f_classif, k='all')
    selector_f.fit(X, y)
    f_scores = pd.DataFrame({
        'Feature': feature_names,
        'F-Score': selector_f.scores_,
        'P-value': selector_f.pvalues_
    })
    f_scores = f_scores.sort_values('F-Score', ascending=False).reset_index(drop=True)
    
    # Mutual Information
    selector_mi = SelectKBest(mutual_info_classif, k='all')
    selector_mi.fit(X, y)
    mi_scores = pd.DataFrame({
        'Feature': feature_names,
        'MI-Score': selector_mi.scores_
    })
    mi_scores = mi_scores.sort_values('MI-Score', ascending=False).reset_index(drop=True)
    
    # در این جا من ترسیم های نتایج رو ایجاد کردم
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 1, 1)
    sns.barplot(x='F-Score', y='Feature', data=f_scores.head(15), palette='viridis')
    plt.title('Top 15 Features by ANOVA F-Score', fontsize=16)
    plt.xlabel('F-Score')
    plt.ylabel('Feature')
    
    plt.subplot(2, 1, 2)
    sns.barplot(x='MI-Score', y='Feature', data=mi_scores.head(15), palette='plasma')
    plt.title('Top 15 Features by Mutual Information', fontsize=16)
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Feature')
    
    plt.tight_layout()
    plt.savefig('univariate_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return f_scores, mi_scores

def model_based_feature_importance(X, y, feature_names):
    """تحلیل اهمیت ویژگی‌ها بر اساس مدل‌های Random Forest و Logistic Regression"""
    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    })
    rf_importances = rf_importances.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Logistic Regression Coefficients
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_importances = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': np.abs(lr.coef_[0])
    })
    lr_importances = lr_importances.sort_values('Coefficient', ascending=False).reset_index(drop=True)
    
    # Recursive Feature Elimination
    rfe = RFE(estimator=lr, n_features_to_select=15)
    rfe.fit(X_train_scaled, y_train)
    rfe_results = pd.DataFrame({
        'Feature': feature_names,
        'Selected': rfe.support_,
        'Rank': rfe.ranking_
    })
    rfe_results = rfe_results.sort_values('Rank').reset_index(drop=True)
    
 
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 1, 1)
    sns.barplot(x='Importance', y='Feature', data=rf_importances.head(15), palette='viridis')
    plt.title('Top 15 Features by Random Forest Importance', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    plt.subplot(2, 1, 2)
    sns.barplot(x='Coefficient', y='Feature', data=lr_importances.head(15), palette='plasma')
    plt.title('Top 15 Features by Logistic Regression Coefficient Magnitude', fontsize=16)
    plt.xlabel('|Coefficient|')
    plt.ylabel('Feature')
    
    plt.tight_layout()
    plt.savefig('model_based_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return rf_importances, lr_importances, rfe_results

def feature_pairs_visualization(df, feature_names):
    """تجسم‌سازی جفت ویژگی‌های مهم"""
    #
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df[feature_names], df['target'])
    top_features = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    })
    top_features = top_features.sort_values('Importance', ascending=False).reset_index(drop=True)
    top5_features = top_features.head(5)['Feature'].tolist()
    
    
    plt.figure(figsize=(20, 16))
    pair_idx = 1
    for i in range(len(top5_features)):
        for j in range(i+1, len(top5_features)):
            plt.subplot(2, 5, pair_idx)
            sns.scatterplot(x=top5_features[i], y=top5_features[j], 
                          hue='diagnosis', data=df, alpha=0.7)
            plt.title(f'{top5_features[i]} vs {top5_features[j]}')
            pair_idx += 1
    
    plt.tight_layout()
    plt.savefig('feature_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()

def write_report(high_corr, f_scores, mi_scores, rf_importances, lr_importances, rfe_results):
    """نوشتن گزارش تحلیل ویژگی‌ها"""
    with open('feature_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("# گزارش تحلیل ویژگی‌های مجموعه داده سرطان سینه\n\n")
        
        f.write("## همبستگی ویژگی‌ها\n\n")
        f.write("ویژگی‌های با همبستگی بالا (> 0.9):\n\n")
        f.write(high_corr.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## تحلیل اهمیت تک متغیره\n\n")
        f.write("### برترین ویژگی‌ها بر اساس ANOVA F-Score\n\n")
        f.write(f_scores.head(10).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### برترین ویژگی‌ها بر اساس Mutual Information\n\n")
        f.write(mi_scores.head(10).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## تحلیل اهمیت بر اساس مدل‌ها\n\n")
        f.write("### برترین ویژگی‌ها بر اساس Random Forest\n\n")
        f.write(rf_importances.head(10).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### برترین ویژگی‌ها بر اساس Logistic Regression\n\n")
        f.write(lr_importances.head(10).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### ویژگی‌های انتخاب شده توسط Recursive Feature Elimination\n\n")
        f.write(rfe_results[rfe_results['Selected'] == True].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## نتیجه‌گیری\n\n")
        
        
        common_top10 = set(f_scores.head(10)['Feature']) & set(mi_scores.head(10)['Feature']) & \
                      set(rf_importances.head(10)['Feature']) & set(lr_importances.head(10)['Feature'])
        
        f.write("### ویژگی‌های مشترک در تمام روش‌های تحلیل (10 ویژگی برتر):\n\n")
        for feature in common_top10:
            f.write(f"- {feature}\n")
        f.write("\n")
        
        f.write("### توصیه‌ها:\n\n")
        f.write("1. ویژگی‌های با همبستگی بالا می‌توانند به یک ویژگی کاهش یابند تا از افزایش بی‌مورد پیچیدگی مدل جلوگیری شود.\n")
        f.write("2. استفاده از ویژگی‌های مشترک در همه روش‌ها برای ایجاد مدل‌هایی با پیچیدگی کمتر و قابلیت تفسیر بیشتر.\n")
        f.write("3. تمرکز بر روی ویژگی‌هایی که به صورت مستقیم به اندازه و شکل سلول‌ها مرتبط هستند، زیرا اهمیت بالاتری در تشخیص دارند.\n")

def main():
    """اجرای تمام تحلیل‌ها و تولید گزارش"""
    print("بارگذاری داده‌ها...")
    df, X, y, feature_names = load_data()
    
    print("تحلیل همبستگی ویژگی‌ها...")
    high_corr = feature_correlation(df, feature_names)
    
    print("تجسم‌سازی توزیع ویژگی‌ها...")
    feature_distributions(df, feature_names)
    
    print("تحلیل اهمیت تک متغیره ویژگی‌ها...")
    f_scores, mi_scores = univariate_feature_importance(X, y, feature_names)
    
    print("تحلیل اهمیت ویژگی‌ها بر اساس مدل‌ها...")
    rf_importances, lr_importances, rfe_results = model_based_feature_importance(X, y, feature_names)
    
    print("تجسم‌سازی جفت ویژگی‌های مهم...")
    feature_pairs_visualization(df, feature_names)
    
    print("نوشتن گزارش تحلیل...")
    write_report(high_corr, f_scores, mi_scores, rf_importances, lr_importances, rfe_results)
    
    print("تحلیل کامل شد. فایل‌های خروجی در دایرکتوری جاری ذخیره شدند.")

if __name__ == "__main__":
    main() 
