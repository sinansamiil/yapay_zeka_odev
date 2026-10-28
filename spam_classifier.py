"""
Spam Mail / SMS Tespiti ve Algoritma Karşılaştırma Projesi
Uçtan Uca Makine Öğrenmesi & Doğruluk Optimizasyonu
"""

import sys
import re
import string
import json
import warnings

# Windows terminali için UTF-8 yapılandırması
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# NLTK kütüphanesinden gerekli veri setlerini indir
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

def load_and_prepare_data(url):
    print("=" * 80)
    print("ADIM 1: VERİ SETİNİN YÜKLENMESİ VE İNCELENMESİ")
    print("=" * 80)
    print(f"[*] Veri seti indiriliyor: {url}")
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    print(f"[+] Toplam Örnek Sayısı: {len(df)}")
    
    # Sayısallaştırma (ham=0, spam=1)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    ham_count = (df['label_num'] == 0).sum()
    spam_count = (df['label_num'] == 1).sum()
    print(f"[+] Sınıf Dağılımı:")
    print(f"    - Normal (Ham - 0): {ham_count} (%{ham_count / len(df) * 100:.2f})")
    print(f"    - Spam (Spam - 1)  : {spam_count} (%{spam_count / len(df) * 100:.2f})")
    
    # Train / Test ayrımı (%80 Train, %20 Test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'],
        df['label_num'],
        test_size=0.20,
        random_state=42,
        stratify=df['label_num']
    )
    print(f"[+] Eğitim Kümesi (Train): {len(X_train)} örnek (%80)")
    print(f"[+] Test Kümesi (Test)    : {len(X_test)} örnek (%20)\n")
    return df, X_train, X_test, y_train, y_test

def train_baseline_models(X_train_raw, X_test_raw, y_train, y_test):
    print("=" * 80)
    print("ADIM 2: TEMEL (BASELINE) MODELLERİN EĞİTİMİ (CountVectorizer)")
    print("=" * 80)
    print("[*] Ham metinler standart CountVectorizer() ile vektörleştiriliyor...")
    
    count_vec = CountVectorizer()
    X_train_counts = count_vec.fit_transform(X_train_raw)
    X_test_counts = count_vec.transform(X_test_raw)
    
    baseline_models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Support Vector Machine (SVM)': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    baseline_accuracies = {}
    for name, model in baseline_models.items():
        model.fit(X_train_counts, y_train)
        preds = model.predict(X_test_counts)
        acc = accuracy_score(y_test, preds)
        baseline_accuracies[name] = acc
        print(f"    -> {name:<30}: İlk Doğruluk (Baseline Accuracy) = %{acc * 100:.2f}")
    
    print()
    return baseline_accuracies

def advanced_text_preprocessing(series):
    print("[*] Gelişmiş NLP Metin Ön İşleme Adımları Uygulanıyor:")
    print("    1. Küçük harfe dönüştürme")
    print("    2. Web linklerinin ' url ' ile etiketlenmesi")
    print("    3. Sayıların ' num ' ile etiketlenmesi")
    print("    4. Noktalama işaretlerinin kaldırılması")
    print("    5. NLTK İngilizce Stopwords temizliği ve PorterStemmer ile kök bulma")
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        # 1. Küçük harf
        text = str(text).lower()
        # 2. URL -> ' url '
        text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
        # 3. Sayılar -> ' num '
        text = re.sub(r'\b\d+\b', ' num ', text)
        # 4. Noktalama işaretleri
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 5. Kelime ayırma, stopwords ve stemming
        tokens = text.split()
        cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return ' '.join(cleaned_tokens)
    
    return series.apply(clean_text)

def optimize_and_evaluate(X_train_clean, X_test_clean, y_train, y_test, baseline_accs):
    print("\n" + "=" * 80)
    print("ADIM 3: DOĞRULUK ORANINI ARTIRMA YÖNTEMLERİ (TF-IDF & GRIDSEARCHCV)")
    print("=" * 80)
    
    print("[*] TF-IDF N-Gram (1, 2) Vektörleştirici uygulanıyor...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_tfidf = tfidf.transform(X_test_clean)
    print(f"[+] Çıkarılan TF-IDF Öznitelik Sayısı: {X_train_tfidf.shape[1]}")
    
    # Hiperparametre gridleri
    param_grids = {
        'Multinomial Naive Bayes': {
            'estimator': MultinomialNB(),
            'params': {'alpha': [0.05, 0.1, 0.5, 1.0]}
        },
        'Support Vector Machine (SVM)': {
            'estimator': SVC(probability=True, random_state=42),
            'params': {'C': [1, 5, 10], 'kernel': ['linear', 'rbf']}
        },
        'Logistic Regression': {
            'estimator': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'C': [1, 10, 50], 'class_weight': [None, 'balanced']}
        }
    }
    
    best_models = {}
    results = {}
    
    for name, config in param_grids.items():
        print(f"\n[*] {name} için GridSearchCV (5-Fold CV) çalıştırılıyor...")
        grid = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_train_tfidf, y_train)
        best_estimator = grid.best_estimator_
        best_models[name] = best_estimator
        
        # Test kümesi tahminleri
        y_pred = best_estimator.predict(X_test_tfidf)
        y_pred_proba = best_estimator.predict_proba(X_test_tfidf)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        results[name] = {
            'baseline_acc': baseline_accs[name],
            'optimized_acc': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'best_params': grid.best_params_,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr
        }
        
        print(f"[+] En İyi Parametreler : {grid.best_params_}")
        print(f"[+] Baseline Doğruluk   : %{baseline_accs[name] * 100:.2f}")
        print(f"[+] Optimize Doğruluk   : %{acc * 100:.2f} (Değişim: +%{(acc - baseline_accs[name]) * 100:.2f})")
        print(f"[+] F1-Skoru            : %{f1 * 100:.2f} | ROC-AUC: %{auc * 100:.2f}")

    # 4. Ensemble (Soft Voting Classifier)
    print("\n" + "-" * 80)
    print("[*] AŞAMA 2.4: ENSEMBLE (SOFT VOTING CLASSIFIER) OLUŞTURULUYOR")
    print("-" * 80)
    voting_clf = VotingClassifier(
        estimators=[
            ('nb', best_models['Multinomial Naive Bayes']),
            ('svm', best_models['Support Vector Machine (SVM)']),
            ('lr', best_models['Logistic Regression'])
        ],
        voting='soft'
    )
    voting_clf.fit(X_train_tfidf, y_train)
    y_pred_vote = voting_clf.predict(X_test_tfidf)
    y_pred_proba_vote = voting_clf.predict_proba(X_test_tfidf)[:, 1]
    
    acc_vote = accuracy_score(y_test, y_pred_vote)
    prec_vote = precision_score(y_test, y_pred_vote)
    rec_vote = recall_score(y_test, y_pred_vote)
    f1_vote = f1_score(y_test, y_pred_vote)
    auc_vote = roc_auc_score(y_test, y_pred_proba_vote)
    cm_vote = confusion_matrix(y_test, y_pred_vote)
    fpr_vote, tpr_vote, _ = roc_curve(y_test, y_pred_proba_vote)
    
    results['Soft Voting Ensemble'] = {
        'baseline_acc': None,
        'optimized_acc': acc_vote,
        'precision': prec_vote,
        'recall': rec_vote,
        'f1_score': f1_vote,
        'roc_auc': auc_vote,
        'best_params': {'estimators': 'NB + SVM + LR (Soft Voting)'},
        'y_pred': y_pred_vote,
        'y_pred_proba': y_pred_proba_vote,
        'confusion_matrix': cm_vote,
        'fpr': fpr_vote,
        'tpr': tpr_vote
    }
    
    print(f"[+] Soft Voting Ensemble Doğruluk: %{acc_vote * 100:.2f}")
    print(f"[+] Soft Voting Ensemble F1-Skoru: %{f1_vote * 100:.2f} | ROC-AUC: %{auc_vote * 100:.2f}")
    
    return results, best_models

def display_summary_table(results):
    print("\n" + "=" * 115)
    print("ADIM 4: DETAYLI ALGORİTMA KARŞILAŞTIRMA VE PERFORMANS TABLOSU")
    print("=" * 115)
    header = f"{'Algoritma':<30} | {'İlk Acc':<9} | {'Opt. Acc':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'En İyi Parametreler'}"
    print(header)
    print("-" * 115)
    
    best_algo = None
    highest_acc = -1.0
    
    for name, m in results.items():
        base_str = f"%{m['baseline_acc'] * 100:.2f}" if m['baseline_acc'] is not None else "  ---  "
        opt_str = f"%{m['optimized_acc'] * 100:.2f}"
        prec_str = f"%{m['precision'] * 100:.2f}"
        rec_str = f"%{m['recall'] * 100:.2f}"
        f1_str = f"%{m['f1_score'] * 100:.2f}"
        auc_str = f"%{m['roc_auc'] * 100:.2f}"
        params_str = str(m['best_params'])
        
        print(f"{name:<30} | {base_str:<9} | {opt_str:<9} | {prec_str:<9} | {rec_str:<9} | {f1_str:<9} | {auc_str:<9} | {params_str}")
        
        # En iyi bireysel algoritmayı tespit et
        if name != 'Soft Voting Ensemble' and m['optimized_acc'] > highest_acc:
            highest_acc = m['optimized_acc']
            best_algo = name
            
    print("=" * 115)
    print(f"\n[EN IYI BIREYSEL MODEL] : {best_algo} (Dogruluk: %{highest_acc * 100:.2f})")
    if 'Soft Voting Ensemble' in results:
        print(f"[ENSEMBLE MODELI]      : Soft Voting Ensemble (Dogruluk: %{results['Soft Voting Ensemble']['optimized_acc'] * 100:.2f})")
    print("=" * 115)
    return best_algo

def plot_visualizations(results):
    print("\n" + "=" * 80)
    print("ADIM 5: GRAFİKLERİN ÇİZİLMESİ VE YÜKSEK ÇÖZÜNÜRLÜKTE KAYDEDİLMESİ")
    print("=" * 80)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})
    
    # 1. Confusion Matrices (1x3 Subplot)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    target_names = ['Normal (0)', 'Spam (1)']
    
    algos = [
        'Multinomial Naive Bayes',
        'Support Vector Machine (SVM)',
        'Logistic Regression'
    ]
    
    colors = ['Blues', 'Greens', 'Purples']
    
    for idx, (algo, cmap) in enumerate(zip(algos, colors)):
        cm = results[algo]['confusion_matrix']
        ax = axes[idx]
        
        # Karmaşıklık matrisindeki her hücre için metin açıklaması oluştur
        # TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([
            [f"TN\n{tn}", f"FP\n{fp}"],
            [f"FN\n{fn}", f"TP\n{tp}"]
        ])
        
        sns.heatmap(
            cm,
            annot=labels,
            fmt='',
            cmap=cmap,
            cbar=False,
            ax=ax,
            xticklabels=target_names,
            yticklabels=target_names,
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(f"{algo}\n(Acc: %{results[algo]['optimized_acc']*100:.2f} | F1: %{results[algo]['f1_score']*100:.2f})", fontsize=12, pad=12, weight='bold')
        ax.set_xlabel('Tahmin Edilen Sınıf', fontsize=11, labelpad=8)
        ax.set_ylabel('Gerçek Sınıf', fontsize=11, labelpad=8)
    
    plt.suptitle("Optimize Edilmiş Makine Öğrenmesi Modelleri - Karmaşıklık Matrisleri (Confusion Matrices)", fontsize=15, weight='bold', y=1.03)
    plt.tight_layout()
    cm_path = 'confusion_matrices.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Confusion Matrix görseli kaydedildi: {cm_path} (dpi=300)")
    
    # 2. ROC Curves
    plt.figure(figsize=(10, 7.5), dpi=300)
    
    curve_styles = {
        'Multinomial Naive Bayes': {'color': '#1f77b4', 'lw': 2.2, 'ls': '-'},
        'Support Vector Machine (SVM)': {'color': '#2ca02c', 'lw': 2.5, 'ls': '-'},
        'Logistic Regression': {'color': '#9467bd', 'lw': 2.2, 'ls': '-'},
        'Soft Voting Ensemble': {'color': '#d62728', 'lw': 2.8, 'ls': '--'}
    }
    
    for name, style in curve_styles.items():
        if name in results:
            fpr = results[name]['fpr']
            tpr = results[name]['tpr']
            auc = results[name]['roc_auc']
            plt.plot(
                fpr,
                tpr,
                color=style['color'],
                lw=style['lw'],
                linestyle=style['ls'],
                label=f"{name} (AUC = {auc:.4f} / %{auc*100:.2f})"
            )
            
    # Rastgele Tahmin Çizgisi
    plt.plot([0, 1], [0, 1], color='#e74c3c', lw=1.8, linestyle=':', label='Rastgele Tahmin Sınırı (AUC = 0.5000)')
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)', fontsize=12, labelpad=10)
    plt.ylabel('True Positive Rate (Duyarlılık / Doğru Pozitif Oranı)', fontsize=12, labelpad=10)
    plt.title('Alıcı İşlem Karakteristiği (ROC) Eğrileri ve AUC Skorları Karşılaştırması', fontsize=14, weight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    roc_path = 'roc_curves.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] ROC Eğrisi görseli kaydedildi: {roc_path} (dpi=300)")

def save_metrics_to_json(results, df, best_algo):
    """Metrikleri akademik raporu otomatik doldurabilmek için JSON formatında kaydeder."""
    summary_data = {
        'total_samples': len(df),
        'ham_count': int((df['label_num'] == 0).sum()),
        'spam_count': int((df['label_num'] == 1).sum()),
        'best_algorithm': best_algo,
        'models': {}
    }
    for name, data in results.items():
        cm = data['confusion_matrix'].tolist()
        summary_data['models'][name] = {
            'baseline_acc': float(data['baseline_acc']) if data['baseline_acc'] is not None else None,
            'optimized_acc': float(data['optimized_acc']),
            'precision': float(data['precision']),
            'recall': float(data['recall']),
            'f1_score': float(data['f1_score']),
            'roc_auc': float(data['roc_auc']),
            'best_params': data['best_params'],
            'confusion_matrix': cm
        }
    with open('metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=4)
    print("[+] Model metrikleri 'metrics_summary.json' dosyasına kaydedildi.")

def main():
    df, X_train_raw, X_test_raw, y_train, y_test = load_and_prepare_data(DATA_URL)
    
    # Aşama 1: Baseline Modeller
    baseline_accs = train_baseline_models(X_train_raw, X_test_raw, y_train, y_test)
    
    # Aşama 2: Gelişmiş NLP Ön İşleme
    X_train_clean = advanced_text_preprocessing(X_train_raw)
    X_test_clean = advanced_text_preprocessing(X_test_raw)
    
    # Aşama 2 & 3: TF-IDF, GridSearchCV & Ensemble
    results, best_models = optimize_and_evaluate(X_train_clean, X_test_clean, y_train, y_test, baseline_accs)
    
    # Tablo ve En İyi Model
    best_algo = display_summary_table(results)
    
    # Görseller
    plot_visualizations(results)
    
    # Metrikleri kaydet
    save_metrics_to_json(results, df, best_algo)
    
    print("\n" + "=" * 80)
    print("[BASARILI] TUM EGITIM, TEST VE GORSELLESTIRME ASAMALARI BASARIYLA TAMAMLANDI!")
    print("=" * 80)

if __name__ == '__main__':
    main()
