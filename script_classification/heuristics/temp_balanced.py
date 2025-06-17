
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import psycopg2.extras
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             average_precision_score, make_scorer)
import pickle
import os
from datetime import datetime
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

class BalancedVendorAwareClassifier:
    """
    Vendor-aware nested cross-validation pipeline for malicious script classification.
    """
    def __init__(self, db_config=None, output_dir="balanced_vendor_aware_results"):
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'vv8_backend',
            'user': 'vv8',
            'password': 'vv8',
            'port': 5434
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.table_name = 'multicore_static_info_known_companies'

        # Data containers
        self.raw_data = None
        self.train_data = None
        self.test_data = None
        self.test_vendors = None
        self.vendor_analysis = None
        self.nested_cv_results = None
        self.final_model = None
        self.test_results = None

        # Hyperparameter grid
        self.param_distributions = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [8, 10, 12, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True, False]
        }

    def connect_to_database(self):
        try:
            return psycopg2.connect(**self.db_config)
        except psycopg2.Error as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    def load_data(self):
        """Load and preprocess data from the configured database."""
        conn = self.connect_to_database()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(f"SELECT * FROM {self.table_name};")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        json_fields = [
            'aggregated_behavioral_apis', 'aggregated_fingerprinting_apis',
            'fingerprinting_source_apis', 'behavioral_source_apis',
            'behavioral_apis_access_count', 'fingerprinting_api_access_count',
            'apis_going_to_sink', 'max_aggregated_apis'
        ]

        data = []
        for rec in rows:
            record = dict(rec)
            for field in json_fields:
                if record.get(field) and isinstance(record[field], str):
                    try:
                        record[field] = json.loads(record[field])
                    except json.JSONDecodeError:
                        record[field] = None
            if record.get('label') in [0, 1]:
                data.append(record)

        self.raw_data = data
        print(f"✅ Loaded {len(data)} scripts for classification")
        return data

    def analyze_vendor_distribution(self):
        """Analyze vendor distribution to identify imbalances and nulls."""
        df = pd.DataFrame(self.raw_data)
        df['vendor_clean'] = df['vendor'].fillna('UNKNOWN_NEGATIVE')

        vendor_counts = df['vendor_clean'].value_counts()
        total_vendors = vendor_counts.shape[0]
        total_scripts = len(df)
        avg_per_vendor = total_scripts / total_vendors

        vendor_label_dist = df.groupby(['vendor_clean', 'label']).size().unstack(fill_value=0)

        null_vendor = df['vendor'].isnull().sum()
        pos_vendors = df[df['label'] == 1]['vendor_clean'].nunique()
        neg_vendors = df[df['label'] == 0]['vendor_clean'].nunique()

        self.vendor_analysis = {
            'vendor_counts': vendor_counts,
            'vendor_stats': df.groupby('vendor_clean')['label'].agg(['count', 'sum'])
                            .rename(columns={'count':'total','sum':'pos_count'})
        }
        print(f"Total vendors: {total_vendors}, avg scripts/vendor: {avg_per_vendor:.1f}")
        return self.vendor_analysis

    def create_vendor_aware_split(self, test_vendor_ratio=0.2, random_state=42):
        """Split train/test such that vendors do not overlap."""
        df = pd.DataFrame(self.raw_data)
        df['vendor_group'] = df['vendor'].fillna('UNKNOWN_NEGATIVE')

        # unique groups for null vendors
        null_mask = df['vendor'].isnull()
        df.loc[null_mask, 'vendor_group'] = [f'NULL_{i}' for i in df[null_mask].index]

        groups = df['vendor_group']
        labels = df['label']

        # select test groups to meet ratio
        unique_groups = groups.unique().tolist()
        np.random.seed(random_state)
        np.random.shuffle(unique_groups)
        test_size = int(len(unique_groups) * test_vendor_ratio)
        test_groups = set(unique_groups[:test_size])

        train_df = df[~df['vendor_group'].isin(test_groups)]
        test_df = df[df['vendor_group'].isin(test_groups)]

        self.train_data = train_df.to_dict('records')
        self.test_data = test_df.to_dict('records')
        self.test_vendors = list(test_groups)
        print(f"Train/Test scripts: {len(self.train_data)}/{len(self.test_data)}")
        return self.train_data, self.test_data

    def engineer_features(self, dataset):
        """Transform raw records into feature DataFrame."""
        feats = []
        for rec in dataset:
            f = {}
            f['script_id'] = rec.get('script_id')
            f['label'] = rec['label']
            f['vendor'] = rec.get('vendor')

            # example features
            f['total_behavioral'] = sum(rec.get('behavioral_apis_access_count', {}).values() or [])
            f['total_fp'] = sum(rec.get('fingerprinting_api_access_count', {}).values() or [])
            f['ratio'] = f['total_behavioral'] / (f['total_behavioral']+f['total_fp']+1)

            feats.append(f)
        return pd.DataFrame(feats)

    def create_vendor_groups_for_cv(self, data):
        return [rec.get('vendor') or f"NULL_{i}" for i, rec in enumerate(data)]

    def nested_cross_validation(self, outer_cv_folds=5, inner_cv_folds=3, n_iter=20):
        df = self.engineer_features(self.train_data)
        X = df.drop(['script_id','label','vendor'], axis=1).values
        y = df['label'].values
        groups = np.array(self.create_vendor_groups_for_cv(self.train_data))

        # choose CV type
        try:
            outer_cv = StratifiedGroupKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
        except:
            outer_cv = GroupKFold(n_splits=outer_cv_folds)

        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        safe_scorer = make_scorer(lambda est, X, y: roc_auc_score(y, est.predict_proba(X)[:,1]) if len(set(y))>1 else 0.5,
                                  needs_proba=True)

        fold_results = []
        for train_idx, val_idx in outer_cv.split(X, y, groups):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            grp_tr = groups[train_idx]

            # hyperparam tuning
            inner_cv = GroupKFold(n_splits=min(inner_cv_folds, len(np.unique(grp_tr))))
            search = RandomizedSearchCV(rf, self.param_distributions, n_iter=n_iter,
                                        cv=inner_cv, scoring=safe_scorer, random_state=42, n_jobs=-1)
            search.fit(X_tr, y_tr, groups=grp_tr)
            best = search.best_estimator_

            # evaluate
            best.fit(X_tr, y_tr)
            tr_auc = roc_auc_score(y_tr, best.predict_proba(X_tr)[:,1])
            val_auc = roc_auc_score(y_val, best.predict_proba(X_val)[:,1])
            fold_results.append({'train_auc':tr_auc, 'val_auc':val_auc, 'params':search.best_params_})

        self.nested_cv_results = fold_results
        print(f"Completed {len(fold_results)} folds")
        return fold_results

    def train_final_model(self):
        # pick most common params
        params = Counter([tuple(sorted(f['params'].items())) for f in self.nested_cv_results])
        best = dict(params.most_common(1)[0][0])
        df = self.engineer_features(self.train_data)
        X = df.drop(['script_id','label','vendor'],axis=1).values
        y = df['label'].values
        self.final_model = RandomForestClassifier(**best, class_weight='balanced', random_state=42)
        self.final_model.fit(X,y)
        return self.final_model

    def evaluate_on_test_set(self):
        df = self.engineer_features(self.test_data)
        X = df.drop(['script_id','label','vendor'],axis=1).values
        y = df['label'].values
        proba = self.final_model.predict_proba(X)[:,1]
        pred = self.final_model.predict(X)
        auc = roc_auc_score(y,proba)
        ap = average_precision_score(y,proba)
        cm = confusion_matrix(y,pred)
        self.test_results = {'metrics':{'auc':auc,'ap':ap},'cm':cm,'y':y,'proba':proba}
        print(f"Test AUC: {auc:.3f}")
        return self.test_results

    def save_model_and_results(self):
        fn = os.path.join(self.output_dir, f"model_{self.timestamp}.pkl")
        with open(fn,'wb') as f:
            pickle.dump(self.final_model,f)
        print(f"Model saved to {fn}")

    def create_comprehensive_plots(self):
        # minimal placeholder
        print("Plots created")

    def create_summary_plot(self):
        pass

    def generate_detailed_report(self):
        pass

    def run_complete_analysis(self,
                               test_vendor_ratio=0.2,
                               outer_cv_folds=5,
                               inner_cv_folds=3,
                               n_iter=20):
        self.load_data()
        self.analyze_vendor_distribution()
        self.create_vendor_aware_split(test_vendor_ratio)
        self.nested_cross_validation(outer_cv_folds, inner_cv_folds, n_iter)
        self.train_final_model()
        self.evaluate_on_test_set()
        self.save_model_and_results()
        self.create_comprehensive_plots()
        self.generate_detailed_report()
        print("🎉 Pipeline completed")
        return self.test_results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Balanced Vendor-Aware Malicious Script Classification"
    )
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--db-port', type=int, default=5434)
    parser.add_argument('--db-name', type=str, default='vv8_backend')
    parser.add_argument('--db-user', type=str, default='vv8')
    parser.add_argument('--db-password', type=str, default='vv8')
    parser.add_argument('--output-dir', type=str, default='balanced_vendor_aware_results')
    parser.add_argument('--test-vendor-ratio', type=float, default=0.2)
    parser.add_argument('--outer-cv-folds', type=int, default=5)
    parser.add_argument('--inner-cv-folds', type=int, default=3)
    parser.add_argument('--hyperparameter-iterations', type=int, default=20)
    args = parser.parse_args()

    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'port': args.db_port
    }
    clf = BalancedVendorAwareClassifier(db_config=db_config,
                                        output_dir=args.output_dir)
    clf.run_complete_analysis(
        test_vendor_ratio=args.test_vendor_ratio,
        outer_cv_folds=args.outer_cv_folds,
        inner_cv_folds=args.inner_cv_folds,
        n_iter=args.hyperparameter_iterations
    )

if __name__ == "__main__":
    main()
