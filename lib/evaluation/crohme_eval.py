# CROHME Evaluator for Mathematical Expression Graph Generation
# Evaluates relation prediction performance on handwritten mathematical formulas

import contextlib
import copy
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from util.misc import all_gather


class CROHMEEvaluator:
    """
    CROHME evaluator for mathematical expression graph generation.
    Focuses on relation prediction metrics.
    """

    def __init__(self, rel_categories: Dict[str, int], num_classes: int):
        """
        Args:
            rel_categories: Relation category dictionary {"Right": 1, "Above": 2, ...}
            num_classes: Number of symbol classes
        """
        assert isinstance(rel_categories, dict)
        
        self.rel_categories = rel_categories
        self.num_classes = num_classes
        self.id_to_rel = {v: k for k, v in rel_categories.items()}
        
        # Initialize storage
        self.predictions = {}
        self.img_ids = []
        
        # Statistics
        self.relation_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        self.total_correct_graphs = 0
        self.total_graphs = 0
        
        # Edge-level metrics (relation existence regardless of type)
        self.edge_tp = 0
        self.edge_fp = 0
        self.edge_fn = 0

    def update(self, predictions: Dict):
        """
        Update evaluator with predictions for a batch.
        
        Args:
            predictions: Dictionary mapping image_id to prediction dict
                {
                    image_id: {
                        'relations': [[subj, rel, obj], ...],  # predicted relations
                        'gt_relations': [[subj, rel, obj], ...],  # ground truth relations
                        'matched_indices': (pred_idx, gt_idx),  # Hungarian matching
                    }
                }
        """
        img_ids = list(predictions.keys())
        self.img_ids.extend(img_ids)
        
        for img_id, pred_data in predictions.items():
            self._evaluate_single_sample(
                pred_data['relations'],
                pred_data['gt_relations'],
                pred_data.get('matched_indices', None)
            )

    def _evaluate_single_sample(
        self,
        pred_relations: np.ndarray,
        gt_relations: np.ndarray,
        matched_indices: Tuple = None
    ):
        """
        Evaluate a single sample.
        
        Args:
            pred_relations: Predicted relations [N, 3] (subject, relation, object)
            gt_relations: Ground truth relations [M, 3]
            matched_indices: Optional tuple of (pred_idx, gt_idx) from Hungarian matching
        """
        self.total_graphs += 1
        
        # Convert to sets for easier comparison
        # If matched_indices are provided, remap predicted indices to match GT
        if matched_indices is not None:
            pred_indices, gt_indices = matched_indices
            gt_to_pred = {gt_idx: pred_idx for pred_idx, gt_idx in zip(pred_indices, gt_indices)}
            
            # Remap GT relations to predicted index space
            gt_rel_set = set()
            for subj, rel, obj in gt_relations:
                if subj in gt_to_pred and obj in gt_to_pred:
                    pred_subj = gt_to_pred[subj]
                    pred_obj = gt_to_pred[obj]
                    gt_rel_set.add((pred_subj, pred_obj, int(rel)))
        else:
            # Direct comparison without remapping
            gt_rel_set = set((int(s), int(o), int(r)) for s, o, r in gt_relations)
        
        # Predicted relations set
        pred_rel_set = set((int(s), int(o), int(r)) for s, o, r in pred_relations)
        
        # Compute metrics
        self._compute_relation_metrics(gt_rel_set, pred_rel_set)
        
        # Graph-level exact match
        if gt_rel_set == pred_rel_set:
            self.total_correct_graphs += 1

    def _compute_relation_metrics(self, gt_rels: set, pred_rels: set):
        """
        Compute TP, FP, FN for relations.
        
        Args:
            gt_rels: Set of ground truth relations (subj, obj, rel)
            pred_rels: Set of predicted relations (subj, obj, rel)
        """
        # Edge existence metrics (ignoring relation type)
        gt_edges = {(s, o) for s, o, r in gt_rels}
        pred_edges = {(s, o) for s, o, r in pred_rels}
        
        self.edge_tp += len(gt_edges & pred_edges)
        self.edge_fp += len(pred_edges - gt_edges)
        self.edge_fn += len(gt_edges - pred_edges)
        
        # Per-relation type metrics
        for rel_type in self.rel_categories.values():
            if rel_type == 0:  # Skip background
                continue
            
            gt_rels_type = {(s, o) for s, o, r in gt_rels if r == rel_type}
            pred_rels_type = {(s, o) for s, o, r in pred_rels if r == rel_type}
            
            tp = len(gt_rels_type & pred_rels_type)
            fp = len(pred_rels_type - gt_rels_type)
            fn = len(gt_rels_type - pred_rels_type)
            
            self.relation_stats[rel_type]['tp'] += tp
            self.relation_stats[rel_type]['fp'] += fp
            self.relation_stats[rel_type]['fn'] += fn

    def synchronize_between_processes(self):
        """Synchronize metrics across distributed processes."""
        # Gather all metrics from all processes
        all_edge_tp = all_gather(self.edge_tp)
        all_edge_fp = all_gather(self.edge_fp)
        all_edge_fn = all_gather(self.edge_fn)
        all_total_graphs = all_gather(self.total_graphs)
        all_correct_graphs = all_gather(self.total_correct_graphs)
        
        # Sum across processes
        self.edge_tp = sum(all_edge_tp)
        self.edge_fp = sum(all_edge_fp)
        self.edge_fn = sum(all_edge_fn)
        self.total_graphs = sum(all_total_graphs)
        self.total_correct_graphs = sum(all_correct_graphs)
        
        # Synchronize relation stats
        all_relation_stats = all_gather(dict(self.relation_stats))
        merged_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for stats_dict in all_relation_stats:
            for rel_type, counts in stats_dict.items():
                merged_stats[rel_type]['tp'] += counts['tp']
                merged_stats[rel_type]['fp'] += counts['fp']
                merged_stats[rel_type]['fn'] += counts['fn']
        
        self.relation_stats = merged_stats

    def accumulate(self):
        """Accumulate statistics (compatibility with COCO evaluator interface)."""
        # Statistics are accumulated in real-time, so this is a no-op
        pass

    def summarize(self):
        """Print evaluation summary."""
        metrics = self.compute()
        
        print("\n" + "="*70)
        print("CROHME Mathematical Expression Graph Evaluation Results")
        print("="*70)
        
        # Edge-level metrics
        print("\n[Edge Detection Metrics] (Relation existence, type-agnostic)")
        print(f"  Precision: {metrics['edge_precision']:.4f}")
        print(f"  Recall:    {metrics['edge_recall']:.4f}")
        print(f"  F1-score:  {metrics['edge_f1']:.4f}")
        
        # Relation-level macro metrics
        print("\n[Relation Classification Metrics] (Macro-averaged)")
        print(f"  Precision: {metrics['relation_precision']:.4f}")
        print(f"  Recall:    {metrics['relation_recall']:.4f}")
        print(f"  F1-score:  {metrics['relation_f1']:.4f}")
        
        # Per-relation metrics
        print("\n[Per-Relation Type Metrics]")
        rel_metrics = []
        for rel_type, rel_name in sorted(self.id_to_rel.items()):
            if rel_type == 0:  # Skip background
                continue
            if f'{rel_name}_f1' in metrics:
                rel_metrics.append((
                    rel_name,
                    metrics[f'{rel_name}_precision'],
                    metrics[f'{rel_name}_recall'],
                    metrics[f'{rel_name}_f1']
                ))
        
        # Sort by F1 score descending
        rel_metrics.sort(key=lambda x: x[3], reverse=True)
        
        for rel_name, prec, rec, f1 in rel_metrics:
            print(f"  {rel_name:20s}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
        
        # Graph-level metrics
        print("\n[Expression-Level Metrics]")
        print(f"  Graph Accuracy: {metrics['graph_accuracy']:.4f} "
              f"({self.total_correct_graphs}/{self.total_graphs})")
        print("="*70 + "\n")
        
        return metrics

    def compute(self) -> Dict[str, float]:
        """
        Compute final evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Edge-level metrics (relation existence)
        edge_precision = self.edge_tp / (self.edge_tp + self.edge_fp + 1e-10)
        edge_recall = self.edge_tp / (self.edge_tp + self.edge_fn + 1e-10)
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall + 1e-10)
        
        metrics['edge_precision'] = edge_precision
        metrics['edge_recall'] = edge_recall
        metrics['edge_f1'] = edge_f1
        
        # Per-relation metrics and macro-average
        all_tp, all_fp, all_fn = 0, 0, 0
        
        for rel_type, stats in self.relation_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            all_tp += tp
            all_fp += fp
            all_fn += fn
            
            rel_name = self.id_to_rel.get(rel_type, f"rel_{rel_type}")
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            metrics[f'{rel_name}_precision'] = precision
            metrics[f'{rel_name}_recall'] = recall
            metrics[f'{rel_name}_f1'] = f1
        
        # Macro-averaged relation metrics
        metrics['relation_precision'] = all_tp / (all_tp + all_fp + 1e-10)
        metrics['relation_recall'] = all_tp / (all_tp + all_fn + 1e-10)
        metrics['relation_f1'] = (
            2 * metrics['relation_precision'] * metrics['relation_recall'] / 
            (metrics['relation_precision'] + metrics['relation_recall'] + 1e-10)
        )
        
        # Graph-level accuracy
        metrics['graph_accuracy'] = self.total_correct_graphs / (self.total_graphs + 1e-10)
        
        return metrics

    def reset(self):
        """Reset all metrics."""
        self.predictions = {}
        self.img_ids = []
        self.relation_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        self.total_correct_graphs = 0
        self.total_graphs = 0
        self.edge_tp = 0
        self.edge_fp = 0
        self.edge_fn = 0