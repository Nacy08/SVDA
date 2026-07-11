from typing import Dict, List, Sequence


def compute_classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    id2label: Dict[int, str],
) -> Dict[str, object]:
    ordered_ids = sorted(id2label.keys())
    ordered_labels = [id2label[idx] for idx in ordered_ids]
    id_to_position = {label_id: position for position, label_id in enumerate(ordered_ids)}

    matrix = [[0 for _ in ordered_ids] for _ in ordered_ids]
    correct = 0
    for gold, predicted in zip(labels, predictions):
        if gold == predicted:
            correct += 1
        if gold in id_to_position and predicted in id_to_position:
            matrix[id_to_position[gold]][id_to_position[predicted]] += 1

    accuracy = correct / len(labels) if labels else 0.0

    per_class_metrics: List[Dict[str, object]] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []
    for position, label_name in enumerate(ordered_labels):
        true_positive = matrix[position][position]
        predicted_positive = sum(row[position] for row in matrix)
        actual_positive = sum(matrix[position])

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        per_class_metrics.append(
            {
                "label_id": ordered_ids[position],
                "label": label_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(actual_positive),
            }
        )

    divisor = len(ordered_ids) or 1
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(sum(precision_values) / divisor),
        "macro_recall": float(sum(recall_values) / divisor),
        "macro_f1": float(sum(f1_values) / divisor),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": matrix,
        "labels": ordered_labels,
    }
