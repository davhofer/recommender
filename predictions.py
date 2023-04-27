import csv


def get_predictions(data, probas, threshold=0.5):
    y_pred = (probas > threshold).float()
    return [(item[0], item[1], item[2], pred.item()) for item, pred in zip(data, y_pred)]


def write_outputs(data, model_description, output_dir):
    outputs_path = f"{output_dir}/{model_description}_outputs.csv"

    with open(outputs_path, 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['user_id', 'topic_id', 'was_interaction', 'interaction_prediction'])

        for row in data:
            csv_out.writerow(row)
