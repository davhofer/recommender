import csv


def get_predictions(data, probas):
    return [(item[0], item[1], item[2], proba.item()) for item, proba in zip(data, probas)]


def write_outputs(data, model_description, output_dir):
    outputs_path = f"{output_dir}/{model_description}_outputs.csv"

    with open(outputs_path, 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['user_id', 'topic_id', 'was_interaction', 'predict_proba'])

        for row in data:
            csv_out.writerow(row)
