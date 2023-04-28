import csv


def get_predictions(data, probas):
    return [(item[0], item[1], item[3], proba.item()) for item, proba in zip(data, probas)]


def write_outputs(data, loss_logs, model_description, output_dir):
    probas_output_path = f"{output_dir}/{model_description}_probas.csv"

    with open(probas_output_path, 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['user_id', 'topic_id', 'was_interaction', 'predict_proba'])

        for row in data:
            csv_out.writerow(row)

    loss_outputs_path = f"{output_dir}/{model_description}_loss.csv"

    with open(loss_outputs_path, 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['loss_value', 'iteration'])
        for idx, loss_value in enumerate(loss_logs, 1):
            csv_out.writerow((loss_value, idx))

