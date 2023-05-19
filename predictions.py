import csv


def get_predictions(data, probas, is_sequential=False, topic_ids=[]):
    if not is_sequential:
        return [(item[0], item[1], item[3], proba.item()) for item, proba in zip(data, probas)]

    result = []
    for user_id, (user_data, user_probas) in enumerate(zip(data, probas)):
        interaction_topic = user_data[2]
        for topic_id_idx, user_topic_proba in enumerate(user_probas):
            topic_id = topic_ids[topic_id_idx]
            result.append((user_id, topic_id, int(topic_id == interaction_topic), user_topic_proba.item()))

    return result


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

