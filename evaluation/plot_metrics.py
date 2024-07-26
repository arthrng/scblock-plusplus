from matplotlib import pyplot as plt
import json
import os

def create_plots(dataset_name):
    def plot(all_foc, metrics, labels):
        """
        Plot metrics for different blocking methods.

        Parameters:
        - all_foc: Dictionary with blocking methods as keys and lists of FOC values as values.
        - metrics: Dictionary with metric names as keys and their values as lists.
        - labels: Dictionary with blocking methods as keys and their labels as values.
        """
        # Define colors and markers for different blocking methods
        colors = ['#8B0000', '#006400', '#00008B', '#FF8C00', '#800080', '#008B8B']
        markers = ['o', 's', '^', 'D', 'x', '*']

        # Initialize plot
        plt.figure(figsize=(10, 6))
        plt.grid(True)

        lines = []

        # Plot each blocking method
        for i, (blocking_method, foc) in enumerate(all_foc.items()):
            line, = plt.plot(foc, metrics[blocking_method], color=colors[i % len(colors)],
                             linewidth=2, marker=markers[i % len(markers)], markersize=10,
                             label=labels[blocking_method])
            lines.append(line)

        # Set plot labels
        plt.xlabel('k', fontsize=18)
        plt.ylabel('F1 score' if 'F1 score' in metrics else metrics['label'], fontsize=18)
        plt.xlim(left=-0.004)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # Add legend
        plt.legend(handles=lines)

        # Ensure the output directory exists
        output_dir = f'./figures/{dataset_name}/'
        os.makedirs(output_dir, exist_ok=True)

        # Save and show the plot
        plt.savefig(os.path.join(output_dir, metrics['filename']), bbox_inches='tight')
        plt.show()

    # Define blocking methods and their labels
    blocking_methods = [
        'scblock',
        'scblock-with-flooding-0.005',
        'scblock-with-adaflood',
        'barlowtwins',
        'sbert'
    ]
    blocking_methods_labels = {
        'scblock': 'SC-Block',
        'scblock-with-flooding-0.005': 'SC-Block with flooding',
        'scblock-with-adaflood': 'SC-Block with AdaFlood',
        'barlowtwins': 'Barlow Twins',
        'sbert': 'Sentence RoBERTa'
    }

    # Initialize dictionaries to store metrics
    all_precision_scores = {}
    all_recall_scores = {}
    all_f1_scores = {}
    all_foc = {}

    # Process each blocking method
    for blocking_method in blocking_methods:
        # Load results from file
        file_path = f'./results/{dataset_name}/aggregated_results_{blocking_method}.json'
        try:
            with open(file_path, "r") as f:
                results = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"File not found: {file_path}. Skipping.")
            continue
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}. Skipping.")
            continue

        # Extract metrics from results
        precision_scores = []
        recall_scores = []
        f1_scores = []
        num_retrieved_evidences = []

        for result in results:
            precision_scores.append(result.get('precision_test_all', 0))
            recall_scores.append(result.get('recall_test_all', 0))
            f1_scores.append(result.get('f1_test_all', 0))
            num_retrieved_evidences.append(result.get('k', 0))

        # Store metrics
        all_precision_scores[blocking_method] = precision_scores
        all_recall_scores[blocking_method] = recall_scores
        all_f1_scores[blocking_method] = f1_scores
        all_foc[blocking_method] = num_retrieved_evidences

    # Define metrics for plotting
    all_metrics = {
        'F1 score': {'data': all_f1_scores, 'label': 'F1 score', 'filename': 'f1_score.png'},
        'Pair completeness': {'data': all_recall_scores, 'label': 'Pair completeness', 'filename': 'pair_completeness.png'},
        'Pair quality': {'data': all_precision_scores, 'label': 'Pair quality', 'filename': 'pair_quality.png'}
    }

    # Plot all metrics
    for metric_name, metric_info in all_metrics.items():
        plot(all_foc, metric_info['data'], blocking_methods_labels)

if __name__ == '__main__':
    dataset_names = ['walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']
    for dataset_name in dataset_names:
        create_plots(dataset_name)
