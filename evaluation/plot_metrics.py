from matplotlib import pyplot as plt
import gzip
import json

def create_plots(dataset_name):
    def compute_foc(all_num_comparisons, total_num_comparisons):
        return [num_comparisons / total_num_comparisons for num_comparisons in all_num_comparisons]

    def plot(all_foc, metrics, labels):
        # Show the grid
        plt.grid(True)

        # Show the plot
        colors = ['#8B0000', '#006400', '#00008B', '#FF8C00', '#800080', '#008B8B']
        markers = ['o', 's', '^', 'D', 'x', '*']
        lines = []
        for i, (blocking_method, foc) in enumerate(all_foc.items()):
            print()
            line, = plt.plot(foc, metrics[1][blocking_method], color=colors[i], linewidth=2, marker=markers[i], markersize=10, label=labels[blocking_method])
            lines.append(line)

        # Change the labels
        plt.xlabel('k', fontsize=18)
        if metrics[0] == 'F1 score':
            plt.ylabel('F1* score', fontsize=18)
        else:
            plt.ylabel(metrics[0], fontsize=18)

        # Change xlim of the plotW
        plt.xlim(left=-0.004)

        # Change tick sizes
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # Plot legend
        plt.legend(handles=lines)

        # Save the figure
        plt.savefig(f'./figures/{dataset_name}/' + metrics[0], bbox_inches='tight')

        # Show the plot
        plt.show()

    # Define blocking methods
    blocking_methods = ['scblock', 
                        'scblock-with-flooding-0.015',
                        'scblock-with-adaflood',
                        #'lsh',
                        'barlowtwins',
                        'sbert']
    blocking_methods_labels = {'scblock': 'SC-Block',
                            'scblock-with-flooding-0.015': 'SC-Block with flooding',
                            'scblock-with-adaflood': 'SC-Block with AdaFlood',
                            #'lsh': 'LSH',
                            'barlowtwins': 'Barlow Twins',
                            'sbert': 'Sentence RoBERTa'}

    # Acquire metrics for each blocking method
    all_precision_scores = {}
    all_recall_scores = {}
    all_f1_scores = {}
    all_foc = {}

    if dataset_name == 'amazon-google' or dataset_name == 'walmart_amazon':
        total_num_comparisons = 1452600
        query_table_size = 900
    else:
        total_num_comparisons = 359500 
        query_table_size = 500

    for blocking_method in blocking_methods:
        # Import aggregated results
        with open(f'./results/{dataset_name}/aggregated_results_{blocking_method}.json', "r") as f:
            results = [json.loads(line) for line in f]

        # Acquire metrics and number of comparisons
        precision_scores = []
        recall_scores =[]
        f1_scores = []
        num_retrieved_evidences = []
        for result in results:
            precision_scores.append(result['precision_test_all'])
            recall_scores.append(result['recall_test_all'])
            f1_scores.append(result['f1_test_all'])
            num_retrieved_evidences.append(result['k']) #* query_table_size

        # Store metrics
        all_precision_scores[blocking_method] = precision_scores
        all_recall_scores[blocking_method] = recall_scores
        all_f1_scores[blocking_method] = f1_scores
        all_foc[blocking_method] = num_retrieved_evidences#compute_foc(num_retrieved_evidences, total_num_comparisons)

    # Plot
    all_metrics = {'F1 score': all_f1_scores,
                'Pair completeness': all_recall_scores,
                'Pair quality': all_precision_scores}

    for metric in all_metrics.items():
        plot(all_foc, metric, blocking_methods_labels)

if __name__ == '__main__':
    # with open(f'./results/wdcproducts80pair/results_sbert.json', "r") as f:
    #     results = [json.loads(line) for line in f]
        
        # split_counts = {"seen":0, "unseen":0, "right_seen":0,"left_seen":0,"all":0}
        # for result in results:
        #     if result['split'] == 'test':
        #         split_counts[result['seen']] += 1
        # print(split_counts)

    dataset_names = ['walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']
    for dataset_name in dataset_names:
        create_plots(dataset_name)

